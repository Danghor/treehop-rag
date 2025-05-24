import glob
import os
import time
from typing import Union, Iterable
import faiss

import numpy as np
import torch

import src.data
import src.index
import src.normalize_text
import src.slurm
from passage_retrieval import search_passage_results
from src.utils import DEVICE


class Retriever:
    def __init__(self,
        model_name_or_path: str,
        passages: str | list,
        passage_embeddings: str | None = None,
        faiss_index: str | None = None,
        no_fp16=False,
        save_or_load_index=False,
        indexing_batch_size=1000000,
        lowercase=False,
        normalize_text=True,
        per_gpu_batch_size=64,
        query_maxlength=512,
        projection_size=768,
        n_subquantizers=0,
        n_bits=8,
        index_device="cpu"
    ):
        self.model_name_or_path = model_name_or_path
        self.passages = passages
        self.passage_embeddings = passage_embeddings
        self.faiss_index = faiss_index
        if passage_embeddings is None and faiss_index is None:
            raise ValueError("Either passage_embeddings or faiss_index must be provided")

        self.no_fp16 = no_fp16
        self.save_or_load_index = save_or_load_index
        self.indexing_batch_size = indexing_batch_size
        self.lowercase = lowercase
        self.normalize_text = normalize_text
        self.per_gpu_batch_size = per_gpu_batch_size
        self.query_maxlength = query_maxlength
        self.projection_size = projection_size
        self.n_subquantizers = n_subquantizers
        self.n_bits = n_bits
        self.index_device = index_device

        self.setup_retriever()

    @torch.no_grad
    def embed_queries(self, queries):
        embeddings, batch_query = [], []
        for k, q in enumerate(queries):
            if self.lowercase:
                q = q.lower()
            if self.normalize_text:
                q = src.normalize_text.normalize(q)
            batch_query.append(q)

            if len(batch_query) == self.per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = self.tokenizer.batch_encode_plus(
                    batch_query,
                    return_tensors="pt",
                    max_length=self.query_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.to(DEVICE) for k, v in encoded_batch.items()}
                output = self.model(**encoded_batch)
                embeddings.append(output.to(self.index_device))

                batch_query.clear()
                # getattr(torch, DEVICE).empty_cache()

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def index_encoded_data(self, input_paths, indexing_batch_size):
        input_paths = sorted(glob.glob(input_paths))
        all_ids = []
        all_embeddings = []
        start_idx = 0

        print(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        for i, file_path in enumerate(input_paths):
            data = src.data.load_regular_data(file_path)
            if isinstance(data, tuple):
                ids, embeddings = data
            else:
                embeddings = data
                ids = list(range(start_idx, start_idx + len(embeddings)))
                start_idx += len(embeddings)

            all_ids.extend(ids)
            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)
        while all_embeddings.shape[0] > 0:
            all_embeddings, all_ids = self._batch_add_embeddings(
                all_embeddings, all_ids, indexing_batch_size
            )

        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

    def _batch_add_embeddings(self, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_to_add = ids[:end_idx]
        embeddings_to_add = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        self.indexer.index_data(ids_to_add, embeddings_to_add)
        return embeddings, ids

    def add_passages(self, passages, top_passages_and_scores):
        # add passages to original data
        lst_docs = []
        for passage_ids, scores in top_passages_and_scores:
            lst_doc = []
            for p_id, score in zip(passage_ids, scores):
                doc = passages[p_id].copy()
                doc["score"] = float(score)
                lst_doc.append(doc)

            lst_docs.append(lst_doc)

        return lst_docs

    def setup_retriever(self):
        print(f"Loading model from: {self.model_name_or_path}")
        self.model, self.tokenizer, _ = src.load_retriever(
            self.model_name_or_path, self.index_device
        )
        self.model.eval()
        self.model = self.model.to(DEVICE)
        if not self.no_fp16:
            self.model = self.model.half()

        self.indexer = src.index.Indexer(
            self.projection_size, self.n_subquantizers, self.n_bits
        )
        if getattr(self.index_device, "type", self.index_device).startswith("cuda"):
            if src.slurm.is_distributed():
                self.indexer.index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), src.slurm.local_rank, self.indexer.index
                )
            else:
                n_gpus = faiss.get_num_gpus()
                if n_gpus <= 0:
                    raise LookupError("Fiass cannot detect a gpu")

                if n_gpus == 1:
                    self.indexer.index = faiss.index_cpu_to_gpu(
                        faiss.StandardGpuResources(), 0, self.indexer.index
                    )
                else:
                    self.indexer.index = faiss.index_cpu_to_all_gpus(self.indexer.index)

        # index all passages
        input_paths = glob.glob(self.faiss_index or self.passage_embeddings)
        embeddings_dir = os.path.dirname(input_paths[0])
        if isinstance(self.faiss_index, str) and os.path.exists(self.faiss_index):
            self.indexer.deserialize_from(embeddings_dir)
        elif os.path.exists(self.passage_embeddings):
            self.index_encoded_data(self.passage_embeddings, self.indexing_batch_size)
            if self.save_or_load_index:
                if getattr(self.index_device, "type", self.index_device).startswith("cuda"):
                    self.indexer.index = faiss.index_gpu_to_cpu(self.indexer.index)
                self.indexer.serialize(embeddings_dir)
        else:
            raise FileNotFoundError(
                f"Passage embeddings not found at {self.passage_embeddings}, "
                f"or faiss index not found at {self.faiss_index}"
            )

        # load passages
        if isinstance(self.passages, str):
            self.passages = src.data.load_regular_data(self.passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print(f"{len(self.passages)} passages have been loaded")

    def get_passage_embedding_by_id(self, passage_ids):
        if isinstance(passage_ids, int):
            return self.indexer.index.reconstruct(passage_ids)

        passage_embedding = []
        for p_id in passage_ids:
            passage_embedding.append(self.indexer.index.reconstruct(p_id))

        return passage_embedding

    def search_passages(
        self,
        query: Union[str, Iterable[str], torch.Tensor, np.ndarray],
        top_n=10,
        index_batch_size=2048,
        return_query_embeddings=False,
        return_passage_embeddings=False
    ):
        queries = [query] if isinstance(query, str) else query

        query_embeddings = \
            queries \
            if isinstance(queries, (torch.Tensor, np.ndarray)) \
            else self.embed_queries(queries)

        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.cpu().numpy()

        # get top k results
        top_ids_and_scores = self.indexer.search_knn(
            query_vectors=query_embeddings,
            top_docs=top_n,
            index_batch_size=index_batch_size
        )

        lst_passages = self.add_passages(self.passage_id_map, top_ids_and_scores)

        lst_passage_embeddings = []
        lst_scores = []
        for passage_ids, scores in top_ids_and_scores:
            lst_scores.append(scores)
            passage_embeddings = self.get_passage_embedding_by_id(passage_ids)
            lst_passage_embeddings.append(passage_embeddings)

        score = np.vstack(lst_scores)
        if return_passage_embeddings:
            passage_embedding = (np.vstack(lst_passage_embeddings)
                                 .reshape((query_embeddings.shape[0], top_n, -1)))
        else:
            passage_embedding = None

        return search_passage_results(
            passage=lst_passages,
            score=score,
            query_embedding=query_embeddings if return_query_embeddings else None,
            passage_embedding=passage_embedding
        )
