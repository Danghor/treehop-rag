from typing import Union, List

import numpy as np
import numpy.typing as npt
from tqdm.asyncio import tqdm

from Retriever import Retriever
from passage_retrieval import multihop_search_passage_results
from tree_hop import TreeHopModel, TreeHopGraph


class MultiHopRetriever(Retriever):
    def __init__(
        self,
        model_name_or_path: str,
        passages: str | list,
        tree_hop_model: TreeHopModel,
        passage_embeddings: str | npt.NDArray | None = None,
        faiss_index: str | None = None,
        no_fp16=True,
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
        super().__init__(
            model_name_or_path=model_name_or_path,
            passages=passages,
            faiss_index=faiss_index,
            passage_embeddings=passage_embeddings,
            no_fp16=no_fp16,
            save_or_load_index=save_or_load_index,
            indexing_batch_size=indexing_batch_size,
            lowercase=lowercase,
            normalize_text=normalize_text,
            per_gpu_batch_size=per_gpu_batch_size,
            query_maxlength=query_maxlength,
            projection_size=projection_size,
            n_subquantizers=n_subquantizers,
            n_bits=n_bits,
            index_device=index_device
        )

        self.tree_hop_model = tree_hop_model

    def multihop_search_passages(
        self,
        query: Union[List[str], str],
        n_hop: int,
        top_n=10,
        index_batch_size=10240,
        generate_batch_size=1024,
        show_progress=True,
        redundant_pruning=True,
        layerwise_top_pruning: Union[int, bool] = True,
        return_tree=False
    ):
        assert isinstance(n_hop, int) and n_hop > 0, "n_hop must be a positive integer"
        pbar = tqdm(
            total=n_hop,
            desc="Retrieving",
            postfix={"num_query": len(query)},
            leave=True,
            disable=not show_progress
        )

        query = [query] if isinstance(query, str) else query
        # start_time_search = time.time()
        search_result = self.search_passages(
            query,
            top_n=top_n,
            index_batch_size=index_batch_size,
            return_query_embeddings=True,
            return_passage_embeddings=True
        )
        # pbar.set_postfix({"num_query": len(query_embeddings),
        #                   "elapsed": time.time() - start_time_search})

        self.tree_hop_model.reset_query()

        pbar.set_description("Generating")
        # start_time_generate = time.time()
        q_emb = self.tree_hop_model.next_query(
            q_emb=search_result.query_embedding,
            ctx_embs=search_result.passage_embedding,
            batch_size=generate_batch_size
        )
        pbar.set_postfix({"num_query": len(q_emb),
                        #   "elapsed": time.time() - start_time_generate
                          })

        tree_hop_graphs = [TreeHopGraph(q, [psg], top_n=top_n,
                                        redundant_pruning=redundant_pruning,
                                        layerwise_top_pruning=layerwise_top_pruning)
                           for q, psg in zip(query, search_result.passage)]

        lst_results = [[graph.filtered_passages for graph in tree_hop_graphs]]
        pbar.update(1)

        if n_hop == 1:
            pbar.close()
            return multihop_search_passage_results(
                passage=lst_results,
                tree_hop_graph=tree_hop_graphs if return_tree else None,
                # query_similarity=query_sims if return_query_similarity else None
            )

        query_passage_masks = [graph.query_passage_mask for graph in tree_hop_graphs]
        ary_query_passage_masks = np.concatenate(query_passage_masks, axis=None)
        last_q_emb = q_emb[ary_query_passage_masks]

        for i_hop in range(1, n_hop):
            pbar.set_description("Retrieving")
            # start_time_search = time.time()
            search_result = self.search_passages(
                last_q_emb,
                top_n=top_n,
                index_batch_size=index_batch_size,
                return_passage_embeddings=True
            )

            pbar.set_description("Generating")
            query_passage_masks = [graph.query_passage_mask for graph in tree_hop_graphs]
            # start_time_generate = time.time()
            # assume embeddings reconstructed from faiss are normalized before stored
            # filter out semantically distant passage embedddings
            q_emb = self.tree_hop_model.next_query(
                ctx_embs=search_result.passage_embedding.reshape(-1, last_q_emb.shape[1]),
                query_passage_masks=query_passage_masks,
                batch_size=generate_batch_size
            )
            pbar.set_postfix({"num_query": len(q_emb),
                            #   "elapsed": time.time() - start_time_generate
                              })

            query_passage_masks = []
            lst_passages = []
            i_current = 0
            for i, graph in enumerate(tree_hop_graphs):
                num_query = graph.query_passage_mask.sum(axis=None)
                if num_query <= 0:
                    lst_passages.append([])
                    continue

                passage_layer = search_result.passage[i_current: i_current + num_query]
                graph.add_passage_layer(
                    passage_layer,
                    redundant_pruning=redundant_pruning,
                    layerwise_top_pruning=layerwise_top_pruning,
                )

                query_passage_masks.append(graph.query_passage_mask)
                lst_passages.append(graph.filtered_passages)
                i_current += num_query

            ary_query_passage_masks = np.concatenate(query_passage_masks, axis=None)
            last_q_emb = q_emb[ary_query_passage_masks]

            lst_results.append(lst_passages)
            pbar.update(1)
            pbar.set_postfix({"num_query": len(last_q_emb),
                            #   "elapsed": time.time() - start_time_search
                              })

        pbar.close()
        return multihop_search_passage_results(
            passage=lst_results,
            tree_hop_graph=tree_hop_graphs if return_tree else None
        )
