# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse

from collections import namedtuple

import src.data, src.index, src.slurm, src.normalize_text
from Retriever import Retriever
from utils import load_file_jsonl, save_file_jsonl

os.environ["TOKENIZERS_PARALLELISM"] = "true"


multihop_search_passage_results = namedtuple(
    "multihop_search_passage_results",
    fields := ["passage", "tree_hop_graph"],
    defaults=(None,) * len(fields)
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--passages",
        type=str,
        help="Path to passages (.tsv file)"
    )
    parser.add_argument(
        "--passage_embeddings",
        type=str,
        default=None,
        help="Path to encoded passages in Numpy format"
    )
    parser.add_argument(
        "--faiss_index",
        type=str,
        default=None,
        help="Path to encoded passages in Faiss format"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="dir path to save embeddings"
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Id of the current shard"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards"
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=10,
        help="Number of documents to retrieve per questions"
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=64,
        help="Batch size for question encoding"
    )
    parser.add_argument(
        "--save_or_load_index",
        action="store_true",
        help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="path to directory containing model weights and config file"
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="inference in fp32")
    parser.add_argument(
        "--question_maxlength",
        type=int,
        default=512,
        help="Maximum number of tokens in a question"
    )
    parser.add_argument(
        "--indexing_batch_size",
        type=int,
        default=1000000,
        help="Batch size of the number of passages indexed"
    )
    parser.add_argument(
        "--projection_size",
        type=int,
        default=768
    )
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument(
        "--n_bits",
        type=int,
        default=8,
        help="Number of bits per subquantizer"
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="lowercase text before encoding"
    )
    parser.add_argument(
        "--normalize_text",
        action="store_true",
        help="normalize text"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_shards > 1:
        src.slurm.init_distributed_mode(args)

    # for debugging
    # data_paths = glob.glob(args.data)
    retriever = Retriever(
        model_name_or_path=args.model_name_or_path,
        passages=args.passages,
        passage_embeddings=args.passage_embeddings,
        faiss_index=args.faiss_index,
        no_fp16=args.no_fp16,
        save_or_load_index=args.save_or_load_index,
        indexing_batch_size=args.indexing_batch_size,
        lowercase=args.lowercase,
        normalize_text=args.normalize_text,
        per_gpu_batch_size=args.per_gpu_batch_size,
        query_maxlength=args.question_maxlength,
        projection_size=args.projection_size,
        n_subquantizers=args.n_subquantizers,
        n_bits=args.n_bits
    )

    query = args.query
    if os.path.exists(query):
        query = load_file_jsonl(query)

        shard_size = len(query) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = start_idx + shard_size
        if args.shard_id == args.num_shards - 1:
            end_idx = len(query)

        query = query[start_idx: end_idx]
        print("query length:", len(query))

    retrieved_documents = retriever.search_passages(query, args.n_docs).passage
    if isinstance(args.output, str):
        if isinstance(query, str):
            data = [{"question": query, "ctxs": retrieved_documents}]
        else:
            data = [{"question": question, "ctxs": ctx}
                    for question, ctx in zip(query, retrieved_documents)]
        save_file_jsonl(data, args.output)
    else:
        print(retrieved_documents)


if __name__ == "__main__":
    # --query "What is the occupation of Obama?" --passages ./wikipedia_data/psgs_w100.tsv --passage_embeddings "./wikipedia_data/embedding_contriever-msmarco/*" --model_name_or_path "facebook/contriever-msmarco" --output ./train_data/extractor_retrieve_wiki.jsonl
    main()