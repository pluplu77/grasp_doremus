import argparse
import os
import random

from search_rdf.model import TextEmbeddingModel
from tqdm import tqdm
from universal_ml_utils.io import dump_jsonl
from universal_ml_utils.logging import setup_logging

from grasp.baselines.grisp.data import (
    GRISPMaterializedSample,
    GRISPSample,
    load_samples,
    prepare_selection,
    prepare_skeleton,
)
from grasp.configs import KgConfig
from grasp.manager import KgManager, load_kg_manager
from grasp.utils import get_available_knowledge_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize GRISP data for a given number of materializations per sample"
    )
    parser.add_argument(
        "knowledge_graph",
        type=str,
        choices=get_available_knowledge_graphs(),
        help="Knowledge graph to use",
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file containing data samples",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file to save materialized data",
    )
    parser.add_argument(
        "num_materializations",
        type=int,
        help="Number of materializations per sample",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Custom SPARQL endpoint for the knowledge graph",
    )
    parser.add_argument(
        "--skeleton-p",
        type=float,
        default=0.2,
        help="Augmentation probability for skeletons",
    )
    parser.add_argument(
        "--selection-p",
        type=float,
        default=0.2,
        help="Augmentation probability for selections",
    )
    parser.add_argument(
        "--val-output-file",
        type=str,
        default=None,
        help="Path to the output file to save validation materialized data",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Proportion of data to use for validation split (only if --val-output-file is provided)",
    )
    parser.add_argument(
        "--is-val",
        action="store_true",
        help="Whether the input file is a validation set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Text embedding model to use for embedding index",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists",
    )
    return parser.parse_args()


def materialize_sample(
    sample: GRISPSample,
    manager: KgManager,
    n: int,
    is_val: bool = False,
) -> GRISPMaterializedSample:
    if is_val:
        n = 1

    skeletons = [prepare_skeleton(sample, is_val) for _ in range(n)]

    if sample.has_placeholders:
        selections = [prepare_selection(sample, manager, is_val) for _ in range(n)]
    else:
        selections = []

    return GRISPMaterializedSample(
        skeletons=skeletons,
        selections=selections,
    )


def main(args: argparse.Namespace) -> None:
    # to show info from kg manager
    setup_logging("INFO")

    if os.path.exists(args.output_file) and not args.overwrite:
        raise FileExistsError(
            f"Output file {args.output_file} already exists. "
            "Use --overwrite to overwrite."
        )
    elif args.is_val and args.val_output_file is not None:
        raise ValueError("Cannot specify --val-output-file when --is-val is set.")

    random.seed(args.seed)
    desc = "validation" if args.is_val else "training"

    samples = load_samples([args.input_file])
    if args.val_output_file is not None:
        val_size = int(len(samples) * args.val_split)
        assert val_size > 0, "Validation split is too small."
        random.shuffle(samples)
        val_samples = samples[:val_size]
        samples = samples[val_size:]
    else:
        val_samples = None

    config = KgConfig(kg=args.knowledge_graph, endpoint=args.endpoint)
    manager = load_kg_manager(config)

    if config.has_embedding_index:
        model = TextEmbeddingModel(args.embedding_model)
        manager.set_embedding_model(model)

    materialized = []
    for sample in tqdm(samples, desc=f"Materializing {desc} samples"):
        assert isinstance(sample, GRISPSample), "Expected non-materialized GRISP sample"
        materialized_sample = materialize_sample(
            sample,
            manager,
            args.num_materializations,
            args.is_val,
        )
        materialized.append(materialized_sample.model_dump())

    dump_jsonl(materialized, args.output_file)

    if args.val_output_file is None or val_samples is None:
        return

    materialized = []
    for sample in tqdm(val_samples, desc="Materializing validation samples"):
        assert isinstance(sample, GRISPSample), "Expected non-materialized GRISP sample"

        materialized_sample = materialize_sample(sample, manager, 1, is_val=True)
        materialized.append(materialized_sample.model_dump())

    dump_jsonl(materialized, args.val_output_file)


if __name__ == "__main__":
    main(parse_args())
