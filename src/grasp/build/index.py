import os
import time
from logging import Logger

import safetensors
from search_rdf import Data, EmbeddingIndex, KeywordIndex
from search_rdf.model import TextEmbeddingModel
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import flatten

from grasp.manager.utils import load_data
from grasp.utils import get_index_dir


def build_index(
    index_dir: str,
    index_type: str,
    logger: Logger,
    overwrite: bool = False,
    embedding_model: str | None = None,
    embedding_device: str | None = None,
    embedding_batch_size: int = 256,
    embedding_dim: int | None = None,
) -> None:
    data = load_data(index_dir)

    out_dir = os.path.join(index_dir, index_type)
    if os.path.exists(out_dir) and not overwrite:
        logger.info(
            f"Index of type {index_type} already exists at {out_dir}. Skipping build."
        )
        return

    os.makedirs(out_dir, exist_ok=True)
    start = time.perf_counter()
    logger.info(f"Building {index_type} index at {out_dir}")

    if index_type == "keyword":
        KeywordIndex.build(data, out_dir)

    elif index_type == "embedding":
        assert embedding_model is not None, (
            "Embedding model must be specified for embedding index"
        )
        embeddings_path = os.path.join(index_dir, "data", "embeddings.safetensors")

        generate_embeddings(
            data,
            embeddings_path,
            model=embedding_model,
            device=embedding_device,
            batch_size=embedding_batch_size,
            dim=embedding_dim,
        )

        EmbeddingIndex.build(
            data,
            embeddings_path,
            out_dir,
        )
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    end = time.perf_counter()
    logger.info(f"Index build took {end - start:.2f} seconds")


def generate_embeddings(
    data: Data,
    out_path: str,
    model: str,
    device: str | None = None,
    batch_size: int = 256,
    dim: int | None = None,
) -> None:
    embedding_model = TextEmbeddingModel(model, device)

    texts = list(flatten(fields for _, fields in data))
    embeddings = embedding_model.embed(texts, dim, batch_size, show_progress=True)

    safetensors.serialize_file(
        {"embeddings": embeddings},
        filename=out_path,
        metadata={"model": model},
    )


def build_indices(
    kg: str,
    entities_type: str,
    properties_type: str,
    overwrite: bool = False,
    log_level: str | int | None = None,
    embedding_model: str | None = None,
    embedding_device: str | None = None,
    embedding_batch_size: int = 256,
    embedding_dim: int | None = None,
) -> None:
    logger = get_logger("GRASP INDEX", log_level)

    index_dir = get_index_dir(kg)

    # entities
    entities_dir = os.path.join(index_dir, "entities")
    build_index(
        entities_dir,
        entities_type,
        logger,
        overwrite,
        embedding_model,
        embedding_device,
        embedding_batch_size,
        embedding_dim,
    )

    # properties
    properties_dir = os.path.join(index_dir, "properties")
    build_index(
        properties_dir,
        properties_type,
        logger,
        overwrite,
        embedding_model,
        embedding_device,
        embedding_batch_size,
        embedding_dim,
    )
