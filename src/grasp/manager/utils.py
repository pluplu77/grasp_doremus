import os
import time
from pathlib import Path

from search_rdf import Data, EmbeddingIndex, KeywordIndex, FuzzyIndex
from universal_ml_utils.io import dump_json, load_json
from universal_ml_utils.logging import get_logger

from grasp.manager.cache import Cache
from grasp.manager.normalizer import Normalizer, WikidataPropertyNormalizer
from grasp.sparql.utils import find_longest_prefix, get_endpoint, load_qlever_prefixes
from grasp.sparql.types import ObjType
from grasp.utils import get_index_dir

SearchIndex = KeywordIndex | EmbeddingIndex | FuzzyIndex


def load_data(index_dir: str) -> Data:
    try:
        data = Data.load(os.path.join(index_dir, "data"))
    except Exception as e:
        raise ValueError(f"Failed to load index data from {index_dir}") from e

    return data


def load_index(
    index_dir: str,
    index_type: str,
) -> SearchIndex | None:
    logger = get_logger("KG INDEX LOADING")
    start = time.perf_counter()

    try:
        data = load_data(index_dir)
    except Exception as e:
        logger.warning(f"Failed to load data from {index_dir}: {e}")
        return None

    index_dir = os.path.join(index_dir, index_type)
    load_kwargs = {"data": data, "index_dir": index_dir}

    if index_type == "keyword":
        index_cls = KeywordIndex
    elif index_type == "fuzzy":
        index_cls = FuzzyIndex
    elif index_type == "embedding":
        index_cls = EmbeddingIndex
        load_kwargs["embedding_path"] = os.path.join(index_dir, "embedding.safetensors")
    else:
        raise ValueError(f"Unknown index type {index_type}")

    try:
        index = index_cls.load(**load_kwargs)
    except Exception as e:
        logger.warning(f"Failed to load {index_type} index from {index_dir}: {e}")
        return None

    end = time.perf_counter()

    logger.debug(f"Loading {index_type} index from {index_dir} took {end - start:.2f}s")

    return index


def load_entity_index(
    kg: str,
    index_type: str,
) -> SearchIndex | None:
    index_dir = os.path.join(get_index_dir(kg), "entities")
    return load_index(index_dir, index_type)


def load_property_index(
    kg: str,
    index_type: str,
) -> SearchIndex | None:
    index_dir = os.path.join(get_index_dir(kg), "properties")
    return load_index(index_dir, index_type)


def load_kg_normalizers(kg: str) -> tuple[Normalizer, Normalizer]:
    ent_normalizer = Normalizer()
    prop_normalizer = ent_normalizer
    if kg.startswith("wikidata"):
        prop_normalizer = WikidataPropertyNormalizer()
    return ent_normalizer, prop_normalizer


def load_kg_prefixes(kg: str, endpoint: str | None = None) -> dict[str, str]:
    kg_index_dir = get_index_dir(kg)
    prefix_file = Path(kg_index_dir, "prefixes.json")
    if prefix_file.exists():
        prefixes = load_json(prefix_file.as_posix())
    else:
        try:
            prefixes = load_qlever_prefixes(endpoint or get_endpoint(kg))
            # save for future use
            dump_json(prefixes, prefix_file.as_posix(), indent=2)
        except Exception:
            prefixes = {}

    common_prefixes = get_common_sparql_prefixes()
    values = set(prefixes.values())

    # add common prefixes that might not be covered by the
    # specified prefixes
    for short, long in common_prefixes.items():
        if short in prefixes or long in values:
            continue

        prefixes[short] = long

    return prefixes


def load_kg_info_sparqls(kg: str) -> tuple[str | None, str | None]:
    kg_index_dir = get_index_dir(kg)
    ent_info_file = Path(kg_index_dir, "entities", "info.sparql")
    prop_info_file = Path(kg_index_dir, "properties", "info.sparql")

    if ent_info_file.exists():
        ent_info = ent_info_file.read_text()
    else:
        ent_info = None

    if prop_info_file.exists():
        prop_info = prop_info_file.read_text()
    else:
        prop_info = None

    return ent_info, prop_info


def load_kg_caches(kg: str) -> tuple[Cache | None, Cache | None]:
    logger = get_logger("KG CACHE LOADING")
    kg_index_dir = get_index_dir(kg)

    start = time.perf_counter()
    ent_cache_dir = os.path.join(kg_index_dir, "entities", "info.cache", "db")
    try:
        ent_cache = Cache.load(ent_cache_dir)
        end = time.perf_counter()
        logger.debug(
            f"Loading entity cache from {ent_cache_dir} took {end - start:.2f}s",
        )
    except Exception as e:
        logger.warning(f"Failed to load entity cache from {ent_cache_dir}: {e}")
        ent_cache = None

    start = time.perf_counter()
    prop_cache_dir = os.path.join(kg_index_dir, "properties", "info.cache", "db")
    try:
        prop_cache = Cache.load(prop_cache_dir)
        end = time.perf_counter()
        logger.debug(
            f"Loading property cache from {prop_cache_dir} took {end - start:.2f}s",
        )
    except Exception as e:
        logger.warning(f"Failed to load property cache from {prop_cache_dir}: {e}")
        prop_cache = None

    return ent_cache, prop_cache


def load_kg_indices(
    kg: str,
    entities_type: str,
    properties_type: str,
) -> tuple[SearchIndex | None, SearchIndex | None]:
    ent_index = load_entity_index(kg, entities_type)
    prop_index = load_property_index(kg, properties_type)
    return ent_index, prop_index


def get_common_sparql_prefixes() -> dict[str, str]:
    return {
        "rdf": "<http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "<http://www.w3.org/2000/01/rdf-schema#",
        "owl": "<http://www.w3.org/2002/07/owl#",
        "xsd": "<http://www.w3.org/2001/XMLSchema#",
        "foaf": "<http://xmlns.com/foaf/0.1/",
        "skos": "<http://www.w3.org/2004/02/skos/core#",
        "dct": "<http://purl.org/dc/terms/",
        "dc": "<http://purl.org/dc/elements/1.1/",
        "prov": "<http://www.w3.org/ns/prov#",
        "schema": "<http://schema.org/",
        "geo": "<http://www.opengis.net/ont/geosparql#",
        "geosparql": "<http://www.opengis.net/ont/geosparql#",
        "geof": "<http://www.opengis.net/def/function/geosparql/",
        "gn": "<http://www.geonames.org/ontology#",
        "bd": "<http://www.bigdata.com/rdf#",
        "hint": "<http://www.bigdata.com/queryHints#",
        "wikibase": "<http://wikiba.se/ontology#",
        "qb": "<http://purl.org/linked-data/cube#",
        "void": "<http://rdfs.org/ns/void#",
    }


def find_obj_type_from_prefixes(
    iri: str,
    prefixes: dict[str, str],
    common_prefixes: dict[str, str],
) -> ObjType:
    # we have three cases:
    # 1. the IRI matches a common prefix but not a known prefix -> COMMON (e.g. rdf:type)
    # 2. the IRI matches a known prefix -> UNINDEXED (e.g. wd:Q42)
    # 3. the IRI matches neither -> UNKNOWN (e.g. fb:en.barack_obama in Wikidata)
    is_common = find_longest_prefix(iri, common_prefixes) is not None
    is_known = find_longest_prefix(iri, prefixes) is not None

    if is_common:
        return ObjType.COMMON
    elif is_known:
        return ObjType.UNINDEXED
    else:
        return ObjType.UNKNOWN


def is_embedding_index(index: SearchIndex) -> bool:
    return index.index_type == "embedding"


def describe_index(index: SearchIndex | str) -> tuple[str, str]:
    if isinstance(index, SearchIndex):
        index = index.index_type

    if index == "keyword":
        title = "Keyword index"
        desc = "Retrieves items by overlap between their label words and \
the query keywords. The query keywords can match label words exactly or \
as prefixes. No special query operators like AND/OR are supported."

    elif index == "fuzzy":
        title = "Fuzzy keyword index"
        desc = "Retrieves items by overlap between their label words and \
the query keywords. The query keywords must not match label words exactly, but \
some fuzziness is allowed. The longer a query keyword is, the more it can deviate \
from a label word and still be considered a match, though it will also contribute \
less to the overall score. No special query operators like AND/OR are supported."

    elif index == "embedding":
        title = "Embedding index"
        desc = "Retrieves items by cosine similarity between their \
label embeddings and the query embedding."

    else:
        raise ValueError(f"Unknown index type {index}")

    return title, desc
