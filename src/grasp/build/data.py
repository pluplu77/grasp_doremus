import csv
import json
import os
from logging import Logger
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote_plus

import ijson
import requests
from search_rdf import Data
from tqdm import tqdm
from universal_ml_utils.io import dump_jsonl, dump_text, load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.manager.utils import (
    get_common_sparql_prefixes,
    load_kg_prefixes,
)
from grasp.sparql.utils import (
    find_longest_prefix,
    get_endpoint,
    load_entity_index_sparql,
    load_iri_and_literal_parser,
    load_property_index_sparql,
    parse_into_binding,
)
from grasp.utils import get_index_dir, ordered_unique

Binding = tuple[str, str, list[str]]  # (id, value, tags)


def download_data(
    out_dir: str,
    endpoint: str,
    sparql: str,
    logger: Logger,
    prefixes: dict[str, str],
    params: dict[str, str] | None = None,
    add_id_as_label: None | str = None,
    overwrite: bool = False,
    format: str = "json",
) -> None:
    data_file = Path(out_dir, "data.jsonl")
    if data_file.exists() and not overwrite:
        logger.info(f"Data already exists at {data_file}, skipping download")
        return

    logger.info(
        f"Downloading data to {data_file} from {endpoint} "
        f"with parameters {params or {}} and SPARQL:\n{sparql}"
    )

    if format == "json":
        bindings = stream_json(endpoint, sparql, params)
    elif format == "csv":
        bindings = stream_csv(endpoint, sparql, params)
    elif format == "tsv":
        bindings = stream_tsv(endpoint, sparql, logger, params)
    else:
        raise ValueError(
            f"Unknown format: {format!r}, expected 'json', 'csv', or 'tsv'"
        )

    dump_jsonl(
        prepare_items(bindings, prefixes, add_id_as_label, logger),
        data_file.as_posix(),
    )


def build_data_and_mapping(
    index_dir: str,
    logger: Logger,
    overwrite: bool = False,
) -> None:
    data_file = Path(index_dir, "data.jsonl")
    data_dir = Path(index_dir, "data")
    if not data_dir.exists() or overwrite:
        # build index data
        logger.info(f"Building data at {data_dir}")
        Data.build_from_jsonl(data_file.as_posix(), data_dir.as_posix())
    else:
        logger.info(f"Data already exists at {data_dir}, skipping build")


def get_data(
    kg: str,
    endpoint: str | None = None,
    entity_query: str | None = None,
    property_query: str | None = None,
    query_params: dict[str, str] | None = None,
    overwrite: bool = False,
    add_id_as_label: str | None = None,
    log_level: str | int | None = None,
    format: str = "json",
) -> None:
    logger = get_logger("GRASP DATA", log_level)

    if endpoint is None:
        endpoint = get_endpoint(kg)
        logger.info(
            f"Using endpoint {endpoint} for {kg} because "
            "no endpoint is set in the config"
        )

    prefixes = get_common_sparql_prefixes()
    prefixes.update(load_kg_prefixes(kg))

    logger.info(f"Using prefixes:\n{json.dumps(prefixes, indent=2)}")

    kg_dir = get_index_dir(kg)

    # entities
    ent_dir = os.path.join(kg_dir, "entities")
    os.makedirs(ent_dir, exist_ok=True)
    ent_sparql = entity_query or load_entity_index_sparql()
    download_data(
        ent_dir,
        endpoint,
        ent_sparql,
        logger,
        prefixes,
        query_params,
        add_id_as_label,
        overwrite,
        format,
    )
    dump_text(ent_sparql, os.path.join(ent_dir, "index.sparql"))
    build_data_and_mapping(ent_dir, logger, overwrite)

    # properties
    prop_dir = os.path.join(kg_dir, "properties")
    os.makedirs(prop_dir, exist_ok=True)
    prop_sparql = property_query or load_property_index_sparql()
    download_data(
        prop_dir,
        endpoint,
        prop_sparql,
        logger,
        prefixes,
        query_params,
        add_id_as_label="always",  # for properties we also want to search via id
        overwrite=overwrite,
        format=format,
    )
    dump_text(prop_sparql, os.path.join(prop_dir, "index.sparql"))
    build_data_and_mapping(prop_dir, logger, overwrite)


def stream_json(
    endpoint: str,
    sparql: str,
    query_params: dict[str, str] | None = None,
) -> Iterator[Binding]:
    try:
        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/sparql-query",
            "User-Agent": "grasp-data-bot",
        }

        response = requests.post(
            endpoint,
            data=sparql,
            params=query_params,
            headers=headers,
            stream=True,
        )
        response.raise_for_status()
    except Exception as e:
        raise ValueError(f"Failed to stream SPARQL results as JSON: {e}") from e

    class _StreamReader:
        def __init__(self, response: requests.Response) -> None:
            self.stream = response.iter_content(chunk_size=None)

        def read(self, n: int) -> bytes:
            if n == 0:
                return b""
            return next(self.stream, b"")

    bindings = ijson.items(
        _StreamReader(response),
        "results.bindings.item",
        multiple_values=True,
    )

    for binding in bindings:
        id = binding["id"]["value"]
        value = binding["value"]["value"] if "value" in binding else ""

        tag_binding = binding.get("tag", binding.get("tags", None))
        if tag_binding is not None:
            tags = tag_binding["value"].split(",")
        else:
            tags = []

        yield id, value, tags


def stream_csv(
    endpoint: str,
    sparql: str,
    query_params: dict[str, str] | None = None,
) -> Iterator[Binding]:
    try:
        headers = {
            "Accept": "text/csv",
            "Content-Type": "application/sparql-query",
            "User-Agent": "grasp-data-bot",
        }

        response = requests.post(
            endpoint,
            data=sparql,
            params=query_params,
            headers=headers,
            stream=True,
        )
        response.raise_for_status()
    except Exception as e:
        raise ValueError(f"Failed to stream SPARQL results as CSV: {e}") from e

    lines = response.iter_lines(decode_unicode=True)
    reader = csv.DictReader(lines)

    for row in reader:
        id = row["id"]
        value = row.get("value", "") or ""

        tag_value = row.get("tag", row.get("tags", None))
        if tag_value:
            tags = tag_value.split(",")
        else:
            tags = []

        yield id, value, tags


def stream_tsv(
    endpoint: str,
    sparql: str,
    logger: Logger,
    query_params: dict[str, str] | None = None,
) -> Iterator[Binding]:
    try:
        headers = {
            "Accept": "text/tab-separated-values",
            "Content-Type": "application/sparql-query",
            "User-Agent": "grasp-data-bot",
        }

        response = requests.post(
            endpoint,
            data=sparql,
            params=query_params,
            headers=headers,
            stream=True,
        )
        response.raise_for_status()
    except Exception as e:
        raise ValueError(f"Failed to stream SPARQL results as TSV: {e}") from e

    parser = load_iri_and_literal_parser()
    lines = response.iter_lines(decode_unicode=True)

    # first line is header with ?var names
    header_line = next(lines, None)
    if header_line is None:
        return
    columns = [col.lstrip("?") for col in header_line.split("\t")]

    for line in lines:
        if not line:
            continue

        cells = line.split("\t")
        if len(cells) != len(columns):
            logger.warning(
                f"Skipping malformed line with {len(cells):,} values for "
                f"{len(columns):,} columns: {line[:100]}..."
            )
            continue

        row = dict(zip(columns, cells))

        id_cell = row.get("id", "")
        if not id_cell:
            continue

        id_binding = parse_into_binding(id_cell, parser)
        if id_binding is None:
            logger.warning(
                f"Failed to parse id {id_cell} as IRI or literal, skipping row"
            )
            continue

        id = id_binding.value

        value_cell = row.get("value", "")
        if value_cell:
            value_binding = parse_into_binding(value_cell, parser)
            if value_binding is None or value_binding.typ != "literal":
                logger.warning(
                    f"Failed to parse value {value_cell} as literal, skipping row"
                )
                continue

            value = value_binding.value
        else:
            value = ""

        tag_cell = row.get("tag", row.get("tags", ""))
        if tag_cell:
            tag_binding = parse_into_binding(tag_cell, parser)
            if tag_binding is None or tag_binding.typ != "literal":
                logger.warning(
                    f"Failed to parse tag {tag_cell} as literal, skipping row"
                )
                continue

            tags = tag_binding.value.split(",")
        else:
            tags = []

        yield id, value, tags


def split_iri(iri: str) -> tuple[str, str]:
    if "://" not in iri:
        return "", iri

    # split iri into prefix and last part after final / or #
    last_hashtag = iri.rfind("#")
    last_slash = iri.rfind("/")
    last = max(last_hashtag, last_slash)
    if last == -1:
        return "", iri
    else:
        return iri[:last], iri[last + 1 :]


def camel_case_split(s: str) -> str:
    # split camelCase into words
    # find uppercase letters
    words = []
    last = 0
    for i, c in enumerate(s):
        if c.isupper() and i > 0 and s[i - 1].islower():
            words.append(s[last:i])
            last = i

    if last < len(s):
        words.append(s[last:])

    return " ".join(words)


def get_object_name_from_id(obj_id: str, prefixes: dict[str, str]) -> str:
    pfx = find_longest_prefix(obj_id, prefixes)
    if pfx is None:
        # no known prefix, split after final / or # to get objet name
        _, obj_name = split_iri(obj_id)
    else:
        _, long = pfx
        obj_name = obj_id[len(long) :]

    # url decode the object name
    return unquote_plus(obj_name)


def get_value_from_id(obj_id: str, prefixes: dict[str, str]) -> str:
    obj_name = get_object_name_from_id(obj_id, prefixes)
    label = " ".join(camel_case_split(part) for part in split_at_punctuation(obj_name))
    return label.strip()


# we consider _, -, and . as url punctuation
PUNCTUATION = {"_", "-", "."}


def split_at_punctuation(s: str) -> Iterator[str]:
    start = 0
    for i, c in enumerate(s):
        if c not in PUNCTUATION:
            continue

        yield s[start:i]
        start = i + 1

    if start < len(s):
        yield s[start:]


def prepare_items(
    bindings: Iterator[Binding],
    prefixes: dict[str, str],
    add_id_as_label: None | str = None,
    logger: Logger | None = None,
) -> Iterator[dict]:
    # collect all labels for an id (which are consecutive in the stream)
    last_id = None
    fields = []
    for num, (id, value, tags) in enumerate(bindings, start=1):
        if logger and num % 1_000_000 == 0:
            logger.info(f"Processed {num:,} bindings so far")

        if logger:
            logger.debug(
                f"Processing binding #{num:,}: id={id}, value={value}, tags={tags}"
            )

        if last_id is not None and id != last_id:
            # yield previous item
            if add_id_as_label == "always" or (
                add_id_as_label == "empty" and not fields
            ):
                fields.append(
                    {
                        "type": "text",
                        "value": get_value_from_id(last_id, prefixes),
                        "tags": [],
                    }
                )

            yield {
                "identifier": last_id,
                "fields": ordered_unique(fields, key=lambda f: f["value"]),
            }

            fields = []

        last_id = id
        if value:
            fields.append({"type": "text", "value": value, "tags": tags})

    if last_id is None:
        return

    # dont forget final item
    if add_id_as_label == "always" or (add_id_as_label == "empty" and not fields):
        fields.append(
            {
                "type": "text",
                "value": get_value_from_id(last_id, prefixes),
                "tags": [],
            }
        )

    yield {
        "identifier": last_id,
        "fields": ordered_unique(fields, key=lambda f: f["value"]),
    }


def merge_data(
    kgs: list[str],
    sub_dir: str,
    out_dir: str,
    logger: Logger,
    overwrite: bool = False,
):
    out_dir = os.path.join(out_dir, sub_dir)
    data_file = os.path.join(out_dir, "data.jsonl")
    kg_info = ", ".join(kgs)
    if os.path.exists(data_file) and not overwrite:
        logger.info(
            f"Merged data for {sub_dir} of knowledge graphs {kg_info} "
            f"already exists at {data_file}, skipping merge"
        )
        return

    logger.info(
        f"Merging data for {sub_dir} of knowledge graphs {kg_info} into {data_file}"
    )

    os.makedirs(out_dir, exist_ok=True)

    others = []

    for kg in tqdm(kgs[1:], desc="Building mappings for data to merge"):
        kg_data_file = os.path.join(get_index_dir(kg), sub_dir, "data.jsonl")

        items = load_jsonl(kg_data_file)
        others.append({item["identifier"]: item for item in items})

    # first kg is the main one, to which we add data from the others
    kg = kgs[0]
    kg_data_file = os.path.join(get_index_dir(kg), sub_dir, "data.jsonl")

    def merge() -> Iterator[str]:
        with open(kg_data_file, "r") as f:
            for line in tqdm(f, desc="Merging data"):
                item = json.loads(line)

                identifier = item["identifier"]

                # collect fields from other kgs and add them
                # to the current item
                seen = set(field["value"] for field in item["fields"])
                for mapping in others:
                    if identifier not in mapping:
                        continue

                    other_item = mapping[identifier]
                    for field in other_item["fields"]:
                        if field["value"] in seen:
                            continue

                        item["fields"].append(field)
                        seen.add(field["value"])

                yield item

    dump_jsonl(merge(), data_file)


def merge_kgs(
    kgs: list[str],
    out_kg: str,
    overwrite: bool = False,
    log_level: str | int | None = None,
):
    assert len(kgs) >= 2, "At least two knowledge graphs are required to merge"

    logger = get_logger("GRASP MERGE", log_level)

    out_dir = get_index_dir(out_kg)

    merge_data(kgs, "entities", out_dir, logger, overwrite)

    ent_dir = os.path.join(out_dir, "entities")
    build_data_and_mapping(ent_dir, logger, overwrite)

    merge_data(kgs, "properties", out_dir, logger, overwrite)

    prop_dir = os.path.join(out_dir, "properties")
    build_data_and_mapping(prop_dir, logger, overwrite)
