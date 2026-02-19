import math
import os
import sys
import tempfile
import time
from typing import Any, Iterable

from search_rdf import Data, FuzzyIndex
from search_rdf.model import TextEmbeddingModel
from universal_ml_utils.logging import get_logger
from universal_ml_utils.table import generate_table

from grasp.configs import KgConfig
from grasp.manager.cache import Cache
from grasp.manager.normalizer import Normalizer
from grasp.manager.utils import (
    SearchIndex,
    describe_index,
    find_obj_type_from_prefixes,
    get_common_sparql_prefixes,
    load_kg_caches,
    load_kg_indices,
    load_kg_info_sparqls,
    load_kg_normalizers,
    load_kg_prefixes,
)
from grasp.sparql.types import (
    Alternative,
    AskResult,
    Binding,
    ObjType,
    Position,
    Selection,
    SelectResult,
    SelectRow,
    group_selections,
)
from grasp.sparql.utils import (
    READ_TIMEOUT,
    REQUEST_TIMEOUT,
    SPARQLException,
    ask_to_select,
    autocomplete_prefix,
    autocomplete_sparql,
    execute,
    find_longest_prefix,
    fix_prefixes,
    format_iri,
    get_endpoint,
    has_iri,
    load_entity_info_sparql,
    load_iri_and_literal_parser,
    load_property_info_sparql,
    load_sparql_parser,
    parse_string,
    prettify,
    query_type,
)
from grasp.utils import clip, format_list, ordered_unique


class KgManager:
    def __init__(
        self,
        kg: str,
        entity_normalizer: Normalizer,
        property_normalizer: Normalizer,
        entity_index: SearchIndex | None = None,
        property_index: SearchIndex | None = None,
        entity_info_sparql: str | None = None,
        property_info_sparql: str | None = None,
        entity_cache: Cache | None = None,
        property_cache: Cache | None = None,
        prefixes: dict[str, str] | None = None,
        endpoint: str | None = None,
    ):
        self.kg = kg

        self.entity_index = entity_index
        self.entity_data = entity_index.data() if entity_index else None
        self.entity_cache = entity_cache

        self.property_index = property_index
        self.property_data = property_index.data() if property_index else None
        self.property_cache = property_cache

        self.entity_normalizer = entity_normalizer
        self.property_normalizer = property_normalizer

        self.sparql_parser = load_sparql_parser()
        self.iri_literal_parser = load_iri_and_literal_parser()

        self.prefixes = prefixes or {}

        self.entity_info_sparql = entity_info_sparql or load_entity_info_sparql()
        self.property_info_sparql = property_info_sparql or load_property_info_sparql()
        self.disable_info_retrieval = False

        self.endpoint = endpoint or get_endpoint(self.kg)

        self.embedding_model: TextEmbeddingModel | None = None

        self.logger = get_logger(f"{self.kg.upper()} KG MANAGER")

    def set_embedding_model(self, model: TextEmbeddingModel) -> None:
        self.embedding_model = model

    def set_info_retrieval(self, enable: bool) -> None:
        self.disable_info_retrieval = not enable

    def prettify(
        self,
        sparql: str,
        indent: int = 2,
        is_prefix: bool = False,
    ) -> str:
        return prettify(sparql, self.sparql_parser, indent, is_prefix)

    def check_sparql(self, sparql: str, is_prefix: bool = False) -> bool:
        try:
            parse_string(
                sparql,
                self.sparql_parser,
                skip_empty=True,
                collapse_single=True,
                is_prefix=is_prefix,
            )
            return True
        except Exception as e:
            self.logger.debug(f"Invalid SPARQL query {sparql}: {e}")
            return False

    def execute_sparql(
        self,
        sparql: str,
        request_timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
        max_retries: int = 0,
        force_select_result: bool = False,
        read_timeout: float | None = READ_TIMEOUT,
    ) -> SelectResult | AskResult:
        if force_select_result:
            # ask_to_select returns None if sparql is not an ask query
            sparql = ask_to_select(sparql, self.sparql_parser) or sparql

        sparql = self.fix_prefixes(sparql)

        self.logger.debug(f"Executing SPARQL query against {self.endpoint}:\n{sparql}")
        return execute(
            sparql,
            self.endpoint,
            request_timeout,
            max_retries,
            read_timeout,
        )

    def format_sparql_result(
        self,
        result: SelectResult | AskResult,
        show_top_rows: int = 5,
        show_bottom_rows: int = 5,
        show_left_columns: int = 5,
        show_right_columns: int = 5,
        column_names: list[str] | None = None,
    ) -> str:
        # run sparql against endpoint, format result as string
        if isinstance(result, AskResult):
            return str(result.boolean)

        if result.num_rows == 0:
            return f"Got no rows and {result.num_columns:,} columns"

        assert column_names is None or len(column_names) == result.num_columns, (
            f"Expected {result.num_columns:,} column names"
        )
        assert show_top_rows or show_bottom_rows, "At least one row must be shown"
        assert show_left_columns or show_right_columns, (
            "At least one column must be shown"
        )

        left_end = min(show_left_columns, result.num_columns)
        right_start = result.num_columns - show_right_columns
        if right_start > left_end:
            column_indices = list(range(left_end))
            column_indices.append(-1)
            column_indices.extend(range(right_start, result.num_columns))
        else:
            column_indices = list(range(result.num_columns))

        def format_row(row: SelectRow) -> list[str]:
            formatted_row = []
            for c in column_indices:
                if c < 0:
                    formatted_row.append("...")
                    continue

                var = result.variables[c]
                val = row.get(var, None)
                if val is None:
                    formatted_row.append("")

                elif val.typ == "bnode":
                    formatted_row.append(val.identifier())

                elif val.typ == "literal":
                    formatted = clip(val.value)
                    if val.lang is not None:
                        formatted += f" (lang:{val.lang})"
                    elif val.datatype is not None:
                        datatype = self.format_iri(f"<{val.datatype}>")
                        formatted += f" ({datatype})"

                    formatted_row.append(formatted)

                else:
                    assert val.typ == "uri"
                    identifier = val.identifier()
                    formatted = self.format_iri(identifier)

                    # for uri check whether it is in one of the datasets
                    obj_type = ObjType.ENTITY
                    norm = self.normalize(identifier, obj_type)

                    if norm is None or self.label(norm[0], obj_type) is None:
                        obj_type = ObjType.PROPERTY
                        norm = self.normalize(identifier, obj_type)

                    # still not found, just output the formatted iri
                    if norm is None or self.label(norm[0], obj_type) is None:
                        formatted_row.append(formatted)
                        continue

                    label = self.label(norm[0], obj_type)
                    assert label is not None, "should not happen"
                    formatted = f"{clip(label)} ({formatted})"
                    formatted_row.append(formatted)

            return formatted_row

        # generate a nicely formatted table
        column_names = column_names or result.variables
        header = [column_names[c] if c >= 0 else "..." for c in column_indices]
        top_end = min(show_top_rows, result.num_rows)
        bottom_start = max(result.num_rows - show_bottom_rows, top_end)

        data = [format_row(row) for row in result.rows(end=top_end)]

        if bottom_start > top_end:
            data.append(["..."] * len(header))

        data.extend(
            format_row(row) for row in result.rows(bottom_start, result.num_rows)
        )

        table = generate_table(
            data,
            [header],
            alignments=["left"] * len(header),
            max_column_width=sys.maxsize,
        )

        comp = "" if result.complete else "more than "
        formatted = (
            f"Got {comp}{result.num_rows:,} row{'s' * (result.num_rows != 1)} and "
            f"{result.num_columns:,} column{'s' * (result.num_columns != 1)}"
        )

        showing = []

        if right_start > left_end:
            # columns restricted
            show_columns = []
            if show_left_columns:
                show_columns.append(f"first {show_left_columns}")
            if show_right_columns:
                show_columns.append(f"last {show_right_columns}")

            showing.append(f"the {' and '.join(show_columns)} columns")

        if bottom_start > top_end:
            # rows restricted
            show_rows = []
            if show_top_rows:
                show_rows.append(f"first {show_top_rows}")
            if show_bottom_rows:
                show_rows.append(f"last {show_bottom_rows}")

            showing.append(f"the {' and '.join(show_rows)} rows")

        if showing:
            formatted += ", showing " + " and ".join(showing) + " below"

        formatted += f":\n{table}"
        return formatted

    def get_formatted_sparql_result(
        self,
        sparql: str,
        request_timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
        max_retries: int = 0,
        max_rows: int = 10,
        max_columns: int = 10,
    ) -> str:
        half_rows = math.ceil(max_rows / 2)
        half_columns = math.ceil(max_columns / 2)
        try:
            result = self.execute_sparql(sparql, request_timeout, max_retries)
            return self.format_sparql_result(
                result,
                half_rows,
                half_rows,
                half_columns,
                half_columns,
            )
        except Exception as e:
            return f"SPARQL execution failed:\n{e}"

    def find_longest_prefix(self, iri: str) -> tuple[str, str] | None:
        return find_longest_prefix(iri, self.prefixes)

    def format_iri(self, iri: str, base_uri: str | None = None) -> str:
        return format_iri(iri, self.prefixes, base_uri)

    def fix_prefixes(
        self,
        sparql: str,
        is_prefix: bool = False,
        remove_known: bool = False,
        sort: bool = False,
    ) -> str:
        return fix_prefixes(
            sparql,
            self.sparql_parser,
            self.prefixes,
            is_prefix,
            remove_known,
            sort,
        )

    def normalizer(
        self,
        obj_type: ObjType,
    ) -> Normalizer | None:
        if obj_type == ObjType.ENTITY:
            return self.entity_normalizer
        elif obj_type == ObjType.PROPERTY:
            return self.property_normalizer
        else:
            return None

    def index(
        self,
        obj_type: ObjType,
    ) -> SearchIndex | None:
        if obj_type == ObjType.ENTITY:
            return self.entity_index
        elif obj_type == ObjType.PROPERTY:
            return self.property_index
        else:
            return None

    def data(
        self,
        obj_type: ObjType,
    ) -> Data | None:
        if obj_type == ObjType.ENTITY:
            return self.entity_data
        elif obj_type == ObjType.PROPERTY:
            return self.property_data
        else:
            return None

    def normalize(
        self,
        identifier: str,
        obj_type: ObjType,
    ) -> tuple[str, str | None] | None:
        if obj_type == ObjType.ENTITY:
            return self.entity_normalizer.normalize(identifier)
        elif obj_type == ObjType.PROPERTY:
            return self.property_normalizer.normalize(identifier)
        else:
            return None

    def denormalize(
        self,
        identifier: str,
        obj_type: ObjType,
        variant: str | None = None,
    ) -> str | None:
        if obj_type == ObjType.ENTITY:
            return self.entity_normalizer.denormalize(identifier, variant)
        elif obj_type == ObjType.PROPERTY:
            return self.property_normalizer.denormalize(identifier, variant)
        else:
            return identifier

    def check_identifier(
        self,
        identifier: str,
        obj_type: ObjType,
    ) -> bool:
        if obj_type == ObjType.ENTITY:
            data = self.entity_data
        elif obj_type == ObjType.PROPERTY:
            data = self.property_data
        else:
            data = None

        if data is None:
            return False

        return data.id_from_identifier(identifier) is not None

    def label(
        self,
        identifier: str,
        obj_type: ObjType,
    ) -> str | None:
        if obj_type == ObjType.ENTITY:
            data = self.entity_data
        elif obj_type == ObjType.PROPERTY:
            data = self.property_data
        else:
            data = None

        if data is None:
            return None

        id = data.id_from_identifier(identifier)
        if id is None:
            return None

        return data.main_field(id) or data.field(id, 0)

    def build_alternative_with_infos(
        self,
        identifier: str,
        infos: dict | None = None,
        variants: list[str] | None = None,
        matched_via: str | None = None,
    ) -> Alternative:
        if infos is None:
            infos = {}

        # extract needed data from infos dict
        label = infos.get("label")
        aliases = infos.get("alias", [])
        added_infos = infos.get("info", [])

        return self.build_alternative(
            identifier,
            label,
            aliases,
            added_infos,
            variants,
            matched_via,
        )

    def build_alternative(
        self,
        identifier: str,
        label: str | None = None,
        aliases: list[str] | None = None,
        infos: list[str] | None = None,
        variants: list[str] | None = None,
        matched_via: str | None = None,
    ) -> Alternative:
        # preprocess some fields
        if variants is not None:
            variants = ordered_unique(variants)

        if aliases is not None:
            aliases = ordered_unique(aliases, filter=lambda alias: alias != label)

        if infos is not None:
            infos = ordered_unique(infos)

        return Alternative(
            identifier=identifier,
            short_identifier=self.format_iri(identifier),
            label=label,
            variants=variants,
            aliases=aliases,
            infos=infos,
            matched_label=matched_via,
        )

    def parse_bindings(self, result: Iterable[Binding | None]) -> dict[ObjType, Any]:
        entities = {}
        properties = {}
        others = []
        literals = []
        for binding in result:
            if binding is None:
                continue

            elif binding.typ == "bnode":
                # ignore bnodes
                continue

            identifier = binding.identifier()
            infos = []

            if binding.typ == "literal":
                if binding.datatype is not None:
                    datatype = self.format_iri(f"<{binding.datatype}>")
                    infos.append(datatype)
                elif binding.lang is not None:
                    infos.append(binding.lang)

                literals.append((identifier, binding.value, infos))
                continue

            # typ is uri
            unmatched = True
            for identifier_map, obj_type in [
                (entities, ObjType.ENTITY),
                (properties, ObjType.PROPERTY),
            ]:
                norm = self.normalize(identifier, obj_type)
                if norm is None or not self.check_identifier(norm[0], obj_type):
                    continue

                identifier, variant = norm
                if identifier not in identifier_map:
                    identifier_map[identifier] = []

                if variant is not None:
                    identifier_map[identifier].append(variant)

                unmatched = False

            if unmatched:
                others.append((identifier, self.format_iri(identifier), infos))

        common_prefixes = get_common_sparql_prefixes()

        unindexed = []
        common = []
        for item in others:
            obj_type = find_obj_type_from_prefixes(
                item[0],
                self.prefixes,
                common_prefixes,
            )
            if obj_type == ObjType.UNINDEXED:
                unindexed.append(item)
            elif obj_type == ObjType.COMMON:
                common.append(item)

        return {
            ObjType.ENTITY: entities,
            ObjType.PROPERTY: properties,
            ObjType.UNINDEXED: others,
            ObjType.COMMON: common,
            ObjType.LITERAL: literals,
        }

    def search_entity(
        self,
        query: str | None = None,
        k: int = 10,
        identifier_map: dict[str, list[str]] | None = None,
        **search_kwargs: Any,
    ) -> list[Alternative]:
        return self.search(
            ObjType.ENTITY,
            query,
            k,
            identifier_map,
            **search_kwargs,
        )

    def search_property(
        self,
        query: str | None = None,
        k: int = 10,
        identifier_map: dict[str, list[str]] | None = None,
        **search_kwargs: Any,
    ) -> list[Alternative]:
        return self.search(
            ObjType.PROPERTY,
            query,
            k,
            identifier_map,
            **search_kwargs,
        )

    def retrieve_infos_for_identifiers(
        self,
        identifiers: Iterable[str],
        info_sparql: str,
    ) -> dict[str, dict]:
        infos = {}

        try:
            assert "{IDS}" in info_sparql, (
                "SPARQL must contain {IDS} placeholder for identifiers"
            )
            info_sparql = info_sparql.replace("{IDS}", " ".join(identifiers))
            self.logger.debug(f"Retrieving infos with SPARQL:\n{info_sparql}")
            result = self.execute_sparql(
                info_sparql,
                # set info timeouts to something shorter than usual
                request_timeout=(4.0, 6.0),
                read_timeout=6.0,
            )
            assert isinstance(result, SelectResult) and result.num_columns == 3, (
                "Expected a SELECT query with three columns for info SPARQL"
            )
            id_var = result.variables[0]
            text_var = result.variables[1]
            type_var = result.variables[2]
            for row in result.rows():
                assert id_var in row, "Identifier column not found in result row"
                assert row[id_var].typ == "uri"
                assert row[text_var].typ == "literal"
                assert row[type_var].typ == "literal"

                identifier = row[id_var].identifier()
                if identifier not in infos:
                    infos[identifier] = {}

                typ = row[type_var].value
                assert typ in {"label", "alias", "info"}
                if typ == "label":
                    # only keep one label
                    infos[identifier]["label"] = row[text_var].value
                    continue

                # keep list for other types
                if typ not in infos[identifier]:
                    infos[identifier][typ] = []

                text = row[text_var].value
                infos[identifier][typ].append(text)

        except Exception as e:
            self.logger.warning(f"Failed to retrieve infos for identifiers: {e}")

        return infos

    def get_infos_for_identifiers_of_type(
        self,
        identifiers: Iterable[str],
        obj_type: ObjType,
    ) -> dict[str, dict]:
        if obj_type == ObjType.ENTITY:
            info_sparql = self.entity_info_sparql
            info_cache = self.entity_cache
            data = self.entity_data
        elif obj_type == ObjType.PROPERTY:
            info_sparql = self.property_info_sparql
            info_cache = self.property_cache
            data = self.property_data
        else:
            self.logger.warning(f"No info retrieval for object type '{obj_type}'")
            return {}

        return self.get_infos_for_identifiers(
            identifiers,
            info_sparql,
            info_cache,
            data,
        )

    def get_infos_for_identifiers(
        self,
        identifiers: Iterable[str],
        info_sparql: None | str = None,
        info_cache: None | Cache = None,
        data: None | Data = None,
    ) -> dict[str, dict]:
        infos = {}

        # try cache first
        if info_cache is not None:
            left = []
            for identifier in identifiers:
                info = info_cache.get(identifier)
                if info is None:
                    left.append(identifier)
                    continue

                infos[identifier] = info

            identifiers = left

        if not identifiers:
            return infos

        # try live SPARQL next
        if info_sparql is not None and not self.disable_info_retrieval:
            live_infos = self.retrieve_infos_for_identifiers(
                identifiers,
                info_sparql,
            )
            infos.update(live_infos)

        # try and fill up remaining from local data
        if data is not None:
            for identifier in identifiers:
                if identifier in infos:
                    continue

                id = data.id_from_identifier(identifier)
                if id is None:
                    continue

                info = {}
                label = data.main_field(id)
                if label is not None:
                    info["label"] = label

                aliases = data.fields(id)
                if aliases:
                    info["alias"] = aliases

                if info:
                    infos[identifier] = info

        return infos

    def search(
        self,
        obj_type: ObjType,
        query: str | None = None,
        k: int = 10,
        identifier_map: dict[str, list[str]] | None = None,
        **search_kwargs: Any,
    ) -> list[Alternative]:
        index = self.index(obj_type)
        data = self.data(obj_type)
        normalizer = self.normalizer(obj_type)
        assert index is not None and data is not None and normalizer is not None, (
            f"No index, data, or normalizer for object type {obj_type}"
        )

        field_map = {}

        if query is None:
            if identifier_map is None:
                # first k
                identifiers = [data.identifier(id) or "" for id in range(k)]
            else:
                # first k with lowest ids
                identifiers = sorted(
                    identifier_map,
                    key=lambda ident: data.id_from_identifier(ident) or len(data),
                )[:k]
        else:
            kwargs = {}
            if index.index_type == "embedding":
                # embedding index can also have min score passed
                kwargs["min_score"] = search_kwargs.get("min_score")
                assert self.embedding_model is not None, (
                    "Embedding model must be set for embedding index search"
                )
                embedding: list[float] = self.embedding_model.embed([query])[0].tolist()  # type: ignore
                kwargs["embedding"] = embedding
            else:
                kwargs["query"] = query

            if identifier_map is None:
                allow_ids = None
            else:
                allow_ids = set()
                for identifier in identifier_map:
                    id = data.id_from_identifier(identifier)
                    if id is not None:
                        allow_ids.add(id)

            identifiers = []
            for id, field, _ in index.search(k=k, allow_ids=allow_ids, **kwargs):
                identifier = data.identifier(id)
                assert identifier is not None, "should not happen"
                identifiers.append(identifier)
                field_map[identifier] = data.field(id, field)

        infos = self.get_infos_for_identifiers_of_type(identifiers, obj_type)

        alternatives = []
        for identifier in identifiers:
            if identifier_map is not None:
                variants = identifier_map[identifier]
            else:
                variants = normalizer.default_variants()

            matched_via = None
            if field_map:
                matched_via = field_map.get(identifier)

            alternative = self.build_alternative_with_infos(
                identifier,
                infos.get(identifier, {}),
                variants,
                matched_via,
            )
            alternatives.append(alternative)

        return alternatives

    def get_temporary_index_alternatives(
        self,
        obj_type: ObjType,
        items: list[tuple[str, str, list[str]]],
        query: str | None = None,
        k: int = 10,
    ) -> list[Alternative]:
        if query is None:
            return [
                Alternative(
                    identifier=identifier,
                    short_identifier=self.format_iri(identifier),
                    label=label,
                    infos=infos,
                )
                for identifier, label, infos in items[:k]
            ]

        def make_text_item(identifier: str, value: str) -> dict:
            field: dict = {"type": "text", "value": value, "tags": ["main"]}
            return {"identifier": identifier, "fields": [field]}

        with tempfile.TemporaryDirectory() as temp_dir:
            # build temporary index and search in it
            data_dir = os.path.join(temp_dir, "data")
            index_dir = os.path.join(temp_dir, "index")
            os.makedirs(index_dir, exist_ok=True)
            self.logger.debug(
                f"Building temporary index in {temp_dir} "
                f"with data at {data_dir} and index in {index_dir}"
            )

            data_items = []
            items_map = {}
            for item in items:
                identifier, label, _ = item
                data_items.append(make_text_item(identifier, label))
                items_map[identifier] = item

            # build index data
            Data.build_from_items(data_items, data_dir)
            data = Data.load(data_dir)

            # use a fuzzy index here because it is faster to build
            # and query
            FuzzyIndex.build(data, index_dir)
            index = FuzzyIndex.load(data, index_dir)

            alternatives = []
            matches = index.search(query, k=k)
            for id, *_ in matches:
                identifier = short_identifier = data.identifier(id)
                assert identifier is not None, "should not happen"

                if obj_type == ObjType.LITERAL:
                    # for literals, clip the identifier to make it more readable
                    short_identifier = clip(identifier)
                else:
                    short_identifier = self.format_iri(identifier)

                identifier, label, infos = items_map[identifier]
                alternatives.append(
                    Alternative(
                        identifier=identifier,
                        short_identifier=short_identifier,
                        label=label,
                        infos=infos,
                    )
                )

            return alternatives

    def autocomplete_prefix(
        self,
        prefix: str,
        limit: int | None = None,
    ) -> tuple[str, str, Position]:
        return autocomplete_prefix(prefix, self.sparql_parser, limit)

    def autocomplete_sparql(
        self,
        sparql: str,
        limit: int | None = None,
    ) -> tuple[str, Position]:
        return autocomplete_sparql(sparql, self.sparql_parser, "search", limit)

    def get_default_search_items(
        self,
        position: Position,
    ) -> dict[ObjType, Any]:
        output = {}
        # entities can be subjects and objects
        if position == Position.SUBJECT or position == Position.OBJECT:
            # None (full index) by default
            output[ObjType.ENTITY] = None

        # properties can only be properties
        if position == Position.PROPERTY:
            # None (full index) by default
            output[ObjType.PROPERTY] = None

        # literals can only be objects
        if position == Position.OBJECT:
            # empty by default
            output[ObjType.LITERAL] = []

        # other iris can always be subjects, properties, and objects
        # empty by default
        output[ObjType.UNINDEXED] = []
        return output

    def get_search_items(
        self,
        sparql: str,
        position: Position,
        max_candidates: int | None = None,
        timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
        max_retries: int = 0,
    ) -> dict[ObjType, Any]:
        # start with defaults
        search_items = self.get_default_search_items(position)

        typ = query_type(sparql, self.sparql_parser)
        if typ != "select":
            # fall back to full search on non-select queries
            raise SPARQLException("SPARQL query is not a SELECT query")

        elif not has_iri(sparql, self.sparql_parser):
            # contains no iris, with no restriction we do not need
            # to query the endpoint for autocompletion
            raise SPARQLException("SPARQL query contains no IRIs to constrain with")

        self.logger.debug(f"Getting search items with {sparql}")
        try:
            result = self.execute_sparql(sparql, timeout, max_retries)
        except Exception as e:
            self.logger.debug(
                f"Getting autocompletion result for position {position} "
                f"with sparql {sparql} failed with error: {e}"
            )
            raise SPARQLException(f"SPARQL execution failed: {e}")

        # some checks that should not happen, just to be sure
        if not isinstance(result, SelectResult):
            raise SPARQLException("SPARQL query is not a select query")
        elif result.num_columns != 1:
            raise SPARQLException("SPARQL query does not return a single column")
        elif max_candidates is not None and len(result) > max_candidates:
            raise SPARQLException(
                f"Got more than the maximum supported number of {position.value} "
                f"candidates ({max_candidates:,})"
            )

        self.logger.debug(
            f"Got {len(result):,} fitting items for position {position} "
            f"with sparql '{sparql}'"
        )

        # split result into entities, properties, other iris
        # and literals
        start = time.perf_counter()
        parsed_search_items = self.parse_bindings(
            next(iter(bindings), None) for bindings in result.bindings()
        )
        end = time.perf_counter()
        self.logger.debug(
            f"Parsing {len(result):,} search items took {1000 * (end - start):.2f}ms"
        )

        # overwrite defaults where needed
        for obj_type in search_items:
            if obj_type not in parsed_search_items:
                continue

            search_items[obj_type] = parsed_search_items[obj_type]

        return search_items

    def get_selection_alternatives(
        self,
        search_query: str | None,
        search_items: dict[ObjType, Any],
        k: int,
        **search_kwargs: Any,
    ) -> dict[ObjType, list[Alternative]]:
        self.logger.debug(
            f'Getting top {k} selection alternatives with query "{search_query}" for '
            f"object types {', '.join(obj_type.value for obj_type in search_items)}"
        )
        alternatives = {}

        start = time.perf_counter()

        if ObjType.ENTITY in search_items:
            alternatives[ObjType.ENTITY] = self.search_entity(
                search_query,
                k,
                search_items[ObjType.ENTITY],
                **search_kwargs,
            )

        if ObjType.PROPERTY in search_items:
            alternatives[ObjType.PROPERTY] = self.search_property(
                search_query,
                k,
                search_items[ObjType.PROPERTY],
                **search_kwargs,
            )

        end = time.perf_counter()
        self.logger.debug(
            f"Getting entity and property alternatives "
            f"took {1000 * (end - start):.2f}ms"
        )

        start = time.perf_counter()

        for obj_type in [ObjType.UNINDEXED, ObjType.LITERAL]:
            if obj_type not in search_items:
                continue

            alternatives[obj_type] = self.get_temporary_index_alternatives(
                obj_type,
                search_items[obj_type],
                search_query,
                k,
            )

        end = time.perf_counter()
        self.logger.debug(
            f"Getting other and literal alternatives took {1000 * (end - start):.2f}ms"
        )

        return alternatives

    def format_selections(self, selections: list[Selection]) -> str:
        rename_obj_type = [
            (ObjType.ENTITY, "entities"),
            (ObjType.PROPERTY, "properties"),
            (ObjType.UNINDEXED, "other (non-indexed) items"),
        ]

        grouped = group_selections(selections)

        return "\n\n".join(
            f"Using {name}:\n"
            + format_list(
                alt.get_selection_string(include_variants=variants)
                for alt, variants in grouped[obj_type]
            )
            for obj_type, name in rename_obj_type
            if obj_type in grouped
        )


def load_kg_manager(
    cfg: KgConfig,
    skip_indices: bool = False,
    skip_caches: bool = False,
) -> KgManager:
    ent_index = prop_index = None
    if not skip_indices:
        ent_index, prop_index = load_kg_indices(
            cfg.kg,
            cfg.entities_type,
            cfg.properties_type,
        )

    prefixes = load_kg_prefixes(cfg.kg, cfg.endpoint)
    ent_norm, prop_norm = load_kg_normalizers(cfg.kg)
    ent_info_sparql, prop_info_sparql = load_kg_info_sparqls(cfg.kg)

    ent_cache = prop_cache = None
    if not skip_caches:
        ent_cache, prop_cache = load_kg_caches(cfg.kg)

    return KgManager(
        cfg.kg,
        ent_norm,
        prop_norm,
        ent_index,
        prop_index,
        ent_info_sparql,
        prop_info_sparql,
        ent_cache,
        prop_cache,
        prefixes,
        cfg.endpoint,
    )


def format_kgs(managers: list[KgManager], kg_notes: dict[str, list[str]]) -> str:
    if not managers:
        return "No knowledge graphs available"

    return format_list(
        format_kg(
            manager,
            kg_notes.get(manager.kg, []),
        )
        for manager in managers
    )


def format_kg(manager: KgManager, notes: list[str]) -> str:
    msg = f"{manager.kg} at {manager.endpoint}"

    parts = []
    if manager.entity_index is not None:
        ent_type, _ = describe_index(manager.entity_index)
        parts.append(f"{ent_type.lower()} for entities")
    if manager.property_index is not None:
        prop_type, _ = describe_index(manager.property_index)
        parts.append(f"{prop_type.lower()} for properties")

    if parts:
        msg += " with " + " and ".join(parts)

    if not notes:
        return msg

    msg += ", and notes:\n" + format_list(notes, indent=2)
    return msg
