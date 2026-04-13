import os

from pydantic import BaseModel

from grasp.manager import KgManager
from grasp.manager.utils import (
    get_common_sparql_prefixes,
    load_index_description,
    load_index_sparql,
    load_info_sparql,
    load_kg_info,
    merge_prefixes,
)
from grasp.model import Message
from grasp.sparql.utils import (
    load_entity_index_sparql,
    load_entity_info_sparql,
    load_property_index_sparql,
    load_property_info_sparql,
)
from grasp.tasks.auto_setup.functions import (
    INDEX_SPARQL_VARS,
    INFO_SPARQL_VARS,
    index_functions,
    info_functions,
    validate_sparql,
)
from grasp.tasks.base import GraspTask
from grasp.tasks.functions import find_frequent, find_frequent_function_definition
from grasp.utils import FunctionCallException, format_prefixes, get_index_dir

# maps index name to (index_sparql, info_sparql) reference defaults
REFERENCE_SPARQLS = {
    "entities": (load_entity_index_sparql(), load_entity_info_sparql()),
    "properties": (load_property_index_sparql(), load_property_info_sparql()),
}


class IndexState(BaseModel):
    index_sparql: str | None = None
    info_sparql: str | None = None
    description: str | None = None


class InfoState(BaseModel):
    prefixes: dict[str, str] = {}
    description: str | None = None


def load_index_state(manager: KgManager, name: str) -> IndexState:
    index_dir = get_index_dir(manager.kg)
    sub_index_dir = os.path.join(index_dir, name)
    index_sparql = load_index_sparql(sub_index_dir, manager.logger)
    info_sparql = load_info_sparql(sub_index_dir, manager.logger)
    description = load_index_description(sub_index_dir, manager.logger)
    return IndexState(
        index_sparql=index_sparql,
        info_sparql=info_sparql,
        description=description,
    )


def load_info_state(manager: KgManager) -> InfoState:
    prefixes, desc = load_kg_info(manager.kg)
    return InfoState(prefixes=prefixes, description=desc)


def format_query(manager: KgManager, query: str | None) -> str:
    if query is None:
        return "None"
    return manager.fix_prefixes(query, remove_known=True)


def format_index_state(state: IndexState, manager: KgManager) -> str:
    return f"""\
Description:
{state.description or "None"}

Index SPARQL:
{format_query(manager, state.index_sparql)}

Info SPARQL:
{format_query(manager, state.info_sparql)}"""


def format_info_state(state: InfoState) -> str:
    return f"""\
Description:
{state.description or "None"}

Prefixes:
{format_prefixes(state.prefixes)}"""


class AutoSetupTask(GraspTask):
    name = "auto-setup"

    def system_information(self) -> str:
        manager = self.managers[0]

        if self.input["phase"] == "index":
            return self._index_system_information(manager)
        else:
            return self._info_system_information()

    def _index_system_information(self, manager: KgManager) -> str:
        name = self.input["name"]
        index_sparql, info_sparql = REFERENCE_SPARQLS[name]

        return f"""\
You are a knowledge graph setup assistant. Your task is to explore \
a SPARQL endpoint and come up with index and info SPARQL queries \
for {name}. If there already exist such queries for {name}, you \
should focus on improving them.

The index SPARQL query is used to build a search index for {name} \
in the knowledge graph. It must be a SELECT query returning three variables:
- ?id: the IRI
- ?value: a string literal (typically label or alias) or the IRI itself \
(if should be searchable)
- ?tags: "main" for the primary label, unbound for secondary values (aliases)
Results should be ordered descending by a relevance score and then by ?id, \
to break ties in search results. A sensible relevant score that can always be \
computed is the total number of occurrences of an IRI in all triples, but some \
knowledge graphs might provide other measures of relevance that you can use \
instead (e.g., Wikidata's "sitelinks").

The info SPARQL query retrieves additional context for {name} found \
during search, to support disambiguation. It must be a SELECT query returning:
- ?id: the IRI
- ?value: a string literal (label, alias, description, type, etc.)
- ?type: one of "label", "alias", or "other"
It must contain the placeholder {{IDS}} which will be replaced with a list \
of IRIs at query time. The infos retrieved per IRI should be limited to the \
most important ones (10 or fewer) to keep the query efficient and the search \
results concise. The goal is to help to characterize and distinguish different \
IRIs, and not to list all associated information.

You should follow a step-by-step approach:
1. Look at and understand the current index and info SPARQLs (if any). It \
might also help to look at the reference SPARQLs below.
2. Throgouly explore the knowledge graph using the provided functions to understand \
its structure and what {name} are available.
3. Iteratively develop and test the index and info SPARQLs. Make sure to always validate \
and execute them against the knowledge graph. If you encounter errors or unexpected results, \
go back to step 2 to gather more information.
4. Perform a final review of the SPARQL queries, and call stop.

Below are generic reference SPARQL queries you can use as a starting point. \
They are used as defaults if no specific queries have been set yet, \
but may not be optimal for this knowledge graph.

Reference index SPARQL:
{format_query(manager, index_sparql)}

Reference info SPARQL:
{format_query(manager, info_sparql)}"""

    def _info_system_information(self) -> str:
        return """\
You are a knowledge graph setup assistant. Your task is to explore \
a SPARQL endpoint and configure SPARQL prefixes and a description for the \
underlying knowledge graph. If there already exist prefixes and a description, \
you should focus on improving them.

Prefixes map short names to IRI namespaces (e.g. "wd" is short for \
"http://www.wikidata.org/entity/"). Only knowledge graph specific \
prefixes are needed as common ones like rdf, rdfs, owl, xsd are \
already available by default.

The description should be a concise summary of what the knowledge graph \
contains (typically one or two sentences about its domain and scope).

You should follow a step-by-step approach:
1. Look at and understand the current prefixes and description (if any).
2. Explore the knowledge graph using the provided functions to discover its \
structure and frequently used IRI namespaces. Add, update, and delete \
current prefixes along the way.
3. Update the description if necessary.
4. Perform a final review of the prefixes and description, and call stop."""

    def rules(self) -> list[str]:
        if self.input["phase"] == "index":
            return self._index_rules()
        else:
            return self._info_rules()

    def _index_rules(self) -> list[str]:
        return [
            "If the user provides additional notes about the setup, make sure to follow them.",
            "If you want to make an IRI searchable even if it does not have any associated literals, "
            "bind the IRI itself as value in the index SPARQL via BIND(?id AS ?value).",
            "When developing the SPARQL queries, try to make them as efficient as possible. For example, "
            "put VALUES { {IDS} } clauses in the info SPARQL inside each UNION.",
        ]

    def _info_rules(self) -> list[str]:
        return [
            "If the user provides additional notes about the setup, make sure to follow them.",
            "Avoid mentions of specific details in the knowledge graph description, but "
            "focus on the bigger picture.",
        ]

    def function_definitions(self) -> list[dict]:
        kgs = [m.kg for m in self.managers]
        functions = [find_frequent_function_definition(kgs, self.config.list_k)]
        if self.input["phase"] == "index":
            functions.extend(index_functions())
        else:
            functions.extend(info_functions())
        return functions

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        assert self.state is not None
        manager = self.managers[0]

        if fn_name == "find_frequent":
            return find_frequent(
                self.managers,
                fn_args["kg"],
                fn_args["position"],
                fn_args.get("subject"),
                fn_args.get("property"),
                fn_args.get("object"),
                fn_args.get("page", 1),
                self.config.list_k,
                known,
                self.config.sparql_request_timeout,
                self.config.sparql_read_timeout,
            )

        elif fn_name == "show_setup":
            return self.show_setup()

        elif fn_name == "set_query":
            return self.set_query(manager, **fn_args)

        elif fn_name == "add_prefix":
            return self.add_prefix(manager, fn_args["short"], fn_args["namespace"])

        elif fn_name == "delete_prefix":
            return self.delete_prefix(manager, fn_args["short"])

        elif fn_name == "update_prefix":
            return self.update_prefix(manager, fn_args["short"], fn_args["namespace"])

        elif fn_name == "set_description":
            return self.set_description(fn_args["description"])

        elif fn_name == "stop":
            return "Stopping."

        raise FunctionCallException(f"Unknown function {fn_name}")

    def set_description(self, description: str) -> str:
        self.state.description = description
        return "Description updated."

    def set_query(self, manager: KgManager, type: str, sparql: str) -> str:
        required = INFO_SPARQL_VARS if type == "info" else INDEX_SPARQL_VARS
        name = self.input["name"]

        try:
            validate_sparql(manager, sparql, required)
        except ValueError as e:
            raise FunctionCallException(
                f"Invalid {name} {type} SPARQL:\n{str(e)}"
            ) from e

        index = manager.try_get(name)
        if type == "info":
            self.state.info_sparql = sparql
            # write directly to manager if available
            if index is not None:
                index.info_sparql = sparql
        else:
            self.state.index_sparql = sparql

        msg = f"{name.capitalize()} {type} SPARQL updated"
        if type == "info" and index is not None:
            msg += " and used in subsequent function calls"
        return msg

    def apply_prefixes(self, manager: KgManager, prefixes: dict[str, str]) -> None:
        merged, _, kg_prefixes = merge_prefixes(
            get_common_sparql_prefixes(),
            prefixes,
            manager.logger,
            do_raise=True,
        )
        self.state.prefixes = kg_prefixes
        manager.kg_prefixes = kg_prefixes
        manager.prefixes = merged

    def show_setup(self) -> str:
        manager = self.managers[0]
        if self.input["phase"] == "index":
            return format_index_state(self.state, manager)
        else:
            return format_info_state(self.state)

    def add_prefix(self, manager: KgManager, short: str, namespace: str) -> str:
        if short in self.state.prefixes:
            raise FunctionCallException(f"Prefix '{short}' already exists.")

        try:
            self.apply_prefixes(manager, {**self.state.prefixes, short: namespace})
        except RuntimeError as e:
            raise FunctionCallException(str(e)) from e

        return f"Prefix '{short}' added and available for subsequent function calls."

    def delete_prefix(self, manager: KgManager, short: str) -> str:
        if short not in self.state.prefixes:
            raise FunctionCallException(f"Prefix '{short}' does not exist.")

        try:
            self.apply_prefixes(
                manager,
                {k: v for k, v in self.state.prefixes.items() if k != short},
            )
        except RuntimeError as e:
            raise FunctionCallException(str(e)) from e

        return f"Prefix '{short}' deleted and no longer available for subsequent function calls."

    def update_prefix(self, manager: KgManager, short: str, namespace: str) -> str:
        if short not in self.state.prefixes:
            raise FunctionCallException(f"Prefix '{short}' does not exist.")

        try:
            self.apply_prefixes(manager, {**self.state.prefixes, short: namespace})
        except RuntimeError as e:
            raise FunctionCallException(str(e)) from e

        return f"Prefix '{short}' updated and available for subsequent function calls."

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    def setup(self, input: dict) -> str:
        assert isinstance(input, dict), "Input for auto-setup must be a dict"
        self.input = input

        manager = self.managers[0]
        if self.input["phase"] == "index":
            self.state = load_index_state(manager, self.input["name"])
            return f"""\
Set up the index and info SPARQLs for {self.input["name"]} of the {manager.kg} knowledge graph.

Additional notes:
{self.input.get("notes")}"""

        else:
            self.state = load_info_state(manager)
            return f"""\
Set up the prefixes and description for the {manager.kg} knowledge graph.

Additional notes:
{self.input.get("notes")}"""

    def output(self, messages: list[Message]) -> dict:
        if self.input["phase"] == "index":
            assert isinstance(self.state, IndexState)
            return {
                "type": "output",
                "phase": "index",
                "name": self.input["name"],
                "index": self.state.index_sparql,
                "info": self.state.info_sparql,
                "description": self.state.description,
            }
        else:
            assert isinstance(self.state, InfoState)
            return {
                "type": "output",
                "phase": "info",
                "description": self.state.description,
                "prefixes": self.state.prefixes,
            }
