from typing import Any

from pydantic import BaseModel

from grasp.configs import GraspConfig, NotesConfig, NotesFromExplorationConfig
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.exploration.functions import call_function as call_note_function
from grasp.tasks.exploration.structural.functions import (
    mark_explored,
    note_function_definitions,
    show_explored,
)
from grasp.tasks.functions import find_frequent, find_frequent_function_definition
from grasp.utils import format_kg_notes, format_notes


class StructuralExplorationState(BaseModel):
    notes: list[str] = []
    kg_notes: dict[str, list[str]] = {}
    explored: dict[str, list[str]] = {}


def rules() -> list[str]:
    return [
        "Frequently connected class-like nodes are typically better \
seed nodes than sparsely connected instance-like nodes.",
        "As you hit the limits on the number of notes and their length, \
gradually generalize your notes, discard unnecessary details, and move \
notes that can be useful across knowledge graphs to the general section.",
        "The exploration- and note-specific functions including find_frequent \
are only available during exploration, but not for downstream tasks, so do not \
take notes about them and their usage.",
    ]


def system_information(config: GraspConfig) -> str:
    assert isinstance(config, NotesFromExplorationConfig)
    return f"""\
You are a note-taking assistant. Your task is to \
explore knowledge graphs around selected seed nodes \
and take notes about them using the provided functions.

You are limited to a maximum of {config.max_notes} notes \
per knowledge graph, plus {config.max_notes} general notes for insights that apply \
across knowledge graphs. Each note is limited to a maximum of \
{config.max_note_length} characters to ensure it is concise and to the point.

Your notes should help you to better understand and navigate the \
knowledge graphs in the future. The notes should generalize and be useful for \
answering all kinds of questions about the knowledge graphs, rather than \
being specific to the seed nodes you explore.

You should follow a step-by-step approach to take notes:
1. Look at the current notes and already explored seed nodes across \
all knowledge graphs to figure out well covered and underexplored areas.
2. Determine a seed node in an underexplored area of one of the knowledge graphs. \
Avoid previously explored nodes or nodes very similar to them.
3. Thoroughly explore the seed nodes' neighborhood in the graph. You can also come \
up with questions targeting this area and try to build SPARQL queries to answer them. \
Make sure to use all of the provided functions during your exploration, and \
take notes about your findings along the way.
4. If there are no more open questions and insights to be gained from exploring \
the seed node, mark it as explored. Before stopping, check all notes (not only \
the ones touched in this exploration) for the above mentioned criteria and clean \
them if needed.

Examples of potentially useful types of notes include:
- overall structure, domain coverage, and schema of the knowledge graphs
- peculiarities of the knowledge graphs
- strategies when encountering certain types of questions or errors
- tips for when and how to use certain functions"""


def output(state: StructuralExplorationState) -> dict:
    formatted = f"""\
Exploration completed.

Knowledge graph specific notes:
{format_kg_notes(state.kg_notes)}

General notes across knowledge graphs:
{format_notes(state.notes)}"""

    return {
        "type": "output",
        "notes": state.notes,
        "kg_notes": state.kg_notes,
        "explored": state.explored,
        "formatted": formatted,
    }


class StructuralExplorationTask(GraspTask):
    name = "exploration_structural"

    def system_information(self) -> str:
        return system_information(self.config)

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        kgs = [m.kg for m in self.managers]
        functions = note_function_definitions(self.managers)
        functions.append(
            find_frequent_function_definition(
                kgs,
                self.config.list_k,
                extra_params={
                    "exclude_explored": {
                        "type": "boolean",
                        "description": "If true, exclude previously explored "
                        "seed nodes from the results",
                    },
                },
            )
        )
        return functions

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        assert isinstance(self.config, NotesConfig)
        assert self.state is not None, "State must be provided for exploration task"

        if fn_name == "mark_explored":
            result = mark_explored(
                self.managers,
                fn_args["kg"],
                self.state.explored,
                fn_args["iri"],
                self.explored_this_round,
            )
            self.explored_this_round = True
            return result

        if fn_name == "show_explored":
            return show_explored(
                self.managers,
                fn_args["kg"],
                self.state.explored,
                fn_args["page"],
                self.config.list_k,
            )

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

        return call_note_function(
            self.state.kg_notes,
            self.state.notes,
            fn_name,
            fn_args,
            self.config.max_notes,
            self.config.max_note_length,
        )

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    def setup(self, input: Any) -> str:
        assert isinstance(input, StructuralExplorationState), (
            "Input for exploration must already be a StructuralExplorationState"
        )
        self.state = input
        self.explored_this_round = False
        return "Choose a seed node and start the exploration. \
Add to, delete from, or update the current notes along the way."

    def output(self, messages: list[Message]) -> dict:
        return output(self.state)
