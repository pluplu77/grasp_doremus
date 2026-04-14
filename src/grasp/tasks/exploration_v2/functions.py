from grasp.functions import find_manager, verify_iri_or_literal
from grasp.manager import KgManager
from grasp.utils import FunctionCallException, clip, format_enumerate, format_notes


def note_functions(managers: list[KgManager]) -> list[dict]:
    kgs: list[str | None] = [manager.kg for manager in managers]
    kgs.append(None)
    return [
        {
            "name": "add_note",
            "description": "Add a general or knowledge graph specific note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": ["string", "null"],
                        "enum": kgs,
                        "description": "The knowledge graph for which to add the note (null for general notes)",
                    },
                    "note": {
                        "type": "string",
                        "description": "The note to add",
                    },
                },
                "required": ["kg", "note"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "delete_note",
            "description": "Delete a general or knowledge graph specific note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": ["string", "null"],
                        "enum": kgs,
                        "description": "The knowledge graph for which to delete the note (null for general notes)",
                    },
                    "num": {
                        "type": "number",
                        "description": "The number of the note to delete",
                    },
                },
                "required": ["kg", "num"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "update_note",
            "description": "Update a general or knowledge graph specific note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": ["string", "null"],
                        "enum": kgs,
                        "description": "The knowledge graph for which to update the note (null for general notes)",
                    },
                    "num": {
                        "type": "number",
                        "description": "The number of the note to update",
                    },
                    "note": {
                        "type": "string",
                        "description": "The new note replacing the old one",
                    },
                },
                "required": ["kg", "num", "note"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "show_notes",
            "description": "Show current general or knowledge graph specific notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": ["string", "null"],
                        "enum": kgs,
                        "description": "The knowledge graph for which to show the notes (null for general notes)",
                    },
                },
                "required": ["kg"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "mark_explored",
            "description": "Mark an IRI as the seed node for this exploration round. "
            "Exactly one seed node must be marked per round.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs[:-1],  # cannot be null for this function
                        "description": "The knowledge graph of the seed node to mark as explored",
                    },
                    "iri": {
                        "type": "string",
                        "description": "The IRI of the seed node to mark as explored",
                    },
                },
                "required": ["kg", "iri"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "show_explored",
            "description": "Show previously explored seed nodes, most recent first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs[:-1],  # cannot be null for this function
                        "description": "The knowledge graph for which to show explored seed nodes",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number (1-indexed) for paginating results (default should be 1)",
                    },
                },
                "required": ["kg", "page"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "stop",
            "description": "Stop the note taking process.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


def check_note(note: str, max_note_length: int) -> None:
    if len(note) > max_note_length:
        raise FunctionCallException(
            f"Note exceeds maximum length of {max_note_length} characters"
        )


def show_notes(notes: list[str]) -> str:
    return format_notes(notes, enumerated=True)


def add_note(notes: list[str], note: str, max_notes: int, max_note_length: int) -> str:
    if len(notes) >= max_notes:
        raise FunctionCallException(f"Cannot add more than {max_notes} notes")

    check_note(note, max_note_length)

    notes.append(note)
    return f"Added note {len(notes)}: {clip(note, 64)}"


def delete_note(notes: list[str], num: int | float) -> str:
    num = int(num)
    if num < 1 or num > len(notes):
        raise FunctionCallException("Note number out of range")

    num -= 1
    note = notes.pop(num)
    return f"Deleted note {num + 1}: {clip(note, 64)}"


def update_note(
    notes: list[str],
    num: int | float,
    note: str,
    max_note_length: int,
) -> str:
    num = int(num)
    if num < 1 or num > len(notes):
        raise FunctionCallException("Note number out of range")

    check_note(note, max_note_length)

    num -= 1
    notes[num] = note
    return f"Updated note {num + 1}: {clip(note, 64)}"


def mark_explored(
    managers: list[KgManager],
    kg: str,
    explored: dict[str, list[str]],
    iri: str,
    explored_this_round: bool,
) -> str:
    if explored_this_round:
        raise FunctionCallException(
            "A seed node was already marked as explored this round"
        )

    manager, _ = find_manager(managers, kg)
    ver_iri = verify_iri_or_literal(iri, manager, allow_literal=False)
    if ver_iri is None:
        raise FunctionCallException(f'"{iri}" is not a valid IRI')

    kg_explored = explored.setdefault(kg, [])
    if ver_iri in kg_explored:
        raise FunctionCallException(
            "This node was already explored in a previous round"
        )

    kg_explored.append(ver_iri[1:-1])
    return f'Marked "{iri}" as explored'


def show_explored(
    managers: list[KgManager],
    kg: str,
    explored: dict[str, list[str]],
    page: int,
    k: int,
) -> str:
    if page < 1:
        raise FunctionCallException("Page number must be at least 1")

    kg_explored = explored.get(kg, [])
    if not kg_explored:
        return "None"

    total_pages = (len(kg_explored) + k - 1) // k
    if page > total_pages:
        raise FunctionCallException(f"Page number exceeds maximum page {total_pages}")

    # most recent first
    kg_explored = list(reversed(kg_explored))

    start = (page - 1) * k
    end = page * k
    page_items = kg_explored[start:end]

    header = f"Most recently explored nodes (page {page} of {total_pages}):\n"

    manager, _ = find_manager(managers, kg)
    infos = manager.get_info_for_identifiers_from_index(page_items, "entities")
    items = []
    for iri in page_items:
        alt = manager.build_alternative_with_info(iri, infos.get(iri))
        items.append(alt.get_selection_string())

    return header + format_enumerate(items, start=start)


def call_function(
    kg_notes: dict[str, list[str]],
    notes: list[str],
    fn_name: str,
    fn_args: dict,
    max_notes: int,
    max_note_length: int,
) -> str:
    if fn_name == "stop":
        return "Stopped process"

    # kg should be there for every function call
    kg = fn_args.get("kg", None)
    if kg is None:
        notes_to_use = notes
    else:
        if kg not in kg_notes:
            kg_notes[kg] = []
        notes_to_use = kg_notes[kg]

    if fn_name == "add_note":
        return add_note(notes_to_use, fn_args["note"], max_notes, max_note_length)
    elif fn_name == "delete_note":
        return delete_note(notes_to_use, fn_args["num"])
    elif fn_name == "update_note":
        return update_note(
            notes_to_use,
            fn_args["num"],
            fn_args["note"],
            max_note_length,
        )
    elif fn_name == "show_notes":
        return show_notes(notes_to_use)
    else:
        raise ValueError(f"Unknown function {fn_name}")
