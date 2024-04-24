"""
Contains the basic data structure used to represent a notebook exception.
NotebookException
    ExceptionExceptionStacktrace
        StacktraceEntry [Notebook | File]
"""

from dataclasses import dataclass
import regex as re
from typing import List, Dict, Tuple, Iterator, Callable
from pathlib import Path
import json

_EXCEPTION_STACKTRACE_SPLIT = (
    "During handling of the above exception, another exception occurred:"
)
_EXCEPTION_STACKTRACE_CAUSE = (
    "The above exception was the direct cause of the following exception:"
)
_SKIP_LINE_ENTRY = "(...)"
_STACK_ORDER_RECENTLAST = "Traceback (most recent call last)"
_SKIP_SIMILAR_FRAME_PREFIX = "[... skipping similar frames:"
_SKIP_HIDDEN_FRAME_PREFIX = "[... skipping hidden"


@dataclass(frozen=True)
class StacktraceEntry:
    exception_line_number: int
    previewed_lines: Dict[int, str]


@dataclass(frozen=True)
class NotebookStacktraceEntry(StacktraceEntry):
    cell_index: int


@dataclass(frozen=True)
class FileStacktraceEntry(StacktraceEntry):
    file_path: str
    function_signature: str


@dataclass(frozen=True)
# TODO: Remove this and use OtherStacktraceEntry instead?
class SkippedStacktraceEntry(StacktraceEntry):
    method_name: str
    times: int


@dataclass(frozen=True)
# TODO: Remove this and use OtherStacktraceEntry instead?
class InternalMethodTraceEntry(StacktraceEntry):
    internal_method_name: str


@dataclass(frozen=True)
class OtherStacktraceEntry(StacktraceEntry):
    details: dict


@dataclass(frozen=True)
class ExceptionStacktrace:
    exception_type: str
    exception_message: str
    stacktrace_order: str
    # An ordered list containing all stacktraces in the exception.
    stacktrace_entries: List[StacktraceEntry]


@dataclass(frozen=True)
class RawNotebookException:
    """Represents the raw exception entry contained in a notebook."""

    stacktrace: str
    exception_name: str
    exception_message: str
    cell_index: int
    notebook_path: Path


@dataclass(frozen=True)
class NotebookException:
    """Data class representing an exception thrown in a Python notebook."""

    # An ordered list containing all inner exceptions.
    # i.e., all the entries separated by "During handling of the above exception, another exception occurred:"
    inner_errors: List[ExceptionStacktrace]
    source: str | RawNotebookException


def get_raw_notebook_exceptions_from(
    notebook_path: Path,
) -> Iterator[RawNotebookException]:
    """Retrieves a list of cell exceptions from the provided notebook."""

    with open(notebook_path, "r", encoding="utf-8") as notebook_file:
        nb = json.loads(notebook_file.read())

    for cell_index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code" or "outputs" not in cell:
            continue

        for output in cell["outputs"]:
            if "traceback" not in output:
                continue

            stacktrace = output["traceback"]
            if isinstance(stacktrace, list):
                stacktrace = "\n".join(stacktrace)

            # Parses ANSII
            ansii_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
            stacktrace = ansii_escape.sub("", stacktrace)

            exception = RawNotebookException(
                stacktrace=stacktrace,
                exception_name=output["ename"],
                exception_message=output["evalue"],
                cell_index=cell_index,
                notebook_path=notebook_path,
            )
            yield exception


def get_cause_iterator(
    notebook_exception: NotebookException,
) -> Iterator[StacktraceEntry]:
    for inner_exception in notebook_exception.inner_errors[::-1]:
        yield from inner_exception.stacktrace_entries


def get_root_exception(
    notebook_exception: NotebookException,
) -> StacktraceEntry | None:
    """Returns the root cause of the notebook's exception."""

    inner_error_root = notebook_exception.inner_errors[-1]

    if inner_error_root.stacktrace_order == _STACK_ORDER_RECENTLAST:
        inner_error_root = notebook_exception.inner_errors[-1]
        last_notebook_entry = None
        for stacktrace_entry in inner_error_root.stacktrace_entries:
            if isinstance(stacktrace_entry, NotebookStacktraceEntry):
                last_notebook_entry = stacktrace_entry
            else:
                break
        return last_notebook_entry
    else:
        print(f'Found unsupported order type "{inner_error_root.stacktrace_order}"')


def try_parse_notebook_exception(
    stacktrace: str = None,
    raw_exception: RawNotebookException = None,
) -> Tuple[bool, NotebookException | Exception]:
    """Helper method that tries to parse the stacktrace to a NotebookException,
    yielding it if it was done successfully, or yielding the exception in case
    it failed."""

    try:
        nb_exception = parse_notebook_exception(stacktrace, raw_exception)
        return True, nb_exception
    except Exception as ex:
        return False, ex


def parse_notebook_exception(
    stacktrace: str = None,
    raw_exception: RawNotebookException = None,
) -> NotebookException:
    """Factory method to parse a single notebook stacktrace into a `NotebookException` object."""
    if stacktrace is None and raw_exception is None:
        raise ValueError("`stacktrace` and `raw_exception` cannot both be `None`.")

    if not raw_exception is None:
        stacktrace = raw_exception.stacktrace
        source = raw_exception
    else:
        source = stacktrace

    stacktrace = re.split(r"\r\n|\r|\n", stacktrace)
    stacktrace = [line for line in stacktrace if line.strip() != ""]

    if stacktrace[0].startswith("---------"):
        stacktrace = stacktrace[1:]

    stacktrace_segments = __identify_stacktrace_segments(stacktrace)
    stacktraces: List[ExceptionStacktrace] = []
    for segment_start, segment_end in stacktrace_segments:
        exception_stacktrace = __parse_stacktrace_segment(
            stacktrace[segment_start:segment_end]
        )
        stacktraces.append(exception_stacktrace)

    return NotebookException(inner_errors=stacktraces, source=source)


def __identify_stacktrace_segments(
    stacktrace: List[str],
) -> Iterator[Tuple[int, int]]:
    """
    Identifies the line ranges in which a single exception is logged.
    Inner exceptions (i.e., those separated with sentences like:
    'During handling of the above exception, another exception occurred:'.)
    """
    stacktrace_start = 0
    for current, line in enumerate(stacktrace):
        if (
            not _EXCEPTION_STACKTRACE_SPLIT in line
            and not _EXCEPTION_STACKTRACE_CAUSE in line
        ):
            continue
        segment_start = (
            stacktrace_start if stacktrace_start == 0 else stacktrace_start + 1
        )
        entry = (segment_start, current)
        stacktrace_start = current
        yield entry

    stacktrace_end = len(stacktrace)
    if stacktrace_end == stacktrace_start:
        return

    if stacktrace_start == 0:
        yield (stacktrace_start, stacktrace_end)
    else:
        yield (stacktrace_start + 1, stacktrace_end)


def __parse_stacktrace_segment(stacktrace: List[str]) -> ExceptionStacktrace:
    """Parses the stacktraces of a single stacktrace segment (i.e.,
    those identified in `__identify_stacktrace_segments`.)"""
    # Parses exception type and the stacktrace order.
    type_and_order = stacktrace[0].split()
    exception_type = type_and_order[0]
    exception_order = " ".join(type_and_order[1:])

    last_unindented_line_index = None
    for index, line in enumerate(stacktrace):
        if line.startswith(f"{exception_type}:"):
            last_unindented_line_index = index
            break

    # # Sometimes, messages contain new lines, but they're always at the end.
    # last_unindented_line_index = None
    # for index, line in enumerate(stacktrace):
    #     # Searches for the last entry without tabs at the start.
    #     if line.strip() == line:
    #         last_unindented_line_index = index
    type_and_message = stacktrace[last_unindented_line_index:]
    type_and_message = " ".join(type_and_message)

    # # Adds lines until the error type is included in the message.
    # while f"{exception_type}:" not in type_and_message:
    #     last_unindented_line_index -= 1
    #     type_and_message = (
    #         f"{stacktrace[last_unindented_line_index]} {type_and_message}"
    #     )

    type_and_message = type_and_message.split(":")
    exception_message = ":".join(type_and_message[1:]).strip()

    # Identifies the different components within the stacktrace.
    stacktrace = stacktrace[1:last_unindented_line_index]
    call_stacktrace_segments = __identify_call_stacktrace_segments(stacktrace)

    # Builds the indiviual stacktrace entries.
    stacktraces = []
    for segment_start, segment_end in call_stacktrace_segments:
        if segment_start == segment_end:
            continue
        stacktrace_entry = __parse_call_stacktrace_segment(
            stacktrace[segment_start:segment_end]
        )
        stacktraces.append(stacktrace_entry)

    return ExceptionStacktrace(
        exception_type, exception_message, exception_order, stacktraces
    )


def __identify_call_stacktrace_segments(
    stacktrace: List[str],
) -> Iterator[Tuple[int, int]]:
    """Identifies the line ranges of a single call stacktrace segment
    (i.e., the window in which lines are shown)."""
    stacktrace_start = 0
    for current, line in enumerate(stacktrace):
        if current == 0:
            continue
        # NOTE: This if-tree is tightly coupled with the if statement in `__parse_call_stacktrace_segment`.
        if not (
            line.startswith("Cell")
            or line.startswith("Input")
            or line.startswith("File")
            or re.match(re.compile(r"<ipython-input-(\d)+-\w+>"), line)
            or line.startswith("/")
            or line.startswith("pandas")
            or ".pyx" in line
            or line.strip().startswith(_SKIP_SIMILAR_FRAME_PREFIX)
            or line.strip().startswith(_SKIP_HIDDEN_FRAME_PREFIX)
            or line.startswith("<timed")
            or re.match(r"~\\AppData\\Local\\Temp\\ipykernel_\d+\\\d+\.py.*", line)
            or line.startswith("~")
            or "anaconda3" in line
            or re.match(r"<(.*)?internals> in (.*)", line)
        ):
            continue
        if stacktrace_start != current:
            entry = (stacktrace_start, current)
            yield entry
        stacktrace_start = current

    stacktrace_end = len(stacktrace)
    if stacktrace != stacktrace_start:
        yield (stacktrace_start, stacktrace_end)


def __parse_call_stacktrace_segment(stacktrace: List[str]) -> StacktraceEntry:
    """Parses a single call stacktrace segment (i.e., those identified in
    `__identify_call_stacktrace_segments`)."""
    origin = stacktrace[0].split()
    origin_type = origin[0]

    # Builds the subclass specific parameters.
    stack_entry_kwargs = {}
    concrete_stacktrace_entry_type: Callable[[], StacktraceEntry] = None

    # NOTE: This if-tree is tightly coupled with the if statement in `__identify_call_stacktrace_segments`.
    if origin_type == "File":
        concrete_stacktrace_entry_type = FileStacktraceEntry
        stack_entry_kwargs["file_path"] = origin[1].split(":")[0]
        stack_entry_kwargs["function_signature"] = " ".join(origin[3:])
    elif origin_type == "Cell":
        concrete_stacktrace_entry_type = NotebookStacktraceEntry
        if origin[1] == "In":
            stack_entry_kwargs["cell_index"] = int(origin[2][1:-2])
        else:
            stack_entry_kwargs["cell_index"] = int(origin[1][3:-2])
    elif origin_type == "Input":
        concrete_stacktrace_entry_type = NotebookStacktraceEntry
        stack_entry_kwargs["cell_index"] = int(origin[2][1:-2])
    elif cell_index := re.match(re.compile(r"<ipython-input-(\d)+-\w+>"), origin_type):
        concrete_stacktrace_entry_type = NotebookStacktraceEntry
        stack_entry_kwargs["cell_index"] = cell_index.group(0)
    elif origin_type.startswith("/tmp/") or origin_type.startswith("/var/"):
        concrete_stacktrace_entry_type = NotebookStacktraceEntry
        stack_entry_kwargs["cell_index"] = None
    elif (
        origin_type.startswith("/opt/")
        or origin_type.startswith("/usr/")
        or origin_type.startswith("/kaggle/")
        or origin_type.startswith("pandas/")
    ):
        concrete_stacktrace_entry_type = FileStacktraceEntry
        stack_entry_kwargs["file_path"] = origin[0]
        stack_entry_kwargs["function_signature"] = " ".join(origin[2:])
    elif origin_type.endswith(".pyx"):
        concrete_stacktrace_entry_type = FileStacktraceEntry
        stack_entry_kwargs["file_path"] = origin[0]
        stack_entry_kwargs["function_signature"] = " ".join(origin[2:])
    elif (
        origin_type.strip().startswith("[...")
        and origin[1] == "skipping"
        and origin[2] == "similar"
    ):
        concrete_stacktrace_entry_type = OtherStacktraceEntry
        # entry = origin[len(concrete_stacktrace_entry_type) :].split()
        stack_entry_kwargs["details"] = {
            "type": "similar layers",
            "method_name": origin[4],
            "times": int(origin[8][1:]),
        }
        stack_entry_kwargs["exception_line_number"] = origin[7]
    elif (
        origin_type.strip().startswith("[...")
        and origin[1] == "skipping"
        and origin[2] == "hidden"
    ):
        concrete_stacktrace_entry_type = OtherStacktraceEntry
        stack_entry_kwargs["exception_line_number"] = None
        stack_entry_kwargs["details"] = {
            "type": "hidden layers",
            "number": int(origin[3]),
        }
    elif re.match(r"~\\AppData\\Local\\Temp\\ipykernel_\d+\\\d+\.py.*", origin_type):
        # TODO: Sometimes this mentions the underlying method that failed. Consider including this.
        concrete_stacktrace_entry_type = NotebookStacktraceEntry
        stack_entry_kwargs["cell_index"] = None
    elif origin_type.startswith("~"):
        concrete_stacktrace_entry_type = FileStacktraceEntry
        stack_entry_kwargs["file_path"] = origin[0]
        stack_entry_kwargs["function_signature"] = " ".join(origin[2:])
    elif "anaconda3" in origin_type:
        concrete_stacktrace_entry_type = FileStacktraceEntry
        stack_entry_kwargs["file_path"] = origin[0]
        stack_entry_kwargs["function_signature"] = " ".join(origin[2:])
    elif internal_method := re.match(r"<(.*)?internals> in (.*)", stacktrace[0]):
        concrete_stacktrace_entry_type = InternalMethodTraceEntry
        # TODO: There is probably a neater way to do this, but this information is not relevant for now.
        stack_entry_kwargs["internal_method_name"] = (
            f"{internal_method.group(0)}, {internal_method.group(1)}"
        )
    elif origin_type == "_RemoteTraceback:":
        # TODO: I am uncertain how unique this is.
        concrete_stacktrace_entry_type = OtherStacktraceEntry
        stack_entry_kwargs["details"] = {"traceback": stacktrace}
    elif origin_type == "<timed":
        concrete_stacktrace_entry_type = OtherStacktraceEntry
        stack_entry_kwargs["details"] = {"traceback": stacktrace}
    else:
        raise NotImplementedError(f"Unsupported origin type '{origin_type}'.")

    # Builds the generic parameters.
    previewed_lines = {}
    exception_line_number = None

    if concrete_stacktrace_entry_type != OtherStacktraceEntry:
        window_stacktrace = stacktrace[1:]
        for window_line in window_stacktrace:
            line_number, line, is_exception_line = __parse_window_line(window_line)
            if not line_number:
                continue
            previewed_lines[line_number] = line
            if is_exception_line:
                exception_line_number = line_number

    # Builds entry.
    stack_entry_kwargs_default = {
        "exception_line_number": exception_line_number,
        "previewed_lines": previewed_lines,
    }
    stack_entry_kwargs_default.update(stack_entry_kwargs)
    stacktrace_entry = concrete_stacktrace_entry_type(
        **stack_entry_kwargs_default,
    )
    return stacktrace_entry


def __parse_window_line(line: str) -> Tuple[int, str, bool]:
    if line.strip() == _SKIP_LINE_ENTRY:
        return None, None, None
    terms = line.split()
    # If it cannot parse the first element, it probably starts with an --->
    try:
        line_number = int(terms[0])
        terms = terms[1:]
        is_exception = False
    except:
        line_number = int(terms[1])
        is_exception = True
        terms = terms[2:]
    line = " ".join(terms)
    return line_number, line, is_exception


if __name__ == "__main__":
    """
    Tests the code by loading all of the kaggle notebooks, identifying
    all of the exceptions in them, filtering exceptions that are part
    of the exclusion list, and attempting to parse the leftover
    exceptions.
    """

    from os import path
    from wmutils.file import iterate_through_files_in_nested_folders
    from config import builtin_exps_excluded

    # Having a static path helps with debugging a specific notebook.
    static_path = None
    # static_path = "./data/notebooks/nbdata_err_kaggle/nbdata_err_kaggle/nbdata_k_error/nbdata_k_error/230102/adeelsoomro00_introduction-to-python.ipynb"
    if not static_path:
        # Loads all notebooks.
        files = list(
            file
            for file in iterate_through_files_in_nested_folders(
                "./data/notebooks/nbdata_err_kaggle/nbdata_err_kaggle/nbdata_k_error/nbdata_k_error/",
                10_000,
            )
            if path.isfile(file)
        )
    else:
        files = [static_path]
    files = [Path(path) for path in files]
    print(f"Loaded {len(files)} notebooks.")

    # Parses the exceptions in those notebooks.
    failed = 0
    total = 0

    # creating exclusion list.
    skipped_exception_types = set(builtin_exps_excluded)
    skipped_exception_types = skipped_exception_types.union(
        ["syntaxerror", "indentationerror"]
    )

    for index, file in enumerate(files):

        # Finds all exceptions in the notebook file and attempts to parse them.
        raw_excs = get_raw_notebook_exceptions_from(file)
        for raw_exc in raw_excs:

            # Applies exclusion criteria.
            if raw_exc.exception_name.lower() in skipped_exception_types:
                continue

            success, nb_exc = try_parse_notebook_exception(raw_exception=raw_exc)

            if not success:
                failed += 1
                print(index, file, raw_exc.exception_name)
            total += 1

    print(
        f"Parsed {total - failed}/{total} ({(total-failed)/total * 100:.2f}%) exceptions successfully."
    )
