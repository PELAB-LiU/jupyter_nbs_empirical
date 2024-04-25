from uuid import uuid3, NAMESPACE_DNS, UUID
from pathlib import Path


def generate_uuid_for_nb_exception(
    file_name: Path | str, nb_cell_index: int, exception_name: str
) -> UUID:
    """
    Generates a unique ID of an exception, using `uuid3` and the DNS
    namespace. This process is not case-sensitive.
    :param file_name: the path in which the exception was thrown.
    This can be any type of path, only the file name and file extension
    are used.
    :param cell_index: The cell in which the exception was raised. This
    is the cell index in the note book (i.e., counting markdown cells
    etc. as well). This is zero-indexed.
    :param exception_name: The type of the exception.
    """

    if isinstance(file_name, str):
        file_name = Path(file_name)

    file_name = file_name.name
    nb_cell_index = str(nb_cell_index)

    entry = f"nb_exception://{file_name}.{nb_cell_index}.{exception_name}"
    entry = entry.lower()

    entry_id = uuid3(NAMESPACE_DNS, entry)
    return entry_id
