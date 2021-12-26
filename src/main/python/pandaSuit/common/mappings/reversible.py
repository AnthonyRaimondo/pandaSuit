import pandas

from pandaSuit.common.constant.decorators import UPDATE, APPEND, INSERT
from pandaSuit.common.util.list_operations import create_index_list


def intermediate_update_args(kwargs: dict) -> dict:
    return {
        "column": kwargs.get("column"),
        "row": kwargs.get("row"),
        "pandas_return_type": True
    }


def update_args(kwargs: dict) -> dict:
    return {
        "column": kwargs.get("column"),
        "row": kwargs.get("row"),
        "to": kwargs.get("to")
    }


def append_args(kwargs: dict) -> dict:
    if "row" in kwargs:
        return {
            "row": -1
        }
    else:
        return {
            "column": -1
        }


def insert_args(kwargs: dict) -> dict:
    index = kwargs.get("index")
    if kwargs.get("row") is not None:
        if isinstance(kwargs.get("row"), pandas.DataFrame):
            args = {"row": create_index_list(start=index, stop=index+kwargs.get("row").shape[0])}
        else:
            args = {"row": index}
    else:
        if isinstance(kwargs.get("column"), pandas.DataFrame):
            args = {"column": create_index_list(start=index, stop=index+kwargs.get("column").shape[1])}
        else:
            args = {"column": index}
    return args


REVERSE_MAPPING = {
    UPDATE: "update",
    APPEND: "remove",
    INSERT: "remove"
}

REVERSE_ARGS = {
    UPDATE: update_args,
    APPEND: append_args,
    INSERT: insert_args
}

INTERMEDIATE_REVERSE_MAPPING = {
    UPDATE: "select"
}

INTERMEDIATE_REVERSE_ARGS = {
    UPDATE: intermediate_update_args
}

ARGUMENT_MAPPING = {
    UPDATE: "to"
}
