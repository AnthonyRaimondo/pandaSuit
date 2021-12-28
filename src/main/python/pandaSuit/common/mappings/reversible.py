import pandas

from pandaSuit.common.constant.decorators import UPDATE, APPEND, INSERT, REMOVE, SELECT
from pandaSuit.common.util.list_operations import create_index_list, find_index


def intermediate_update_args(kwargs: dict) -> dict:
    return {
        "column": kwargs.get("column"),
        "row": kwargs.get("row"),
        "pandas_return_type": True
    }


def intermediate_remove_args(kwargs: dict) -> dict:
    return {
        "column": kwargs.get("column"),
        "row": kwargs.get("row"),
        "pandas_return_type": True
    }


def update_args(**kwargs) -> dict:
    return {
        "column": kwargs.get("arguments").get("column"),
        "row": kwargs.get("arguments").get("row"),
        "to": kwargs.get("arguments").get("to")
    }


def append_args(**kwargs) -> dict:
    if "row" in kwargs.get("arguments"):
        return {"row": -1}
    else:
        return {"column": -1}


def insert_args(**kwargs) -> dict:
    index = kwargs.get("arguments").get("index")
    row_arg, column_arg = kwargs.get("arguments").get("row"), kwargs.get("arguments").get("column")
    if row_arg is not None:
        if isinstance(row_arg, pandas.DataFrame) and isinstance(index, int):
            return {"row": create_index_list(start=index, stop=index+row_arg.shape[0])}
        else:
            return {"row": index}
    else:
        if isinstance(column_arg, pandas.DataFrame) and isinstance(index, int):
            return {"column": create_index_list(start=index, stop=index+column_arg.shape[1])}
        else:
            return {"column": index}


def remove_args(**kwargs) -> dict:
    row_arg = kwargs.get("arguments").get("row")
    column_arg = kwargs.get("arguments").get("column")
    if row_arg is not None:
        if isinstance(row_arg, str):
            return {"index": find_index(kwargs.get("df").__getattribute__("row_names"), row_arg)}
        else:
            return {"index": row_arg}
    elif column_arg is not None:
        if isinstance(column_arg, str):
            return {"index": find_index(kwargs.get("df").__getattribute__("column_names"), column_arg)}
        else:
            return {"index": column_arg}


REVERSE_MAPPING = {
    UPDATE: UPDATE,
    APPEND: REMOVE,
    INSERT: REMOVE,
    REMOVE: INSERT
}

REVERSE_ARGS = {
    UPDATE: update_args,
    APPEND: append_args,
    INSERT: insert_args,
    REMOVE: remove_args
}

INTERMEDIATE_REVERSE_FUNCTION_MAPPING = {
    UPDATE: SELECT,
    REMOVE: SELECT
}

INTERMEDIATE_REVERSE_ARGS = {
    UPDATE: intermediate_update_args,
    REMOVE: intermediate_remove_args
}


def update_reverse_args(kwargs: dict, old_value: object) -> dict:
    kwargs.update({"to": old_value})
    return kwargs


def remove_reverse_args(kwargs: dict, old_value: object) -> dict:
    if "row" in kwargs:
        kwargs.update({"row": old_value})
    else:
        kwargs.update({"column": old_value})
    return kwargs


INTERMEDIATE_ARGUMENT_MAPPING = {
    UPDATE: update_reverse_args,
    REMOVE: remove_reverse_args
}
