from copy import copy

import pandas

from pandaSuit.common.constant import decorators as method_names
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
        else:  # list
            if isinstance(row_arg, list) and any(isinstance(row, str) for row in row_arg):  # row names
                return {"index": [find_index(kwargs.get("df").__getattribute__("row_names"), row) for row in row_arg]}
            else:  # row indexes
                return {"index": row_arg}
    elif column_arg is not None:
        if isinstance(column_arg, str):
            return {"index": find_index(kwargs.get("df").__getattribute__("column_names"), column_arg)}
        else:  # list
            if isinstance(column_arg, list) and any(isinstance(column, str) for column in column_arg):  # column names
                return {"index": [find_index(kwargs.get("df").__getattribute__("column_names"), column) for column in column_arg]}
            else:  # column indexes
                return {"index": column_arg}


def reset_args(**kwargs) -> dict:
    return {"data": copy(kwargs.get("df"))}


def dunder_i_args(**kwargs) -> dict:
    return {"other": kwargs.get("arguments").get("other")}


def i_pow_args(**kwargs) -> dict:
    reciprocal = 1 / kwargs.get("arguments").get("other")
    return {"other": reciprocal}


REVERSE_MAPPING = {
    method_names.UPDATE: method_names.UPDATE,
    method_names.APPEND: method_names.REMOVE,
    method_names.INSERT: method_names.REMOVE,
    method_names.REMOVE: method_names.INSERT,
    method_names.RESET: "_set_underlying_dataframe",  # todo: is there a way to pass __init__ here?
    method_names.IADD: method_names.ISUB,
    method_names.ISUB: method_names.IADD,
    method_names.IMUL: method_names.IDIV,
    method_names.IDIV: method_names.IMUL,
    method_names.IPOW: method_names.IPOW
}

REVERSE_ARGS = {
    method_names.UPDATE: update_args,
    method_names.APPEND: append_args,
    method_names.INSERT: insert_args,
    method_names.REMOVE: remove_args,
    method_names.RESET: reset_args,
    method_names.IADD: dunder_i_args,
    method_names.ISUB: dunder_i_args,
    method_names.IMUL: dunder_i_args,
    method_names.IDIV: dunder_i_args,
    method_names.IPOW: i_pow_args
}

INTERMEDIATE_REVERSE_FUNCTION_MAPPING = {
    method_names.UPDATE: method_names.SELECT,
    method_names.REMOVE: method_names.SELECT
}

INTERMEDIATE_REVERSE_ARGS = {
    method_names.UPDATE: intermediate_update_args,
    method_names.REMOVE: intermediate_remove_args
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
    method_names.UPDATE: update_reverse_args,
    method_names.REMOVE: remove_reverse_args
}
