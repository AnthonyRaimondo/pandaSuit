from pandaSuit.common.constant.decorators import UPDATE, APPEND


def intermediate_update_args(kwargs):
    return {
        "column": kwargs.get("column"),
        "pandas_return_type": True
    }


def update_args(kwargs):
    return {
        "column": kwargs.get("column"),
        "to": kwargs.get("to")
    }


def append_args(kwargs):
    if "row" in kwargs:
        return {
            "row": -1
        }
    else:
        return {
            "column": -1
        }


REVERSE_MAPPING = {
    UPDATE: "update",
    APPEND: "remove"
}

REVERSE_ARGS = {
    UPDATE: update_args,
    APPEND: append_args
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
