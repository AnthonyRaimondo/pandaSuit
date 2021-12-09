from pandaSuit.common.constant.decorators import UPDATE


def intermediate_update_args(kwargs):
    return {
        "column": kwargs.get("column")
    }


def update_args(kwargs):
    return {
        "column": kwargs.get("column"),
        "to": kwargs.get("to")
    }


REVERSE_MAPPING = {
    UPDATE: "update"
}

REVERSE_ARGS = {
    UPDATE: update_args
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
