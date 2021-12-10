import functools
from collections import deque
from copy import copy

from pandaSuit.common.mappings.reversible import *
from pandaSuit.common.unwind import Unwind
from pandaSuit.common.constant.decorators import UNWIND_LIST


def extract_function_name(function_reference: str) -> str:
    return function_reference.split(".")[1].split(" at")[0]


def reversible(func):
    """Allow for reversing an 'in place' operation on pandaSuit object"""
    @functools.wraps(func)
    def wrapper_reverse(*args, **kwargs):
        function_name = extract_function_name(func.__repr__())
        intermediate_reverse_function = INTERMEDIATE_REVERSE_MAPPING.get(function_name)
        intermediate_reverse_args = INTERMEDIATE_REVERSE_ARGS.get(function_name)(kwargs)
        reverse_args = REVERSE_ARGS.get(function_name)(kwargs)
        if intermediate_reverse_function is not None:
            reverse_args.update({
                ARGUMENT_MAPPING.get(function_name):
                    copy(args[0].__getattribute__(intermediate_reverse_function)(**intermediate_reverse_args))
            })
        reverse_function = REVERSE_MAPPING.get(function_name)
        args[0].__setattr__(UNWIND_LIST, args[0].__getattribute__(UNWIND_LIST) + deque([Unwind(reverse_function, reverse_args)]))
        func(*args, **kwargs)
    return wrapper_reverse
