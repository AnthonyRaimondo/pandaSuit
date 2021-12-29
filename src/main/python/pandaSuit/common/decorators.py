import functools
import inspect
from collections import deque
from copy import copy
from typing import Mapping

from pandaSuit.common.mappings.reversible import *
from pandaSuit.common.unwind import Unwind
from pandaSuit.common.constant.decorators import UNWIND_LIST


def extract_function_name(function_reference: str) -> str:
    return function_reference.split(".")[1].split(" at")[0]


def in_place_operation(args: tuple, kwargs: dict, signature: inspect.Signature) -> bool:
    # search for in_place parameter in positional arguments
    if len(args) > 0:
        for count, arg in enumerate(args):
            if list(signature.parameters.keys())[count] == "in_place":  # if in_place parameter is passed
                return arg

    # search for in_place parameter in keyword arguments
    if kwargs.get("in_place") is not None:
        return kwargs.get("in_place")

    # otherwise, use default for in_place parameter
    return signature.parameters.get("in_place").default


def infer_kwargs(args: tuple, signature: inspect.Signature) -> dict:
    kwargs = {}
    for count, arg in enumerate(args):
        kwargs[list(signature.parameters.keys())[count]] = args[count]
    return kwargs


def get_method_signature(df_object: object, name: str) -> inspect.Signature:
    return inspect.signature(df_object.__getattribute__(name))


def get_method_parameters(df_object: object, name: str) -> Mapping[str, inspect.Parameter]:
    return get_method_signature(df_object, name).parameters


def reversible(func):
    """Allow for reversing an 'in place' operation on pandaSuit object"""
    @functools.wraps(func)
    def wrapper_reverse(*args, **kwargs):
        caller_function = inspect.stack()[1][3]
        if caller_function != "undo":  # this occurs when a @reversible method is un-done by another @reversible method
            function_name = extract_function_name(func.__repr__())
            df_object = args[0]
            method_signature = get_method_signature(df_object, function_name)
            if not in_place_operation(args[1:], kwargs, method_signature):
                return func(*args, **kwargs)  # don't create unwind step, but return value from method called
            else:
                if len(args) > 1:  # convert positional args into keyword args
                    kwargs.update(infer_kwargs(args[1:], method_signature))
                    args = (df_object,)  # remove positional args to avoid passing parameters multiple times when calling the function
                intermediate_reverse_function = INTERMEDIATE_REVERSE_FUNCTION_MAPPING.get(function_name)
                reverse_args = REVERSE_ARGS.get(function_name)(df=df_object, arguments=kwargs)
                if intermediate_reverse_function is not None:  # some DF manipulations require an intermediate step to reverse (e.g., .update() & .remove())
                    intermediate_reverse_args = INTERMEDIATE_REVERSE_ARGS.get(function_name)(kwargs)
                    reverse_args_to_add = df_object.__getattribute__(intermediate_reverse_function)(**intermediate_reverse_args)
                    intermediate_arg_mapping_function = INTERMEDIATE_ARGUMENT_MAPPING.get(function_name)
                    reverse_args.update(intermediate_arg_mapping_function(copy(kwargs), copy(reverse_args_to_add)))
                reverse_function = REVERSE_MAPPING.get(function_name)
                if "in_place" in get_method_parameters(df_object, reverse_function):
                    reverse_args.update({"in_place": True})
                df_object.__setattr__(UNWIND_LIST, df_object.__getattribute__(UNWIND_LIST) + deque([Unwind(reverse_function, reverse_args)]))
        func(*args, **kwargs)
    return wrapper_reverse
