import numpy as np


def find_indexes(list_to_search: list, value: object) -> list:
    return [index for index, element in enumerate(list_to_search) if element == value]


def index_dictionary(list_of_values: list) -> dict:
    return {value: find_indexes(list_of_values, value) for value in list_of_values}


def create_index_list(stop: int, start: int = 0, step: int = 1) -> list:
    return list(map(int, np.arange(start=start, stop=stop, step=step)))


def remove_elements(base_list: list, exclude: list or object) -> list:
    return [element for element in base_list if element not in exclude] \
        if isinstance(exclude, list) else [element for element in base_list if element != exclude]


def remove_elements_from_dict(base_dict: dict, exclude: list or object) -> dict:
    return {key: value for key, value in base_dict.items() if key not in exclude} \
        if isinstance(exclude, list) else {key for key, value in base_dict.items() if key != exclude}


def combinations(list1, list2):
    return [(i, j) for j in list2 for i in list1]


def filter_and_find_indexes(base_list: list, exclude: list or object) -> list:
    return [find_indexes(base_list, element)[0] for element in base_list if element not in exclude] \
        if isinstance(exclude, list) else \
        [find_indexes(base_list, element)[0] for element in base_list if element != exclude]
