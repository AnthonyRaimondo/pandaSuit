from copy import copy

import pytest
from pandas import Series

from pandaSuit.df import DF


@pytest.fixture(scope="function")
def sample_df() -> DF:
    data = [{'a': 1, 'b': 2, 'c': 3},
            {'a': 4, 'b': 5, 'c': 6},
            {'a': 7, 'b': 8, 'c': 9}]
    return DF(data=data)


@pytest.fixture(scope="function")
def sample_df_with_row_names() -> DF:
    data = [{'a': 1, 'b': 2, 'c': 3},
            {'a': 4, 'b': 5, 'c': 6},
            {'a': 7, 'b': 8, 'c': 9}]
    df = DF(data=data)
    df.df = df.df.rename({0: 'd', 1: 'e', 2: 'f'}, axis="index")
    return df


class TestDF:

    def test_select_by_index(self, sample_df: DF):
        result = sample_df.select(row=0)
        assert result.shape == (3,)

        result = sample_df.select(column=0)
        assert result.shape == (3,)

        result = sample_df.select(row=[0, 1])
        assert result.shape == (2, 3)

        result = sample_df.select(column=[0, 1])
        assert result.shape == (3, 2)

        result = sample_df.select(row=0, column=0)
        assert result == 1

        result = sample_df.select(row=[0, 1], column=[0, 1])
        assert result.shape == (2, 2)

    def test_select_by_name(self, sample_df_with_row_names: DF):
        result = sample_df_with_row_names.select(row='d')
        assert result.shape == (3,)

        result = sample_df_with_row_names.select(column='a')
        assert result.shape == (3,)

        result = sample_df_with_row_names.select(row=['d', 'e'])
        assert result.shape == (2, 3)

        result = sample_df_with_row_names.select(column=['a', 'b'])
        assert result.shape == (3, 2)

        result = sample_df_with_row_names.select(row='d', column='a')
        assert result == 1

        result = sample_df_with_row_names.select(row=['d', 'e'], column=['a', 'b'])
        assert result.shape == (2, 2)

    # def test_update_row_in_place_by_index(self, sample_df: DF):
    #     row = 1
    #     new_row = Series([7, 8, 9])
    #     sample_df.update(row=row, to=new_row, in_place=True)
    #
    #     assert new_row.equals(sample_df.select(row=row))

    def test_update_column_in_place_by_index(self, sample_df: DF):
        column_index = 1
        old_column = copy(sample_df.select(column=column_index))
        new_column = Series([7, 8, 9])
        sample_df.update(column=column_index, to=new_column, in_place=True)

        assert new_column.equals(sample_df.select(column=column_index))

        sample_df.undo()

        assert old_column.equals(sample_df.select(column=column_index))

    # def test_update_row_and_return_by_index(self, sample_df: DF):
    #     new_column = Series([7, 8, 9])
    #     column_index = 1
    #
    #     result = sample_df.update(column=column_index, to=new_column, in_place=False).select(column=column_index)
    #
    #     assert new_column.equals(result)

    def test_update_column_and_return_by_index(self, sample_df):
        pass

    def test_update_row_in_place_by_name(self, sample_df):
        pass

    def test_update_column_in_place_by_name(self, sample_df: DF):
        column_name = "a"
        old_column = copy(sample_df.select(column=column_name))
        new_column = Series([7, 8, 9])
        sample_df.update(column=column_name, to=new_column, in_place=True)

        assert new_column.equals(sample_df.select(column=column_name))

        sample_df.undo()

        assert old_column.equals(sample_df.select(column=column_name))

    def test_update_row_and_return_by_name(self, sample_df):
        pass

    def test_update_column_and_return_by_name(self, sample_df):
        pass

    def test_sum_product(self, sample_df):
        actual_result_with_names = sample_df.sum_product('a', 'b')
        actual_result_with_indexes = sample_df.sum_product(0, 1)
        actual_result_with_both = sample_df.sum_product(0, 'b')
        expected_result = (1 * 2) + (4 * 5) + (7 * 8)
        assert expected_result == actual_result_with_names == actual_result_with_indexes == actual_result_with_both
