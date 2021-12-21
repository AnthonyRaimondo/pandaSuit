from collections import deque
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


@pytest.fixture(scope="module")
def static_df() -> DF:
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
    df._df.rename({0: 'd', 1: 'e', 2: 'f'}, axis="index", inplace=True)
    return df


class TestDF:

    def test_select_by_index(self, sample_df: DF):
        result = sample_df.select(row=0)
        assert result.shape == (3, 1)

        result = sample_df.select(column=0)
        assert result.shape == (3, 1)

        result = sample_df.select(row=[0, 1])
        assert result.shape == (2, 3)

        result = sample_df.select(column=[0, 1])
        assert result.shape == (3, 2)

        result = sample_df.select(row=0, column=0)
        assert result == 1

        result = sample_df.select(row=[0, 1], column=[0, 1])
        assert result.shape == (2, 2)

    def test_select_by_index__pandas_return(self, sample_df: DF):
        result = sample_df.select(row=0, pandas_return_type=True)
        assert result.shape == (3,)

        result = sample_df.select(column=0, pandas_return_type=True)
        assert result.shape == (3,)

        result = sample_df.select(row=[0, 1], pandas_return_type=True)
        assert result.shape == (2, 3)

        result = sample_df.select(column=[0, 1], pandas_return_type=True)
        assert result.shape == (3, 2)

        result = sample_df.select(row=0, column=0, pandas_return_type=True)
        assert result == 1

        result = sample_df.select(row=[0, 1], column=[0, 1], pandas_return_type=True)
        assert result.shape == (2, 2)

    def test_select_by_name(self, sample_df_with_row_names: DF):
        result = sample_df_with_row_names.select(row='d')
        assert result.shape == (3, 1)

        result = sample_df_with_row_names.select(column='a')
        assert result.shape == (3, 1)

        result = sample_df_with_row_names.select(row=['d', 'e'])
        assert result.shape == (2, 3)

        result = sample_df_with_row_names.select(column=['a', 'b'])
        assert result.shape == (3, 2)

        result = sample_df_with_row_names.select(row='d', column='a')
        assert result == 1

        result = sample_df_with_row_names.select(row=['d', 'e'], column=['a', 'b'])
        assert result.shape == (2, 2)

    def test_select_by_name__pandas_return(self, sample_df_with_row_names: DF):
        result = sample_df_with_row_names.select(row='d', pandas_return_type=True)
        assert result.shape == (3,)

        result = sample_df_with_row_names.select(column='a', pandas_return_type=True)
        assert result.shape == (3,)

        result = sample_df_with_row_names.select(row=['d', 'e'], pandas_return_type=True)
        assert result.shape == (2, 3)

        result = sample_df_with_row_names.select(column=['a', 'b'], pandas_return_type=True)
        assert result.shape == (3, 2)

        result = sample_df_with_row_names.select(row='d', column='a', pandas_return_type=True)
        assert result == 1

        result = sample_df_with_row_names.select(row=['d', 'e'], column=['a', 'b'], pandas_return_type=True)
        assert result.shape == (2, 2)

    def test_slice(self, sample_df, sample_df_with_row_names):
        # from_row only (int)
        assert sample_df.slice(from_row=1).shape == (2, 3)

        # to_row only (int)
        assert sample_df.slice(to_row=1).shape == (1, 3)

        # from_column only (int)
        assert sample_df.slice(from_column=1).shape == (3, 2)

        # to_column only (int)
        assert sample_df.slice(to_column=1).shape == (3, 1)

        # from_row only (str)
        assert sample_df_with_row_names.slice(from_row='e').shape == (2, 3)

        # to_row only (str)
        assert sample_df_with_row_names.slice(to_row='e').shape == (1, 3)

        # from_column only (str)
        assert sample_df_with_row_names.slice(from_column='b').shape == (3, 2)

        # to_column only (str)
        assert sample_df_with_row_names.slice(to_column='b').shape == (3, 1)

        # row slice (int, int)
        assert sample_df.slice(from_row=1, to_row=2).shape == (1, 3)

        # column slice (int, int)
        assert sample_df.slice(from_column=1, to_column=2).shape == (3, 1)

        # row slice (str, str)
        assert sample_df_with_row_names.slice(from_row='e', to_row='f').shape == (1, 3)

        # column slice (str, str)
        assert sample_df_with_row_names.slice(from_column='b', to_column='c').shape == (3, 1)

        # row slice (str, int)
        assert sample_df_with_row_names.slice(from_row='e', to_row=2).shape == (1, 3)

        # column slice (str, int)
        assert sample_df.slice(from_column='b', to_column=2).shape == (3, 1)

        # row slice (int, str)
        assert sample_df_with_row_names.slice(from_row=1, to_row='f').shape == (1, 3)

        # column slice (int, str)
        assert sample_df_with_row_names.slice(from_column=1, to_column='c').shape == (3, 1)

        # row and column slice (int)
        assert sample_df.slice(from_row=1, to_row=2, from_column=1, to_column=2).shape == (1, 1)

        # row and column slice (str)
        assert sample_df_with_row_names.slice(from_row='e', to_row='f', from_column='b', to_column='c').shape == (1, 1)

        # row and column slice (int, str)
        assert sample_df_with_row_names.slice(from_row=1, to_row='f', from_column='b', to_column=2).shape == (1, 1)

    def test_slice__pandas_return(self, sample_df, sample_df_with_row_names):
        # from_row only (int)
        assert sample_df.slice(from_row=1, pandas_return_type=True).shape == (2, 3)

        # to_row only (int)
        assert sample_df.slice(to_row=1, pandas_return_type=True).shape == (1, 3)

        # from_column only (int)
        assert sample_df.slice(from_column=1, pandas_return_type=True).shape == (3, 2)

        # to_column only (int)
        assert sample_df.slice(to_column=1, pandas_return_type=True).shape == (3, 1)

        # from_row only (str)
        assert sample_df_with_row_names.slice(from_row='e', pandas_return_type=True).shape == (2, 3)

        # to_row only (str)
        assert sample_df_with_row_names.slice(to_row='e', pandas_return_type=True).shape == (1, 3)

        # from_column only (str)
        assert sample_df_with_row_names.slice(from_column='b', pandas_return_type=True).shape == (3, 2)

        # to_column only (str)
        assert sample_df_with_row_names.slice(to_column='b', pandas_return_type=True).shape == (3, 1)

        # row slice (int, int)
        assert sample_df.slice(from_row=1, to_row=2, pandas_return_type=True).shape == (1, 3)

        # column slice (int, int)
        assert sample_df.slice(from_column=1, to_column=2, pandas_return_type=True).shape == (3, 1)

        # row slice (str, str)
        assert sample_df_with_row_names.slice(from_row='e', to_row='f', pandas_return_type=True).shape == (1, 3)

        # column slice (str, str)
        assert sample_df_with_row_names.slice(from_column='b', to_column='c', pandas_return_type=True).shape == (3, 1)

        # row slice (str, int)
        assert sample_df_with_row_names.slice(from_row='e', to_row=2, pandas_return_type=True).shape == (1, 3)

        # column slice (str, int)
        assert sample_df.slice(from_column='b', to_column=2, pandas_return_type=True).shape == (3, 1)

        # row slice (int, str)
        assert sample_df_with_row_names.slice(from_row=1, to_row='f', pandas_return_type=True).shape == (1, 3)

        # column slice (int, str)
        assert sample_df_with_row_names.slice(from_column=1, to_column='c', pandas_return_type=True).shape == (3, 1)

        # row and column slice (int)
        assert sample_df.slice(from_row=1, to_row=2, from_column=1, to_column=2, pandas_return_type=True).shape == (1, 1)

        # row and column slice (str)
        assert sample_df_with_row_names.slice(from_row='e', to_row='f', from_column='b', to_column='c', pandas_return_type=True).shape == (1, 1)

        # row and column slice (int, str)
        assert sample_df_with_row_names.slice(from_row=1, to_row='f', from_column='b', to_column=2, pandas_return_type=True).shape == (1, 1)

    # def test_update_row_in_place_by_index(self, sample_df: DF):
    #     row = 1
    #     new_row = Series([7, 8, 9])
    #     sample_df.update(row=row, to=new_row, in_place=True)
    #
    #     assert new_row.equals(sample_df.select(row=row))

    def test_update_column_in_place_by_index(self, sample_df: DF):
        column_index = 1
        old_column = copy(sample_df.select(column=column_index, pandas_return_type=True))
        new_column = Series([7, 8, 9])

        sample_df.update(column=column_index, to=new_column, in_place=True)
        assert new_column.equals(sample_df.select(column=column_index, pandas_return_type=True))

        sample_df.undo()
        assert old_column.equals(sample_df.select(column=column_index, pandas_return_type=True))

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
        old_column = copy(sample_df.select(column=column_name, pandas_return_type=True))
        new_column = Series([7, 8, 9])

        sample_df.update(column=column_name, to=new_column, in_place=True)
        assert new_column.equals(sample_df.select(column=column_name, pandas_return_type=True))

        sample_df.undo()
        assert old_column.equals(sample_df.select(column=column_name, pandas_return_type=True))

    def test_update_row_and_return_by_name(self, sample_df):
        pass

    def test_update_column_and_return_by_name(self, sample_df):
        pass

    def test_append_row(self, sample_df, static_df):
        new_row = Series({'a': 10, 'b': 11, 'c': 12})

        # keyword arg for testing undo() logic
        sample_df.append(row=new_row)
        assert new_row.equals(sample_df.select(row=3, pandas_return_type=True))

        sample_df.undo()
        assert static_df.dataframe.equals(sample_df.dataframe)

        # positional arg for testing undo() logic
        sample_df.append(new_row)
        assert new_row.equals(sample_df.select(row=3, pandas_return_type=True))

        sample_df.undo()
        assert static_df.dataframe.equals(sample_df.dataframe)

        result = sample_df.append(row=new_row, in_place=False)
        assert new_row.equals(result.select(row=3, pandas_return_type=True))

    def test_append_column(self, sample_df, static_df):
        new_column = Series([10, 11, 12])

        sample_df.append(column=new_column)
        assert new_column.equals(sample_df.select(column=3, pandas_return_type=True))

        sample_df.undo()
        assert static_df.dataframe.equals(sample_df.dataframe)

        result = sample_df.append(column=new_column, in_place=False)
        assert new_column.equals(result.select(column=3, pandas_return_type=True))

    def test_append_with_exception(self, sample_df):
        with pytest.raises(Exception):
            sample_df.append()

    def test_sum_product(self, sample_df):
        actual_result_with_names = sample_df.sum_product('a', 'b')
        actual_result_with_indexes = sample_df.sum_product(0, 1)
        actual_result_with_both = sample_df.sum_product(0, 'b')
        expected_result = (1 * 2) + (4 * 5) + (7 * 8)
        assert expected_result == actual_result_with_names == actual_result_with_indexes == actual_result_with_both

    def test_setattr(self, sample_df):
        new_value = [1, 2, 3]
        sample_df.__setattr__("_unwind", new_value)
        assert sample_df._unwind == new_value

    def test_setattr_fallback(self, sample_df_with_row_names):
        new_value = ['g', 'h', 'i']
        sample_df_with_row_names.__setattr__("index", new_value)
        assert sample_df_with_row_names.index == new_value

    def test_getattribute(self, sample_df):
        assert sample_df.__getattribute__("_unwind") == deque()

    def test_getattribute_fallback(self, sample_df):
        new_value = "value"
        sample_df.dataframe.some_new_field = new_value
        assert sample_df.__getattribute__("some_new_field") == new_value

    def test_getattribute_fallout(self, sample_df):
        with pytest.raises(AttributeError):
            sample_df.__getattribute__("some_non_existent_field")
