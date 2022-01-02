from collections import deque
from copy import copy

import pytest
from pandas import Series, DataFrame

from pandaSuit.df import DF, RandomDF, EmptyDF
from pandaSuit.plot.bar import BarPlot
from pandaSuit.plot.histogram import Histogram
from pandaSuit.plot.line import LinePlot
from pandaSuit.plot.pie import PiePlot
from pandaSuit.plot.scatter import ScatterPlot
from pandaSuit.stats.linear import LinearModel
from pandaSuit.stats.logistic import LogisticModel


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
    df.rename({0: 'd', 1: 'e', 2: 'f'}, axis="index", inplace=True)
    return df


@pytest.fixture(scope="module")
def static_df_with_row_names() -> DF:
    data = [{'a': 1, 'b': 2, 'c': 3},
            {'a': 4, 'b': 5, 'c': 6},
            {'a': 7, 'b': 8, 'c': 9}]
    df = DF(data=data)
    df.rename({0: 'd', 1: 'e', 2: 'f'}, axis="index", inplace=True)
    return df


class TestDF:

    def test_init_from_csv(self):
        df = DF(csv="C:\\Users\\antho\\projects\\pandaSuit\\src\\test\\resources\\test.csv")
        assert isinstance(df, DF)

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

    def test_random_row(self, sample_df):
        row = sample_df.random_row()
        comparisons = [row.equals(sample_df.random_row()) for _ in range(19)]

        # in theory, this unit test fails with probability 2.87e-10 --> (1/3)**20
        assert not all(comparisons)

    def test_regress(self):
        rdf = RandomDF(rows=1000, columns=2)

        linear = rdf.regress(y=1, x=0)
        assert isinstance(linear, LinearModel)

        rdf['binary_col'] = rdf.select(column=1) > 0.5
        logit = rdf.regress(y="binary_col", x=0, logit=True)
        assert isinstance(logit, LogisticModel)

    def test_line_plot(self, sample_df):
        single_lp = sample_df.line_plot('a')
        assert isinstance(single_lp, LinePlot)

        multi_lp = sample_df.line_plot('a', 'b')
        assert isinstance(multi_lp, LinePlot)

    def test_bar_plot(self, sample_df):
        single_bp = sample_df.bar_plot('a')
        assert isinstance(single_bp, BarPlot)

        multi_bp = sample_df.bar_plot('a', 'b')
        assert isinstance(multi_bp, BarPlot)

        all_bp = sample_df.bar_plot()
        assert isinstance(all_bp, BarPlot)

    def test_pie_plot(self, sample_df):
        single_pp = sample_df.pie_plot('a')
        assert isinstance(single_pp, PiePlot)

        multi_pp = sample_df.pie_plot('a', 'b')
        assert isinstance(multi_pp, PiePlot)

        all_pp = sample_df.pie_plot()
        assert isinstance(all_pp, PiePlot)

    def test_scatter_plot(self, sample_df):
        single_sp = sample_df.scatter_plot('a')
        assert isinstance(single_sp, ScatterPlot)

        multi_sp = sample_df.scatter_plot('a', 'b')
        assert isinstance(multi_sp, ScatterPlot)

    def test_histogram(self, sample_df):
        hist = sample_df.histogram('c')
        assert isinstance(hist, Histogram)

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

    def test_update_row_and_column(self, sample_df, sample_df_with_row_names):
        new_value = 1000

        # update single value by row and column index
        column_index, row_index = 0, 1
        old_value = sample_df.select(row=row_index, column=column_index, pandas_return_type=True)
        sample_df.update(row=row_index, column=column_index, to=new_value, in_place=True)
        assert new_value == sample_df.select(row=row_index, column=column_index, pandas_return_type=True)
        sample_df.undo()
        assert old_value == sample_df.select(row=row_index, column=column_index, pandas_return_type=True)

        # update single value by row and column name
        column_name, row_name = "a", "e"
        old_value = sample_df_with_row_names.select(row=row_name, column=column_name, pandas_return_type=True)
        sample_df_with_row_names.update(row=row_name, column=column_name, to=new_value, in_place=True)
        assert new_value == sample_df_with_row_names.select(row=row_name, column=column_name, pandas_return_type=True)
        sample_df_with_row_names.undo()
        assert old_value == sample_df_with_row_names.select(row=row_name, column=column_name, pandas_return_type=True)

        # update single value by row index and column name
        old_value = sample_df.select(row=row_index, column=column_name, pandas_return_type=True)
        sample_df.update(row=row_index, column=column_name, to=new_value, in_place=True)
        assert new_value == sample_df.select(row=row_index, column=column_name, pandas_return_type=True)
        sample_df.undo()
        assert old_value == sample_df.select(row=row_index, column=column_name, pandas_return_type=True)

        # update single value by row name and column index
        old_value = sample_df_with_row_names.select(row=row_name, column=column_index, pandas_return_type=True)
        sample_df_with_row_names.update(row=row_name, column=column_index, to=new_value, in_place=True)
        assert new_value == sample_df_with_row_names.select(row=row_name, column=column_index, pandas_return_type=True)
        sample_df_with_row_names.undo()
        assert old_value == sample_df_with_row_names.select(row=row_name, column=column_index, pandas_return_type=True)

    def test_update_column_in_place_by_name(self, sample_df: DF):
        column_name = "a"
        old_column = copy(sample_df.select(column=column_name, pandas_return_type=True))
        new_column = Series([7, 8, 9])

        sample_df.update(column=column_name, to=new_column, in_place=True)
        assert new_column.equals(sample_df.select(column=column_name, pandas_return_type=True))

        sample_df.undo()
        assert old_column.equals(sample_df.select(column=column_name, pandas_return_type=True))

    def test_update_and_return(self, sample_df: DF):
        column_name = "a"
        new_column = Series([7, 8, 9])
        assert new_column.equals(sample_df.update(column=column_name, to=new_column, in_place=False).select(column=column_name, pandas_return_type=True))

    def test_update_row_and_return_by_name(self, sample_df):
        pass

    def test_update_column_and_return_by_name(self, sample_df):
        pass

    def test_update_with_exception(self, sample_df):
        with pytest.raises(Exception):
            sample_df.update()
        assert len(sample_df._unwind) == 0

    def test_append_row(self, sample_df, static_df):
        new_row = Series({'a': 10, 'b': 11, 'c': 12})

        # todo: pull this out into its own test of .undo() method
        # keyword arg for testing undo() logic
        sample_df.append(row=new_row)
        assert new_row.equals(sample_df.select(row=3, pandas_return_type=True))

        sample_df.undo()
        assert static_df.equals(sample_df)

        # positional arg for testing undo() logic
        sample_df.append(new_row)
        assert new_row.equals(sample_df.select(row=3, pandas_return_type=True))

        sample_df.undo()
        assert static_df.equals(sample_df)

        result = sample_df.append(row=new_row, in_place=False)
        assert new_row.equals(result.select(row=3, pandas_return_type=True))

    def test_append_column(self, sample_df, static_df):
        new_column = Series([10, 11, 12])

        sample_df.append(column=new_column)
        assert new_column.equals(sample_df.select(column=3, pandas_return_type=True))

        sample_df.undo()
        assert static_df.equals(sample_df)

        result = sample_df.append(column=new_column, in_place=False)
        assert new_column.equals(result.select(column=3, pandas_return_type=True))

    def test_append_with_exception(self, sample_df):
        with pytest.raises(Exception):
            sample_df.append()
        assert len(sample_df._unwind) == 0

    def test_insert(self, sample_df, static_df):
        index = 2
        row = Series({'a': 10, 'b': 11, 'c': 12})
        rows = DataFrame({'a': [10, 13], 'b': [11, 14], 'c': [12, 15]})
        column = Series([10, 11, 12])
        columns = DataFrame([[10, 11, 12], [13, 14, 15], [16, 17, 18]])

        # insert single row
        sample_df.insert(index=index, row=row)
        assert row.equals(sample_df.select(row=index, pandas_return_type=True))
        sample_df.undo()
        assert sample_df.equals(static_df)

        # insert single column
        sample_df.insert(index=index, column=column)
        assert column.equals(sample_df.select(column=index, pandas_return_type=True))
        sample_df.undo()
        assert sample_df.equals(static_df)

        # insert multiple continuous rows (1 index passed)
        sample_df.insert(index=index, row=rows)
        assert rows.rename({0: 3, 1: 4}, inplace=False, axis=0).equals(sample_df.slice(from_row=index, to_row=index+rows.shape[0], pandas_return_type=True))
        sample_df.undo()
        assert sample_df.equals(static_df)

        # insert multiple continuous rows (multiple indexes passed)
        sample_df.insert(index=[index, index+1], row=rows)
        assert rows.rename({0: 3, 1: 4}, inplace=False, axis=0).equals(sample_df.slice(from_row=index, to_row=index+rows.shape[0], pandas_return_type=True))
        sample_df.undo()
        assert sample_df.equals(static_df)

        # insert discontinuous rows
        sample_df.insert(index=[0, 2], row=rows)
        assert rows.rename({0: 3, 1: 4}, inplace=False, axis=0).equals(sample_df.select(row=[0, 2], pandas_return_type=True))
        sample_df.undo()
        assert sample_df.equals(static_df)

        # insert multiple continuous columns (1 index passed)
        sample_df.insert(index=index, column=columns)
        assert columns.rename({0: 3, 1: 4, 2: 5}, inplace=False, axis=1).equals(sample_df.slice(from_column=index, to_column=index+columns.shape[1], pandas_return_type=True))
        sample_df.undo()
        assert sample_df.equals(static_df)

        # insert multiple continuous columns (multiple indexes passed)
        sample_df.insert(index=[index, index+1, index+2], column=columns)
        assert columns.rename({0: 3, 1: 4, 2: 5}, inplace=False, axis=1).equals(sample_df.slice(from_column=index, to_column=index+columns.shape[1], pandas_return_type=True))
        sample_df.undo()
        assert sample_df.equals(static_df)

        # insert discontinuous columns
        sample_df.insert(index=[0, 2, 4], column=columns)
        assert columns.equals(sample_df.select(column=[0, 2, 4], pandas_return_type=True))
        sample_df.undo()
        assert sample_df.equals(static_df)

        # not in place
        assert columns.equals(sample_df.insert(index=[0, 2, 4], column=columns, in_place=False).select(column=[0, 2, 4], pandas_return_type=True))

    def test_insert_with_exception(self, sample_df):
        with pytest.raises(Exception):
            sample_df.insert(index=1)
        assert len(sample_df._unwind) == 0

        with pytest.raises(Exception):
            sample_df.insert(index=[1, 3], row=sample_df)
        assert len(sample_df._unwind) == 0

    def test_remove(self, sample_df_with_row_names, static_df_with_row_names):
        # remove row by single index
        sample_df_with_row_names.remove(row=1)
        assert sample_df_with_row_names.shape == (2, 3)
        sample_df_with_row_names.undo()
        assert sample_df_with_row_names.equals(static_df_with_row_names)

        # remove row by single name
        sample_df_with_row_names.remove(row='e')
        assert sample_df_with_row_names.shape == (2, 3)
        sample_df_with_row_names.undo()
        assert sample_df_with_row_names.equals(static_df_with_row_names)

        # remove column by single index
        sample_df_with_row_names.remove(column=1)
        assert sample_df_with_row_names.shape == (3, 2)
        sample_df_with_row_names.undo()
        assert sample_df_with_row_names.equals(static_df_with_row_names)

        # remove column by single name
        sample_df_with_row_names.remove(column='b')
        assert sample_df_with_row_names.shape == (3, 2)
        sample_df_with_row_names.undo()
        assert sample_df_with_row_names.equals(static_df_with_row_names)

        # remove rows by continuous index
        sample_df_with_row_names.remove(row=[0, 1])
        assert sample_df_with_row_names.shape == (1, 3)
        sample_df_with_row_names.undo()
        assert sample_df_with_row_names.equals(static_df_with_row_names)

        # remove rows by discontinuous index
        sample_df_with_row_names.remove(row=[0, 2])
        assert sample_df_with_row_names.shape == (1, 3)
        sample_df_with_row_names.undo()
        assert sample_df_with_row_names.equals(static_df_with_row_names)

        # remove rows by continuous names
        sample_df_with_row_names.remove(row=['d', 'e'])
        assert sample_df_with_row_names.shape == (1, 3)
        sample_df_with_row_names.undo()
        assert sample_df_with_row_names.equals(static_df_with_row_names)

        # remove rows by discontinuous names
        sample_df_with_row_names.remove(row=['d', 'f'])
        assert sample_df_with_row_names.shape == (1, 3)
        sample_df_with_row_names.undo()
        assert sample_df_with_row_names.equals(static_df_with_row_names)

        # remove columns by continuous index
        sample_df_with_row_names.remove(column=[0, 1])
        assert sample_df_with_row_names.shape == (3, 1)
        sample_df_with_row_names.undo()
        assert sample_df_with_row_names.equals(static_df_with_row_names)

        # remove columns by discontinuous index
        sample_df_with_row_names.remove(column=[0, 2])
        assert sample_df_with_row_names.shape == (3, 1)
        sample_df_with_row_names.undo()
        assert sample_df_with_row_names.equals(static_df_with_row_names)

        # remove columns by continuous names
        sample_df_with_row_names.remove(column=['a', 'b'])
        assert sample_df_with_row_names.shape == (3, 1)
        sample_df_with_row_names.undo()
        assert sample_df_with_row_names.equals(static_df_with_row_names)

        # remove columns by discontinuous names
        sample_df_with_row_names.remove(column=['a', 'c'])
        assert sample_df_with_row_names.shape == (3, 1)
        sample_df_with_row_names.undo()
        assert sample_df_with_row_names.equals(static_df_with_row_names)

        # not in place
        assert sample_df_with_row_names.remove(column=['a', 'c'], in_place=False).shape == (3, 1)

    def test_remove_exception(self, sample_df):
        with pytest.raises(Exception):
            sample_df.remove()
        with pytest.raises(Exception):
            sample_df.remove(row=0, column=0)

    def test_reset(self, sample_df, static_df):
        sample_df.update(row=2, column='a', to=100)
        sample_df.insert(index=1, column=Series(name='d', data=[222, 333, 444]))
        sample_df *= 2.5
        df_after_undoing_reset = copy(sample_df)

        sample_df.reset()
        assert sample_df.equals(static_df)

        sample_df.undo()
        assert df_after_undoing_reset.equals(sample_df)

        sample_df.update(row=2, column='a', to=100)
        sample_df.insert(index=1, column=Series(name='d', data=[222, 333, 444]))
        sample_df *= 2.5
        assert not df_after_undoing_reset.equals(sample_df)

        assert static_df.equals(sample_df.reset(in_place=False))

    def test_undo_exception(self, sample_df):
        with pytest.raises(Exception):
            sample_df.undo()

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
        new_values = ['g', 'h', 'i']
        sample_df_with_row_names.__setattr__("index", new_values)
        assert all([new_value in sample_df_with_row_names.index for new_value in new_values])

    def test_getattribute(self, sample_df):
        assert sample_df.__getattribute__("_unwind") == deque()

    def test_getattribute_fallback(self, sample_df):
        new_value = "value"
        sample_df.some_new_field = new_value
        assert sample_df.__getattribute__("some_new_field") == new_value

    def test_getattribute_fallout(self, sample_df):
        with pytest.raises(AttributeError):
            sample_df.__getattribute__("some_non_existent_field")

    def test_is_empty(self):
        df = DF()
        assert df.is_empty

    def test_iadd(self, sample_df, static_df):
        sample_df += 5
        gt_df = DF(sample_df > static_df)
        assert all([field for row in gt_df.rows for field in row])

        sample_df.undo()
        assert static_df.equals(sample_df)

    def test_isub(self, sample_df, static_df):
        sample_df -= 1
        lt_df = DF(sample_df < static_df)
        assert all([field for row in lt_df.rows for field in row])

        sample_df.undo()
        assert static_df.equals(sample_df)

    def test_imul(self, sample_df, static_df):
        sample_df *= 2
        gt_df = DF(sample_df > static_df)
        assert all([field for row in gt_df.rows for field in row])

        sample_df.undo()
        assert sample_df.equals(static_df.astype(float))

    def test_itruediv(self, sample_df, static_df):
        sample_df /= 2
        lt_df = DF(sample_df < static_df)
        assert all([field for row in lt_df.rows for field in row])

        sample_df.undo()
        assert sample_df.equals(static_df.astype(float))

    def test_random_df(self):
        rdf = RandomDF(rows=10, columns=10)
        assert rdf.row_count == 10
        assert rdf.column_count == 10

        string_rdf = RandomDF(rows=10, columns=3, data_type=str)
        assert all([isinstance(field, str) for row in string_rdf.rows for field in row])

        none_rdf = RandomDF(rows=10, columns=3, data_type=None)
        assert all([field is None for row in none_rdf.rows for field in row])

    def test_regenerate_random_df(self):
        rdf = RandomDF()
        static_rdf = RandomDF(rows=rdf.row_count, columns=rdf.column_count)
        rdf.remove(row=0)
        rdf.remove(column=0)
        assert rdf.row_count != static_rdf.row_count
        assert rdf.column_count != static_rdf.column_count

        rdf.regenerate()
        assert rdf.row_count == static_rdf.row_count
        assert rdf.column_count == static_rdf.column_count

    def test_random_df_with_unsupported_distribution(self):
        with pytest.raises(Exception):
            RandomDF(distribution='some random distribution')

    def test_random_df_with_unsupported_type(self):
        with pytest.raises(Exception):
            RandomDF(data_type=list)

    def test_empty_df(self):
        edf = EmptyDF(rows=10, columns=3)
        assert all([field is None for row in edf.rows for field in row])
        assert all([isinstance(header, str) for header in edf.column_names])

        edf_with_column_headers = EmptyDF(rows=10, columns=3, column_headers=False)
        assert all([not isinstance(header, str) for header in edf_with_column_headers.column_names])
