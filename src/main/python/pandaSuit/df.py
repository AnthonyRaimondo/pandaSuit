from __future__ import annotations

from collections import deque
from copy import copy

import pandas
from numpy import random as np_random

from pandaSuit.common.constant.date_constants import DATE_GROUPINGS
from pandaSuit.common.constant.df import ALPHABET, DISTRIBUTIONS
from pandaSuit.common.decorators import reversible
from pandaSuit.common.unwind import Unwind
from pandaSuit.common.util.list_operations import index_dictionary, create_index_list, find_indexes
from pandaSuit.plot.bar import BarPlot
from pandaSuit.plot.histogram import Histogram
from pandaSuit.plot.line import LinePlot
from pandaSuit.plot.pie import PiePlot
from pandaSuit.plot.scatter import ScatterPlot
from pandaSuit.stats.linear import LinearModel
from pandaSuit.stats.logistic import LogisticModel


class DF:
    def __init__(self, data=None):
        self.data = data
        if data is not None:
            self._df = pandas.DataFrame(data)
        else:
            self._df = pandas.DataFrame()
        self._unwind = deque()

    def select(self,
               row: list or int or str = None,
               column: list or int or str = None,
               pandas_return_type: bool = False) -> pandas.DataFrame or pandas.Series or DF:
        if row is None:
            if self._names_supplied(column):
                result = self._df[column]
            else:
                result = self._df.iloc[:, column]
        else:
            if column is None:
                if self._names_supplied(row):
                    result = self._df.loc[row]
                else:
                    result = self._df.iloc[row]
            else:
                if self._names_supplied(row) and self._names_supplied(column):
                    result = self._df.loc[row, column]
                else:
                    if self._names_supplied(row):
                        result = self._df.loc[row].iloc[:, column]
                    else:
                        result = self._df.iloc[row, column]
        return result if pandas_return_type else DF(result)

    def slice(self,
              from_row: int or str = 0,
              to_row: int or str = -1,
              from_column: int or str = 0,
              to_column: int or str = -1,
              pandas_return_type: bool = False) -> pandas.DataFrame or pandas.Series or DF:

        if isinstance(from_row, str):
            from_row = find_indexes(self.row_names, from_row)[0]
        if isinstance(to_row, str):
            to_row = find_indexes(self.row_names, to_row)[0]
        if isinstance(from_column, str):
            from_column = find_indexes(self.column_names, from_column)[0]
        if isinstance(to_column, str):
            to_column = find_indexes(self.column_names, to_column)[0]

        if to_row < 0:
            to_row += self.row_count + 1
        if to_column < 0:
            to_column += self.column_count + 1

        result = self._df.iloc[from_row:to_row, from_column:to_column]

        return result if pandas_return_type else DF(result)

    def where(self, column_name: str, some_value: object, pandas_return_type: bool = True) -> pandas.DataFrame:
        if isinstance(some_value, str):
            result = self._df[self._df[column_name].str.contains(some_value, na=False)]
        else:
            result = self._df.loc[self._df[column_name] == some_value]
        return result if pandas_return_type else DF(result)

    def where_not(self, column_name: str, some_value: object, pandas_return_type: bool = True) -> pandas.DataFrame:
        if isinstance(some_value, str):
            result = self._df[~self._df[column_name].isin([some_value])]
        else:
            result = self._df.loc[self._df[column_name] != some_value]
        return result if pandas_return_type else DF(result)

    def random_row(self) -> pandas.DataFrame:
        return self._df.iloc[np_random.randint(0, self._df.shape[0] - 1)]

    def regress(self, y: str or int, x: list or str or int, logit: bool = False) -> LinearModel or LogisticModel:
        if logit:
            return LogisticModel(dependent=self.select(column=y), independent=self.select(column=x))
        else:
            return LinearModel(dependent=self.select(column=y), independent=self.select(column=x))

    # Plotting
    def line_plot(self, *y: int or str, x: int or str or list = None) -> LinePlot:
        """
        Creates a Line Plot with y as response variable(s) and x as explanatory variable.
        :param y: Column name(s)/index(es) of response variable(s)
        :param x: Column name/index of explanatory variable
        :return: LinePlot with y as response variable(s) and x as explanatory variable.
        """
        return LinePlot(x=self.select(column=x) if x is not None else self.row_names,
                        y=[pandas.Series(column[1]) for column in self.select(column=list(y)).iteritems()],
                        y_label=y[0] if len(y) == 1 and isinstance(y[0], str) else None,
                        x_label=x if isinstance(x, str) else None)

    def bar_plot(self, *bars: int or str, x: int or str or list = None) -> BarPlot:
        """
        Creates a Bar Plot with y as response variable(s) and x as explanatory variable.
        :param bars: Column name(s)/index(es) of response variable(s)
        :param x: Column name/index of explanatory variable
        :return: BarPlot with y as response variable(s) and x as explanatory variable.
        """
        return BarPlot(x=self.select(column=x) if x is not None else self.row_names if len(bars) > 0 else self.column_names,
                       y=[pandas.Series(column[1]) for column in self.select(column=list(bars)).iteritems()] if len(bars) > 0 else [self._df.sum()],
                       y_label=bars[0] if len(bars) == 1 and isinstance(bars[0], str) else None,
                       x_label=x if isinstance(x, str) else None)

    def pie_plot(self, *slices) -> PiePlot:
        """
        Creates a Pie Plot for the slice(s) specified.
        :param slices: Column name(s)/index(es) to use for PiePlot sections
        :return: PiePlot for the slice(s) specified.
        """
        if len(slices) > 1:
            return PiePlot(self.select(column=list(slices)).sum().to_dict())
        elif len(slices) == 0:
            return PiePlot(self._df.sum().to_dict())
        else:
            return PiePlot(self.select(column=slices[0]).value_counts().to_dict())

    def scatter_plot(self, *y: int or str, x: int or str or list = None, best_fit_line: bool = False) -> ScatterPlot:
        """
        Creates a Scatter Plot with y as response variable(s) and x as explanatory variable.
        :param y: Column name(s)/index(es) of response variable(s)
        :param x: Column name/index of explanatory variable
        :param best_fit_line: flag indicating whether or not to include a best fit line in the ScatterPlot
        :return: ScatterPlot with y as response variable(s) and x as explanatory variable.
        """
        return ScatterPlot(x=self.select(column=x) if x is not None else self.row_names,
                           y=[pandas.Series(column[1]) for column in self.select(column=list(y)).iteritems()],
                           y_label=y[0] if len(y) == 1 and isinstance(y[0], str) else None,
                           x_label=x if isinstance(x, str) else None,
                           best_fit_line=best_fit_line)

    def histogram(self, y: int or str, bins: int = 10) -> Histogram:
        return Histogram(y=self.select(column=y), bins=bins)

    def where_null(self, column: str, pandas_return_type: bool = True) -> DF or pandas.DataFrame:
        result = self._df[self._df[column].isnull()]
        return result if pandas_return_type else DF(result)

    def where_not_null(self, column: str, pandas_return_type: bool = True) -> DF or pandas.DataFrame:
        result = self._df[self._df[column].notna()]
        return result if pandas_return_type else DF(result)

    def group_by(self, column: int or str = None, row: int or str = None, date_grouping: str = None) -> dict:
        """
        Returns a dictionary object that groups on a Row/Column, using the grouping values as keys, pointing to Table objects containing the Row(s)/Column(s) that contain the key value.
        :param column: Column to group on
        :param row: Row to group on
        :param date_grouping: type of date grouping (e.g. "day", "month", "year")
        :return: Dictionary containing values grouped by (keys) and items belonging to that grouping (values).
        """
        if date_grouping is None:
            return {name: self.select(column=indexes, pandas_return_type=False)
                    if row is not None else self.select(row=indexes, pandas_return_type=False)
                    for name, indexes in index_dictionary(
                    (self.select(row=row, pandas_return_type=True) if row is not None
                     else self.select(column=column, pandas_return_type=True)).values).items()}
        else:
            grouping = DATE_GROUPINGS.get(date_grouping)
            if grouping is None:
                raise Exception(f"Invalid date grouping type \"{date_grouping}\"")
            if column is None:
                raise Exception("Cannot group on a Row of dates")
            date_group_by_object = self._df.groupby(pandas.to_datetime(self.select(column=column)).dt.strftime(grouping))
            return {date_key: DF(date_group_by_object.get_group(date_key)) for date_key in list(date_group_by_object.groups.keys())}

    def sum_product(self, *columns: int or str) -> int or float:
        product_column = pandas.Series([1]*self.row_count)
        for column in columns:
            product_column *= self.select(column=column)
        return product_column.sum()

    @reversible
    def update(self, row: int or str = None, column: int or str = None, to: object = None, in_place: bool = True) -> DF or None:
        if in_place:
            if column is not None:
                if row is not None:
                    if isinstance(column, str):
                        self._df.loc[row, column] = to
                    else:
                        self._df.iloc[row, column] = to
                else:
                    if isinstance(column, str):
                        self._df.loc[create_index_list(self.row_count), column] = to
                    else:
                        self._df.iloc[create_index_list(self.row_count), column] = to
            elif row is not None:
                if isinstance(row, str):
                    pass
                else:
                    pass
            else:
                raise Exception("Please supply a row or column to update.")
        else:
            _df = copy(self)
            _df.update(row=row, column=column, to=to, in_place=True)
            return _df

    def append(self, row: pandas.Series = None, column: pandas.Series = None, in_place: bool = True) -> DF or None:
        if row is not None and column is None:
            if in_place:
                self._append_row(row, in_place)
            else:
                return self._append_row(row, in_place)
        elif row is None and column is not None:
            if in_place:
                self._append_column(column, in_place)
            else:
                return self._append_column(column, in_place)
        elif row is not None and column is not None:
            if len(row) > len(column):
                if in_place:
                    self._append_column(column, in_place)
                    self._append_row(row, in_place)
                else:
                    return DF(copy(self._df))._append_column(column, in_place)._append_row(row, in_place)
            else:
                if in_place:
                    self._append_row(row, in_place)
                    self._append_column(column, in_place)
                else:
                    return DF(copy(self._df))._append_row(row, in_place)._append_column(column, in_place)
        else:
            raise Exception("row or column parameter must be set")

    def undo(self) -> None:
        """
        Reverts the most recent change to the Table instance.
        """
        unwind_object: Unwind = self._unwind.pop()
        self.__getattribute__(unwind_object.function)(**unwind_object.args[0])

    def reset(self) -> None:
        self._df = DF(data=self.data)._df

    def _append_row(self, row: pandas.Series, in_place: bool) -> DF or None:
        if in_place:
            self._df = self._df.append(other=row, ignore_index=True)
        else:
            _df = copy(self._df)
            _df.append(other=row, ignore_index=True)
            return DF(_df)

    def _append_column(self, column: pandas.Series, in_place: bool) -> DF or None:
        if in_place:
            self._df.insert(loc=self.column_count, column=column.name, value=column, allow_duplicates=True)
        else:
            _df = copy(self._df)
            _df.insert(loc=self.column_count, column=column.name, value=column, allow_duplicates=True)
            return DF(_df)

    @staticmethod
    def _names_supplied(selector: int or str or list) -> bool:
        if isinstance(selector, list):
            return isinstance(selector[0], str)
        else:
            return isinstance(selector, str)

    @staticmethod
    def _create_str_column_names(columns: int) -> list:
        letters, headers = [letter for letter in ALPHABET], []
        for column_index in range(columns):
            cycles = column_index // len(letters)
            if cycles == 0:
                headers.append(letters[column_index])
            elif cycles <= len(letters):
                headers.append(letters[cycles - 1] + letters[column_index % len(letters)])
            else:
                headers.append(letters[(cycles // len(letters)) - 1] + letters[(cycles % len(letters)) - 1] + letters[
                    column_index % len(letters)])
        return headers

    @property
    def is_empty(self) -> bool:
        return self._df.empty

    @property
    def rows(self) -> list:
        return [pandas.Series(row[1]) for row in self._df.iterrows()]

    @property
    def row_names(self):
        return [row.name for row in self.rows]

    @property
    def row_count(self) -> int:
        return len(self._df)

    @property
    def column_names(self) -> list:
        return list(self._df.columns)

    @property
    def column_count(self) -> int:
        return len(self._df.columns)

    @property
    def shape(self) -> tuple:
        return self._df.shape

    def __setattr__(self, name, value):
        try:
            super(DF, self).__setattr__(name, value)
        except AttributeError:  # don't immediately raise AttributeError
            self._df.__setattr__(name, value)  # instead, invoke setter on underlying pandas DataFrame

    def __getattribute__(self, name):
        try:
            return super(DF, self).__getattribute__(name)
        except AttributeError:  # don't immediately raise AttributeError
            return self._df.__getattribute__(name)  # instead, invoke getter on underlying pandas DataFrame


class RandomDF(DF):
    def __init__(self,
                 rows: int = None,
                 columns: int = None,
                 data_type: type = float,
                 distribution: str = 'uniform'):
        """
        todo: add summary of class
        :param rows: Number of rows to create RandomDF with. If None, a random number of rows between 5 and 200 will be chosen
        :param columns: Number of columns to create RandomDF with. If None, a random number of columns between 5 and 200 will be chosen
        :param data_type: Type of random object to create and populate DF with. Options are float (default), int, and str
        :param distribution: Type of distribution to draw random numbers from (ignored if data_type=str. Options are uniform (default) and normal
        """
        if rows is None:
            rows = np_random.randint(5, 200)
        if columns is None:
            columns = np_random.randint(5, 200)
        self.number_of_rows = rows
        self.number_of_columns = columns
        self.data_type = data_type
        self.distribution = distribution
        column_names = self._create_str_column_names(columns)
        data = {}
        for column_count in range(self.number_of_columns):
            data[column_names[column_count]] = []
            for _ in range(self.number_of_rows):
                data[column_names[column_count]].append(self._get_random_data_point(data_type, distribution))
        super().__init__(data=data)

    def regenerate(self,
                   number_of_rows: int = None,
                   number_of_columns: int = None,
                   data_type: type = None,
                   distribution: str = None) -> None:
        self._df = RandomDF(rows=number_of_rows if number_of_rows is not None else self.number_of_rows,
                            columns=number_of_columns if number_of_columns is not None else self.number_of_columns,
                            data_type=data_type if data_type is not None else self.data_type,
                            distribution=distribution if distribution is not None else self.distribution)._df

    # Static methods
    @staticmethod
    def _get_random_data_point(data_type: type, distribution: str) -> object:
        if data_type is str:
            return np_random.choice([letter for letter in ALPHABET])
        elif data_type in {float, int}:
            if distribution not in DISTRIBUTIONS:
                raise ValueError(f"Cannot draw random number from {distribution} distribution. "
                                 f"Available distributions include {DISTRIBUTIONS}")
            return data_type(np_random.__getattribute__(distribution)())
        elif data_type is None:
            return None
        else:
            raise TypeError(f"Invalid type for RandomDF values {data_type}")


class EmptyDF(DF):
    def __init__(self,
                 number_of_rows: int = None,
                 number_of_columns: int = None,
                 column_headers: bool = True):
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
        data = {}
        if number_of_columns is not None:
            if column_headers:
                column_names = self._create_str_column_names(number_of_columns)
                for column_count in range(self.number_of_columns):
                    data[column_names[column_count]] = [None for _ in range(self.number_of_rows)]
            else:
                data = [[None for _ in range(number_of_columns)] for _ in range(number_of_rows)]
        super().__init__(data=data)
