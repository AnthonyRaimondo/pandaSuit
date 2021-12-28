from __future__ import annotations

from collections import deque
from copy import copy

import pandas as pd
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
            self._df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        else:
            self._df = pd.DataFrame()
        self._unwind = deque()

    def select(self,
               row: list or int or str = None,
               column: list or int or str = None,
               pandas_return_type: bool = False) -> pd.DataFrame or pd.Series or DF:
        if row is None:
            if self._names_supplied(column):
                result = self.dataframe[column]
            else:
                result = self.dataframe.iloc[:, column]
        else:
            if column is None:
                if self._names_supplied(row):
                    result = self.dataframe.loc[row]
                else:
                    result = self.dataframe.iloc[row]
            else:
                if self._names_supplied(row) and self._names_supplied(column):
                    result = self.dataframe.loc[row, column]
                else:
                    if self._names_supplied(row):
                        if self._names_supplied(column):
                            result = self.dataframe.loc[row, column]
                        else:
                            result = self.dataframe.loc[row, :][column]
                    else:
                        if self._names_supplied(column):
                            result = self.dataframe.iloc[row, :][column]
                        else:
                            result = self.dataframe.iloc[row, column]

        return self._df_query_return(result, pandas_return_type)

    def slice(self,
              from_row: int or str = 0,
              to_row: int or str = -1,
              from_column: int or str = 0,
              to_column: int or str = -1,
              pandas_return_type: bool = False) -> pd.DataFrame or pd.Series or DF:

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

        result = self.dataframe.iloc[from_row:to_row, from_column:to_column]

        return self._df_query_return(result, pandas_return_type)

    def where(self, column_name: str, some_value: object, pandas_return_type: bool = True) -> pd.DataFrame:
        if isinstance(some_value, str):
            result = self.dataframe[self.dataframe[column_name].str.contains(some_value, na=False)]
        else:
            result = self.dataframe.loc[self.dataframe[column_name] == some_value]
        return result if pandas_return_type else DF(result)

    def where_not(self, column_name: str, some_value: object, pandas_return_type: bool = True) -> pd.DataFrame:
        if isinstance(some_value, str):
            result = self.dataframe[~self.dataframe[column_name].isin([some_value])]
        else:
            result = self.dataframe.loc[self.dataframe[column_name] != some_value]
        return result if pandas_return_type else DF(result)

    def random_row(self) -> pd.DataFrame:
        return self.dataframe.iloc[np_random.randint(0, self.dataframe.shape[0] - 1)]

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
                        y=[pd.Series(column[1]) for column in self.select(column=list(y)).iteritems()],
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
                       y=[pd.Series(column[1]) for column in self.select(column=list(bars)).iteritems()] if len(bars) > 0 else [self.dataframe.sum()],
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
            return PiePlot(self.dataframe.sum().to_dict())
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
                           y=[pd.Series(column[1]) for column in self.select(column=list(y)).iteritems()],
                           y_label=y[0] if len(y) == 1 and isinstance(y[0], str) else None,
                           x_label=x if isinstance(x, str) else None,
                           best_fit_line=best_fit_line)

    def histogram(self, y: int or str, bins: int = 10) -> Histogram:
        return Histogram(y=self.select(column=y), bins=bins)

    def where_null(self, column: str, pandas_return_type: bool = True) -> DF or pd.DataFrame:
        result = self.dataframe[self.dataframe[column].isnull()]
        return result if pandas_return_type else DF(result)

    def where_not_null(self, column: str, pandas_return_type: bool = True) -> DF or pd.DataFrame:
        result = self.dataframe[self.dataframe[column].notna()]
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
            date_group_by_object = self.dataframe.groupby(pd.to_datetime(self.select(column=column)).dt.strftime(grouping))
            return {date_key: DF(date_group_by_object.get_group(date_key)) for date_key in list(date_group_by_object.groups.keys())}

    def sum_product(self, *columns: int or str) -> int or float:
        product_column = pd.Series([1]*self.row_count)
        for column in columns:
            product_column *= self.select(column=column, pandas_return_type=True)
        return product_column.sum()

    @reversible
    def update(self, row: int or str = None, column: int or str = None, to: object = None, in_place: bool = True) -> DF or None:
        if in_place:
            if column is not None:
                if row is not None:
                    if isinstance(column, str):
                        if isinstance(row, str):
                            self.dataframe.loc[row, column] = to
                        else:
                            self.dataframe.iloc[row, find_indexes(self.column_names, column)] = to
                    else:
                        if isinstance(row, str):
                            self.dataframe.iloc[find_indexes(self.row_names, row), column] = to
                        else:
                            self.dataframe.iloc[row, column] = to
                else:
                    if isinstance(column, str):
                        self.dataframe.loc[self.row_names, column] = to
                    else:
                        self.dataframe.iloc[create_index_list(self.row_count), column] = to
            elif row is not None:  # todo
                if isinstance(row, str):
                    pass
                else:
                    pass
            else:
                pass  # Note: if row=None and column=None, Exception is thrown from decorator. No need to raise one here.
        else:
            _df = DF(self.dataframe)
            _df.update(row=row, column=column, to=to, in_place=True)
            return _df

    @reversible
    def append(self, row: pd.Series = None, column: pd.Series = None, in_place: bool = True) -> DF or None:
        if row is not None and column is None:
            if in_place:
                self._append_row(row, in_place)
            else:
                return self._append_row(row, in_place)
        elif row is None and column is not None:
            if column.name is None:  # without a column name, undo() does not work because it calls self.dataframe.drop(None)
                column.name = self._create_column_name()
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
                    return DF(copy(self.dataframe))._append_column(column, in_place)._append_row(row, in_place)
            else:
                if in_place:
                    self._append_row(row, in_place)
                    self._append_column(column, in_place)
                else:
                    return DF(copy(self.dataframe))._append_row(row, in_place)._append_column(column, in_place)
        else:
            self._remove_latest_unwind_operation()
            raise Exception("row or column parameter must be set")

    @reversible
    def insert(self, index: int, row: pd.Series or pd.DataFrame = None, column: pd.Series or pd.DataFrame = None, in_place: bool = True) -> DF or None:
        if in_place:
            if row is not None:
                before = self.slice(to_row=index, pandas_return_type=True)
                after = self.slice(from_row=index, pandas_return_type=True)
                if isinstance(row, pd.Series):
                    row.name = self.row_count
                    self._df = pd.concat([before, pd.DataFrame(row).transpose(), after])
                else:  # pd.Dataframe
                    self._df = pd.concat([before, self._update_row_names(row), after], axis=0)
            elif column is not None:
                if isinstance(column, pd.Series):
                    if column.name is None:
                        column.name = self.column_count
                    self.dataframe.insert(loc=index, column=column.name, value=column)
                else:  # pd.DataFrame
                    before = self.slice(to_column=index, pandas_return_type=True)
                    after = self.slice(from_column=index, pandas_return_type=True)
                    self._df = pd.concat([before, self._update_column_names(column), after], axis=1)
            else:
                self._remove_latest_unwind_operation()
                raise Exception("Must pass row OR column")
        else:
            _df = DF(self.dataframe)
            _df.insert(index=index, row=row, column=column, in_place=True)
            return _df

    @reversible
    def remove(self, row: int or str or list = None, column: int or str or list = None):
        if row is not None:
            if isinstance(row, int):  # drop by row index
                rows_to_drop = self.row_names[row]
            elif isinstance(row, list):  # drop by row indexes
                rows_to_drop = [self.row_names[r] for r in row]
            else:  # drop by row name
                rows_to_drop = row
            self.dataframe.drop(rows_to_drop, axis=0, inplace=True)
        if column is not None:
            if isinstance(column, int):  # drop by column index
                columns_to_drop = self.column_names[column]
            elif isinstance(column, list):  # drop by column indexes
                columns_to_drop = [self.column_names[c] for c in column]
            else:  # drop by column name
                columns_to_drop = column
            self.dataframe.drop(columns_to_drop, axis=1, inplace=True)

    def undo(self) -> None:
        """ Reverts the most recent change to self """
        try:
            unwind_object: Unwind = self._unwind.pop()
        except IndexError:
            raise Exception("There are no DF manipulations to undo")
        self.__getattribute__(unwind_object.function)(**unwind_object.args[0])

    def reset(self) -> None:
        self._df = DF(data=self.data).dataframe

    def _append_row(self, row: pd.Series, in_place: bool) -> DF or None:
        if in_place:
            self._df = self.dataframe.append(other=row, ignore_index=True)
        else:
            _df = copy(self.dataframe)
            _df = _df.append(other=row, ignore_index=True)
            return DF(_df)

    def _append_column(self, column: pd.Series, in_place: bool) -> DF or None:
        if in_place:
            self.dataframe.insert(loc=self.column_count, column=column.name, value=column, allow_duplicates=True)
        else:
            _df = copy(self.dataframe)
            _df.insert(loc=self.column_count, column=column.name, value=column, allow_duplicates=True)
            return DF(_df)

    def _create_column_name(self) -> str:
        new_cols = len([col for col in self.column_names if "new_col" in col])
        return f"new_col{new_cols + 1}"

    def _remove_latest_unwind_operation(self):
        del self._unwind[-1]

    def _update_row_names(self, rows: pd.DataFrame) -> pd.DataFrame:
        offset, name_changes = 0, {}
        for row in rows.iterrows():
            if isinstance(row[1].name, int) and row[1].name == offset:  # update row names if it appears they have generic index (i.e., [0, 1, 2...])
                name_changes[row[1].name] = self.row_count + offset
                offset += 1
        return rows.rename(name_changes, inplace=False, axis=0)

    def _update_column_names(self, columns: pd.DataFrame) -> pd.DataFrame:
        offset, name_changes = 0, {}
        for column in columns:
            if isinstance(columns[column].name, int) and columns[column].name == offset:  # update column names if it appears they have generic index (i.e., [0, 1, 2...])
                name_changes[columns[column].name] = self.column_count + offset
                offset += 1
        return columns.rename(name_changes, inplace=False, axis=1)

    @staticmethod
    def _df_query_return(result, pandas_return_type: bool):
        if pandas_return_type or not isinstance(result, (pd.Series, pd.DataFrame)):
            return result
        else:
            return DF(result)

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
    def dataframe(self) -> pd.DataFrame:
        return self._df

    @property
    def is_empty(self) -> bool:
        return self.dataframe.empty

    @property
    def rows(self) -> list:
        return [pd.Series(row[1]) for row in self.dataframe.iterrows()]

    @property
    def row_names(self):
        return [row.name for row in self.rows]

    @property
    def row_count(self) -> int:
        return len(self.dataframe)

    @property
    def column_names(self) -> list:
        return list(self.dataframe.columns)

    @property
    def column_count(self) -> int:
        return len(self.dataframe.columns)

    @property
    def shape(self) -> tuple:
        return self.dataframe.shape

    def __setattr__(self, name, value):
        try:
            super(DF, self).__setattr__(name, value)
        except AttributeError:  # don't immediately raise AttributeError
            self.dataframe.__setattr__(name, value)  # instead, invoke setter on underlying pandas DataFrame

    def __getattribute__(self, name):
        try:
            return super(DF, self).__getattribute__(name)
        except AttributeError:  # don't immediately raise AttributeError
            return self.dataframe.__getattribute__(name)  # instead, invoke getter on underlying pandas DataFrame


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
                            distribution=distribution if distribution is not None else self.distribution).dataframe

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
