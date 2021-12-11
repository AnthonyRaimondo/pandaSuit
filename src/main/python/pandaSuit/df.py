from __future__ import annotations

from collections import deque
from random import randint
from copy import copy

import pandas

from pandaSuit.common.unwind import Unwind
from pandaSuit.common.util.list_operations import index_dictionary, create_index_list
from pandaSuit.stats.linear import LinearModel
from pandaSuit.stats.logistic import LogisticModel
from pandaSuit.common.constant.date_constants import DATE_GROUPINGS
from pandaSuit.common.decorators import reversible


class DF:
    def __init__(self, data=None):
        if data is not None:
            self._df = pandas.DataFrame(data)
        else:
            self._df = pandas.DataFrame()
        self._unwind = deque()

    def select(self,
               row: list or int or str = None,
               column: list or int or str = None,
               pandas_return_type: bool = True) -> pandas.DataFrame or pandas.Series or DF:
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
        if pandas_return_type:
            return result
        else:
            return DF(result)

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
        return self._df.iloc[randint(0, self._df.shape[0] - 1)]

    def regress(self, y: str or int, x: list or str or int, logit: bool = False) -> LinearModel or LogisticModel:
        if logit:
            return LogisticModel(dependent=self.select(column=y), independent=self.select(column=x))
        else:
            return LinearModel(dependent=self.select(column=y), independent=self.select(column=x))

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

    def _append_row(self, row: pandas.Series, in_place: bool) -> DF or None:
        if in_place:
            self._df.append(row, ignore_index=True)
        else:
            _df = copy(self._df)
            _df.append(row, ignore_index=True)
            return DF(_df)

    def _append_column(self, column: pandas.Series, in_place: bool) -> DF or None:
        if in_place:
            self._df.insert(self.column_count, column.name, column, True)
        else:
            _df = copy(self._df)
            _df.insert(self.column_count, column.name, column, True)
            return DF(_df)

    # def undo(self):
    #     """
    #     Reverts the most recent change to the Table instance.
    #     """
    #     try:
    #         unwind_object = self._unwind.pop()
    #         try:
    #             if isinstance(unwind_object, Unwind):
    #                 if len(unwind_object.args) > 0:
    #                     unwind_object.function(unwind_object.args)
    #                 else:
    #                     unwind_object.function()
    #             else:
    #                 for unwind_step in reversed(unwind_object):
    #                     if len(unwind_step.args) > 0:
    #                         unwind_step.function(unwind_step.args)
    #                     else:
    #                         unwind_step.function()
    #             del self._change_log[-1]
    #         except Exception as e:
    #             raise Exception(f"Error occurred when attempting to undo step: {self._change_log[-1]}", e)
    #     except IndexError:
    #         pass

    @staticmethod
    def _names_supplied(selector: int or str or list) -> bool:
        if isinstance(selector, list):
            return isinstance(selector[0], str)
        else:
            return isinstance(selector, str)

    @property
    def is_empty(self) -> bool:
        return self._df.empty

    @property
    def rows(self) -> list:
        return [pandas.Series(row[1]) for row in self._df.iterrows()]

    @property
    def column_count(self) -> int:
        return len(self._df.columns)

    @property
    def row_count(self) -> int:
        return len(self._df)

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
