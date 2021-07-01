from __future__ import annotations

from random import randint
from copy import copy

import pandas
from toolbox.list_elements import index_dictionary

from pandaSuit.linear import LinearModel
from pandaSuit.logistic import LogisticModel
from pandaSuit.constant.date_constants import DATE_GROUPINGS


class DF:
    def __init__(self, data=None):
        if data is not None:
            self.df = pandas.DataFrame(data)
        else:
            self.df = pandas.DataFrame()

    def select(self,
               row: list or int or str = None,
               column: list or int or str = None,
               pandas_return_type: bool = True) -> pandas.DataFrame or pandas.Series or DF:
        if row is None:
            if self._names_supplied(column):
                result = self.df[column]
            else:
                result = self.df.iloc[:, column]
        else:
            if column is None:
                if self._names_supplied(row):
                    result = self.df.loc[row]
                else:
                    result = self.df.iloc[row]
            else:
                if self._names_supplied(row) and self._names_supplied(column):
                    result = self.df.loc[row, column]
                else:
                    if self._names_supplied(row):
                        result = self.df.loc[row].iloc[:, column]
                    else:
                        result = self.df.iloc[row, column]
        if pandas_return_type:
            return result
        else:
            return DF(result)

    def where(self, column_name: str, some_value: object, pandas_return_type: bool = True) -> pandas.DataFrame:
        if isinstance(some_value, str):
            result = self.df[self.df[column_name].str.contains(some_value, na=False)]
        else:
            result = self.df.loc[self.df[column_name] == some_value]
        return result if pandas_return_type else DF(result)

    def where_not(self, column_name: str, some_value: object, pandas_return_type: bool = True) -> pandas.DataFrame:
        if isinstance(some_value, str):
            result = self.df[~self.df[column_name].isin([some_value])]
        else:
            result = self.df.loc[self.df[column_name] != some_value]
        return result if pandas_return_type else DF(result)

    def random_row(self) -> pandas.DataFrame:
        return self.df.iloc[randint(0, self.df.shape[0]-1)]

    def regress(self, y: str or int, x: list or str or int, logit: bool = False) -> LinearModel or LogisticModel:
        if logit:
            return LogisticModel(dependent=self.select(column=y), independent=self.select(column=x))
        else:
            return LinearModel(dependent=self.select(column=y), independent=self.select(column=x))

    def not_null(self, column: str, pandas_return_type: bool = True) -> pandas.DataFrame:
        result = self.df[self.df[column].notna()]
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
            date_group_by_object = self.df.groupby(pandas.to_datetime(self.select(column=column)).dt.strftime(grouping))
            return {date_key: DF(date_group_by_object.get_group(date_key)) for date_key in list(date_group_by_object.groups.keys())}

    def sum_product(self, columns: list) -> int or float:
        product_column = 1
        for column in columns:
            product_column *= self.select(column=column)
        return product_column.sum()

    def update(self,
               row: int or str or list = None,
               column: int or str or list = None,
               to: object = None,
               in_place: bool = True) -> DF or None:
        if in_place:
            if isinstance(column, str):
                self.df.at[row, column] = to
            else:
                self.df.iat[row, column] = to
        else:
            _df = copy(self.df)
            if isinstance(column, str):
                _df.at[row, column] = to
            else:
                _df.iat[row, column] = to
            return DF(_df)

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
                    return DF(copy(self.df))._append_column(column, in_place)._append_row(row, in_place)
            else:
                if in_place:
                    self._append_row(row, in_place)
                    self._append_column(column, in_place)
                else:
                    return DF(copy(self.df))._append_row(row, in_place)._append_column(column, in_place)
        else:
            raise Exception("row or column parameter must be set")

    def _append_row(self, row: pandas.Series, in_place: bool) -> DF or None:
        if in_place:
            self.df.append(row, ignore_index=True)
        else:
            _df = copy(self.df)
            _df.append(row, ignore_index=True)
            return DF(_df)

    def _append_column(self, column: pandas.Series, in_place: bool) -> DF or None:
        if in_place:
            self.df.insert(self.column_count, column.name, column, True)
        else:
            _df = copy(self.df)
            _df.insert(self.column_count, column.name, column, True)
            return DF(_df)

    @staticmethod
    def _names_supplied(selector: int or str or list) -> bool:
        if isinstance(selector, list):
            return isinstance(selector[0], str)
        else:
            return isinstance(selector, str)

    @property
    def is_empty(self) -> bool:
        return self.df.empty

    @property
    def rows(self) -> list:
        return [pandas.Series(row[1]) for row in self.df.iterrows()]

    @property
    def column_count(self) -> int:
        return len(self.df.columns)

    @property
    def row_count(self) -> int:
        return len(self.df)
