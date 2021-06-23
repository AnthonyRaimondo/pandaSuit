from random import randint

from pandas import DataFrame, Series


class DF:
    def __init__(self, data):
        self.df = DataFrame(data)

    def select(self,
               row: list or int or str = None,
               column: list or int or str = None,
               pandas_return_type: bool = True) -> DataFrame or Series:
        if row is None:
            if self._names_supplied(column):
                return self.df[column]
            else:
                return self.df.iloc[:, column]
        else:
            if column is None:
                if self._names_supplied(row):
                    return self.df.loc[row]
                else:
                    return self.df.iloc[row]
            else:
                if self._names_supplied(row) and self._names_supplied(column):
                    return self.df.loc[row, column]
                else:
                    if self._names_supplied(row):
                        return self.df.loc[row].iloc[:, column]
                    else:
                        return self.df.iloc[row, column]

    def where(self, column_name: str, some_value: object, pandas_return_type: bool = True) -> DataFrame:
        result = self.df.loc[self.df[column_name] == some_value]
        if not pandas_return_type:
            return DF(result)
        return result

    def random_row(self) -> DataFrame:
        return self.df.iloc[randint(0, self.df.shape[0]-1)]

    @staticmethod
    def _names_supplied(selector: int or str or list) -> bool:
        if isinstance(selector, list):
            return isinstance(selector[0], str)
        else:
            return isinstance(selector, str)

    @property
    def is_empty(self) -> bool:
        return self.df.empty
