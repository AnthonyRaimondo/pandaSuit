from pandas import DataFrame


class DF:
    def __init__(self, data):
        self.df = DataFrame(data)

    def select(self, row: list or int or str = None, column: list or int or str = None) -> DataFrame:
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
                        return self.df[column].iloc[row]

    @staticmethod
    def _names_supplied(selector: int or str or list) -> bool:
        if isinstance(selector, list):
            return isinstance(selector[0], str)
        else:
            return isinstance(selector, str)
