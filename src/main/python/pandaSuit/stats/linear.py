from sklearn.linear_model import LinearRegression
from pandas import Series, DataFrame


class LinearModel:

    def __init__(self, dependent: Series or DataFrame, independent: Series or DataFrame, intercept: bool = True):
        super().__init__()
        self.dependent = dependent
        self.independent = independent
        self.include_intercept = intercept
        self.model = self._fit()

    def predict(self, input: Series or dict or int or float, practical: bool = False) -> float or list:
        return self._practical_prediction(input) if practical else self._theoretical_prediction(input)

    def _fit(self) -> LinearRegression:
        if isinstance(self.dependent, DataFrame):
            y = self.dependent.values
        else:
            y = self.dependent.to_list()
        if isinstance(self.independent, Series):
            x = self.independent.to_numpy().reshape(-1, 1)
        else:
            x = self.independent
        return LinearRegression(fit_intercept=self.include_intercept).fit(X=x, y=y)

    # Properties
    @property
    def intercept(self) -> float:
        return self.model.intercept_

    @property
    def betas(self) -> list:
        return self.model.coef_.tolist()[0]

    # Private prediction methods
    def _theoretical_prediction(self, input: Series or dict) -> float:
        if isinstance(input, Series):
            return list(self.model.predict(input.to_numpy().reshape(1, -1))[0])
        else:
            return list(self.model.predict(Series(input).to_numpy().reshape(1, -1))[0])

    def _practical_prediction(self, input: Series or dict) -> float:  # todo
        if isinstance(input, dict):
            return (self.betas * Series(input)).sum() + (self.intercept if self.include_intercept else 0.0)
        elif isinstance(input, Series):
            return (self.betas * input).sum() + (self.intercept if self.include_intercept else 0.0)
        elif isinstance(input, (float, int)):
            return (sum(self.betas * input)) + (self.intercept if self.include_intercept else 0.0)
        else:
            raise Exception("Must supply Series, dict, int or float to .predict() method")
