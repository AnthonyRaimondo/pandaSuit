from math import exp
from random import choice

from sklearn.linear_model import LogisticRegression
from pandas import Series, DataFrame


class LogisticModel:

    def __init__(self, dependent: Series, independent: Series or DataFrame, intercept: bool = True):
        self.dependent = dependent
        self.independent = independent
        self.include_intercept = intercept
        self.model = self._fit()

    def predict(self, input: Series or dict or int or float, practical: bool = False) -> float:
        return self._practical_prediction(input) if practical else self._theoretical_prediction(input)

    def _fit(self) -> LogisticRegression:
        y = self.dependent.to_list()
        if isinstance(self.independent, Series):
            x = self.independent.to_numpy().reshape(-1, 1)
        else:
            x = self.independent
        return LogisticRegression(fit_intercept=self.include_intercept).fit(X=x, y=y)

    @property
    def intercept(self) -> float:
        return self.model.intercept_[0]

    @property
    def betas(self) -> list:
        return list(self.model.coef_.tolist()[0])

    # Private prediction methods
    def _theoretical_prediction(self, input: Series or dict) -> bool:
        if isinstance(input, Series):
            return int(list(self.model.predict(input.to_numpy().reshape(1, -1)))[0]) == 1
        else:
            return int(list(self.model.predict(Series(input).to_numpy().reshape(1, -1)))[0]) == 1

    def _practical_prediction(self, input: Series or dict or int or float) -> float:  # todo
        if isinstance(input, dict):
            for predictor_name, predictor_value in input.items():
                input[predictor_name] = predictor_value + (choice([-1, 1]) * self.independent.standard_deviation(column=predictor_name))
            return self._theoretical_prediction(input)
        elif isinstance(input, Series):
            return self._practical_prediction(input.to_dict())
        elif isinstance(input, (float, int)):
            return self._theoretical_prediction(input + self.independent.std() * choice([-1, 1]))
        else:
            raise Exception("Must supply Series, dict, int or float to .predict() method")
