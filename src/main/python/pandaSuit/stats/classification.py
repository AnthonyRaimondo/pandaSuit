from sklearn.tree import DecisionTreeClassifier
from pandas import Series, DataFrame


class ClassificationTree:
    def __init__(self, dependent: Series, independent: Series or DataFrame):
        self.dependent = dependent
        self.independent = independent
        self.model = self._fit()

    def _fit(self) -> DecisionTreeClassifier:
        y = self.dependent.to_list()
        if isinstance(self.independent, Series):
            x = self.independent.to_numpy().reshape(-1, 1)
        else:
            x = self.independent
        return DecisionTreeClassifier(random_state=0).fit(X=x, y=y)

    def predict(self, input: Series or dict or int or float):
        if isinstance(input, (Series, dict)):
            return self.model.predict(DataFrame([input]).values)[0]
        else:
            return self.model.predict(input)[0]
