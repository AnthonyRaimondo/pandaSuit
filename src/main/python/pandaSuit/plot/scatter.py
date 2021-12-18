from __future__ import annotations

import matplotlib.pyplot as plt
from pandas import Series

from pandaSuit.plot.plot import Plot
from pandaSuit.stats.linear import LinearModel


class ScatterPlot(Plot):
    def __init__(self, x: Series or dict or list, y: Series or dict or list, **kwargs):
        self._shown = False
        self.x = x
        self.y = y
        self.figure, self.axes = plt.subplots()
        self.title = kwargs.get('title')
        self.x_label = kwargs.get('x_label')
        self.y_label = kwargs.get('y_label')
        self.best_fit_line = kwargs.get('best_fit_line')
        self.legend = kwargs.get('legend')
        super().__init__(self.figure, self.axes)

    def display(self) -> None:
        if self._shown:
            self = ScatterPlot(self.x, self.y,
                               title=self.title,
                               x_label=self.x_label,
                               y_label=self.y_label,
                               best_fit_line=self.best_fit_line,
                               legend=self.legend)
            self.display()
        else:
            self.create_plot()
            self._shown = True
            plt.show()

    def create_plot(self) -> None:
        for y in self.y:
            self.axes.scatter(self.x if isinstance(self.x, list) else self.x.to_list(),
                              y if isinstance(y, list) else y.to_list(), label=y.name)
        self._add_chart_features()

    def _add_chart_features(self) -> None:
        if self.title is not None:
            self.axes.set_title(self.title)
        if self.x_label is not None:
            self.axes.set_xlabel(self.x_label)
        if self.y_label is not None:
            self.axes.set_ylabel(self.y_label)
        if self.best_fit_line:
            for y in self.y:
                self.axes.plot(self._create_best_fit_line(y))
        if self.legend:
            self.axes.legend()

    def _create_best_fit_line(self, y_values: Series) -> list:
        x_values = self._prepare_x_axis_for_regression()
        lm = LinearModel(y_values, x_values)
        return [lm.predict(x) for x in x_values.to_list()]

    def _prepare_x_axis_for_regression(self) -> Series:
        return Series(name=self.x.name, data=[i for i, _ in enumerate(self.x)]) \
            if any(isinstance(x, str) for x in self.x) else self.x
