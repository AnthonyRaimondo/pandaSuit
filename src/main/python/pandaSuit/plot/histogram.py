from matplotlib import pyplot as plt
from pandas import Series

from pandaSuit.plot.plot import Plot


class Histogram(Plot):
    def __init__(self, y: Series or dict or list, **kwargs):
        self._shown = False
        self.y = y
        self.figure, self.axes = plt.subplots()
        self.title = kwargs.get('title')
        self.x_label = kwargs.get('x_label')
        self.y_label = kwargs.get('y_label')
        self.legend = kwargs.get('legend')
        self.bins = kwargs.get('bins')
        super().__init__(self.figure, self.axes)

    def display(self) -> None:
        if self._shown:
            self = Histogram(self.y,
                             title=self.title,
                             x_label=self.x_label,
                             y_label=self.y_label,
                             legend=self.legend)
            self.display()
        else:
            self.create_plot()
            self._shown = True
            plt.show()

    def create_plot(self) -> None:
        self.axes.hist(self.y, bins=self.bins)
        self._add_chart_features()

    def _add_chart_features(self) -> None:
        if self.title is not None:
            self.axes.set_title(self.title)
        if self.x_label is not None:
            self.axes.set_xlabel(self.x_label)
        if self.y_label is not None:
            self.axes.set_ylabel(self.y_label)
        if self.legend:
            self.axes.legend()
