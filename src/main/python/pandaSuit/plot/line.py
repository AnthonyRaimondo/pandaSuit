from __future__ import annotations

import matplotlib.pyplot as plt
from pandas import Series

from pandaSuit.plot.plot import Plot


class LinePlot(Plot):
    def __init__(self, x: Series or dict or list, y: Series or dict or list, **kwargs):
        self._shown = False
        self.x = x if isinstance(x, Series) else Series(x)
        self.y = y if isinstance(y, Series) else Series(y)
        self.figure, self.axes = plt.subplots()
        self.title = kwargs.get('title')
        self.x_label = kwargs.get('x_label')
        self.y_label = kwargs.get('y_label')
        self.x_ticks = kwargs.get('x_ticks')
        self.x_bottom = kwargs.get("x_bottom")
        self.x_top = kwargs.get("x_top")
        self.y_bottom = kwargs.get("y_bottom")
        self.y_top = kwargs.get("y_top")
        self.legend = kwargs.get('legend')
        self.grid_lines = kwargs.get('grid_lines')
        self.x_scale = kwargs.get('x_scale')
        self.y_scale = kwargs.get('y_scale')
        super().__init__(self.figure, self. axes)

    def display(self) -> None:
        if self._shown:
            self = LinePlot(self.x, self.y,
                            title=self.title,
                            x_label=self.x_label,
                            y_label=self.y_label,
                            x_ticks=self.x_ticks,
                            x_bottom=self.x_bottom,
                            x_top=self.x_top,
                            y_bottom=self.y_bottom,
                            y_top=self.y_top,
                            legend=self.legend,
                            grid_lines=self.grid_lines,
                            x_scale=self.x_scale,
                            y_scale=self.y_scale)
            self.display()
        else:
            self.create_plot()
            self._shown = True
            if self.x_scale:
                plt.xscale(self.x_scale)
            if self.y_scale:
                plt.yscale(self.y_scale)
            plt.show()

    def create_plot(self) -> None:
        self.axes.plot(self.x.to_list(), self.y.to_list(), label=self.y.name)
        self._add_chart_features()

    def _add_chart_features(self) -> None:
        if self.title is not None:
            self.axes.set_title(self.title)
        if self.x_label is not None:
            self.axes.set_xlabel(self.x_label)
        if self.y_label is not None:
            self.axes.set_ylabel(self.y_label)
        if self.x_ticks is not None:
            self.axes.set_xticklabels(self.x_ticks)
        if self.x_bottom is not None:
            self.axes.set_xlim(bottom=self.x_bottom)
        if self.x_top is not None:
            self.axes.set_xlim(top=self.x_top)
        if self.y_bottom is not None:
            self.axes.set_ylim(bottom=self.y_bottom)
        if self.y_top is not None:
            self.axes.set_ylim(top=self.y_top)
        if self.legend:
            self.axes.legend()
        self.axes.grid(False if self.grid_lines is None else self.grid_lines)
