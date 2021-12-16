from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series

from pandaSuit.common.constant.plot import MULTI_BAR_PLOT_TYPES
from pandaSuit.plot.plot import Plot


class BarPlot(Plot):
    def __init__(self,
                 x: Series or dict or list,
                 y: Series or dict or list,
                 multi_bar_type: str = "beside",
                 bar_width: float or int = None,
                 **kwargs):
        self._shown = False
        self.x = x
        self.y = y
        if isinstance(self.y, list) and len(self.y) > 1:
            if multi_bar_type is not None:
                if multi_bar_type in MULTI_BAR_PLOT_TYPES:
                    self.multi_bar_type = multi_bar_type
                else:
                    raise ValueError("Must specify an alternate type of bar plot when supplying multiple "
                                     "y arguments (e.g. stacked, beside, etc.)")
            else:
                raise ValueError("Must specify an alternate type of bar plot when supplying multiple "
                                 "y arguments (e.g. stacked, beside, etc.)")
        else:
            self.multi_bar_type = None
        self.bar_width = bar_width if bar_width is not None else 0.9/len(y)
        self.figure, self.axes = plt.subplots()
        self.title = kwargs.get('title')
        self.x_label = kwargs.get('x_label')
        self.y_label = kwargs.get('y_label')
        self.legend = kwargs.get('legend')
        super().__init__(self.figure, self.axes)

    def display(self) -> None:
        if self._shown:
            self = BarPlot(x=self.x,
                           y=self.y,
                           multi_bar_type=self.multi_bar_type,
                           bar_width=self.bar_width,
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
        if self.multi_bar_type is None:
            self._construct_base_plot()
        elif self.multi_bar_type == "beside":
            self._construct_side_by_side_plot()
        elif self.multi_bar_type == "stacked":
            self._construct_stacked_plot()
        else:
            pass  # todo: add error handler
        self._add_plot_features()

    def _add_plot_features(self) -> None:
        if self.title is not None:
            self.axes.set_title(self.title)
        if self.x_label is not None:
            self.axes.set_xlabel(self.x_label)
        if self.y_label is not None:
            self.axes.set_ylabel(self.y_label)
        if self.legend:
            self.axes.legend()

    def _construct_base_plot(self) -> None:
        self.axes.bar(self.x if isinstance(self.x, list) else self.x.to_list(), self.y[0].to_list(),
                      label=self.y[0].name)

    def _construct_side_by_side_plot(self) -> None:
        x_indexes = np.arange(len(self.x))
        counter = 0
        for y in self.y:
            self.axes.bar(x_indexes-(self.bar_width*counter),
                          y if isinstance(y, list) else y.to_list(),
                          width=self.bar_width,
                          label=y.name)
            counter += 1
        plt.xticks(ticks=x_indexes, labels=self.x if isinstance(self.x, list) else self.x.to_list())

    def _construct_stacked_plot(self) -> None:
        for y in self.y:
            self.axes.bar(self.x if isinstance(self.x, list) else self.x.to_list(),
                          y if isinstance(y, list) else y.to_list(),
                          label=y.name)
