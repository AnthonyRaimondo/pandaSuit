from __future__ import annotations

import matplotlib.pyplot as plt

from pandaSuit.plot.plot import Plot


class PiePlot(Plot):
    def __init__(self, slices: dict, **kwargs):
        self._shown = False
        self.slices = slices
        self.figure, self.axes = plt.subplots()
        self.title = kwargs.get('title')
        self.data_labels = kwargs.get("data_labels")
        self.edge_color = kwargs.get("edge_color")
        super().__init__(self.figure, self. axes)

    def display(self) -> None:
        if self._shown:
            self = PiePlot(self.slices)
            self.display()
        else:
            self.create_plot()
            self._shown = True
            plt.show()

    def create_plot(self) -> None:
        self.axes.pie(list(self.slices.values()),
                      labels=list(self.slices.keys()),
                      wedgeprops={'edgecolor': self.edge_color if self.edge_color else 'black'},
                      autopct='%1.1f%%')
        self._add_chart_features()

    def _add_chart_features(self) -> None:
        if self.title is not None:
            self.axes.set_title(self.title)
        # if self.data_labels is not None:
        #     self.axes.set_autopct('%1.1f%%')
