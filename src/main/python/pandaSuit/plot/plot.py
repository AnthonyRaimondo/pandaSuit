from abc import ABC, abstractmethod


class Plot(ABC):
    def __init__(self, figure, axes):
        self.figure = figure
        self.axes = axes

    @abstractmethod
    def display(self) -> None:
        return None

    @abstractmethod
    def create_plot(self) -> None:
        return None
