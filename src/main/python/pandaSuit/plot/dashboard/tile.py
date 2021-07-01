from pandaSuit.plot.plot import Plot


class Tile:
    def __init__(self, plot: Plot):
        plot.create_plot()
        self.figure = plot.figure
