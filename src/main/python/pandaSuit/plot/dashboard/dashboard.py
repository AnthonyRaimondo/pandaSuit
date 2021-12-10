import tkinter

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import Series

from pandaSuit.df import DF
from pandaSuit.plot.dashboard.tile import Tile
from pandaSuit.plot.plot import Plot


class Dashboard:
    def __init__(self,
                 root: tkinter.Tk = None,
                 rows: int = 1,
                 columns: int = 1,
                 layout: DF = None,
                 title: str = "",
                 background_color: str = "white"):
        self.layout = layout if layout is not None else DF([[None for _ in range(columns)] for _ in range(rows)])
        self.title = title
        self.background_color = background_color
        self.dashboard = root if root is not None else tkinter.Tk()
        self.dashboard.config(bg=self.background_color)
        try:
            self.dashboard.title(self.title)
        except AttributeError:
            pass
        self._shown = False

    def add_plot(self, plot: Plot, row: int = None, column: int = None) -> None:
        if row is not None and column is not None:
            if self.layout.select(row=row, column=column) is None:
                self.layout.update(row=row, column=column, to=Tile(plot))
            else:
                raise Exception(f"There is already a Dashboard Tile placed at position row={row} column={column}")
        else:
            row, column = self._next_available_position()
            self.add_plot(plot, row, column)

    def update_plot(self, row: int, column: int, to: Plot) -> None:
        try:
            self.layout.update(row=row, column=column, to=Tile(to))
        except IndexError:
            raise Exception(f"Dashboard position ({row}, {column}) specified does not exist.")

    def display(self, pop_out: bool = True) -> None:
        if self._shown:
            rows, columns = self.layout._df.shape
            self = Dashboard(rows=rows, columns=columns, layout=self.layout, title=self.title, background_color=self.background_color)
            self.display()
        else:
            row_count = 0
            for row in self.layout.rows:
                column_count = 0
                for tile in row.to_list():
                    if tile is not None:
                        FigureCanvasTkAgg(tile.figure, master=self.dashboard).get_tk_widget().grid(row=row_count, column=column_count)
                    column_count += 1
                row_count += 1
            self._shown = True
            if pop_out:
                self.dashboard.mainloop()
                self.grab_set()

    def add_row(self, columns: int = None) -> None:
        pass

    def add_column(self, rows: int = None) -> None:
        pass

    def grab_set(self):
        self.dashboard.grab_set()

    # Private methods
    def _next_available_position(self) -> tuple:
        row_count = 0
        for row in self.layout.rows:
            column_count = 0
            for tile in list(row):
                if tile is None:
                    return row_count, column_count
                column_count += 1
            row_count += 1
        self._augment_underlying_table()
        return self._next_available_position()

    def _augment_underlying_table(self) -> None:
        if self.layout.column_count > self.layout.row_count:
            self.layout.append(row=Series([None] * self.layout.column_count), in_place=True)
        else:
            self.layout.append(column=Series([None] * self.layout.row_count), in_place=True)
