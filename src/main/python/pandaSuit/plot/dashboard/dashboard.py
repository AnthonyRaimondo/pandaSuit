import tkinter

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import Series

from pandaSuit.df import DF, EmptyDF
from pandaSuit.plot.dashboard.tile import Tile
from pandaSuit.plot.plot import Plot


class Dashboard:
    def __init__(self,
                 *plots,
                 root: tkinter.Tk or tkinter.Frame = None,
                 rows: int = 1,
                 columns: int = 1,
                 layout: DF = None,
                 title: str = "",
                 background_color: str = "white"):
        self.layout = layout if layout is not None else EmptyDF(rows=rows, columns=columns, column_headers=False)
        for plot in plots:
            self.add_plot(plot)
        self.title = title
        self.background_color = background_color
        if root is not None:
            self.root = root
            if isinstance(root, tkinter.Tk):
                self.root.withdraw()
                self.dashboard = tkinter.Toplevel(self.root)
                self.dashboard.title(self.title)
            else:
                self.dashboard = tkinter.Frame(self.root)
        else:
            self.root = tkinter.Tk()
            self.root.withdraw()
            self.dashboard = tkinter.Toplevel(self.root)
            self.dashboard.title(self.title)
        self.dashboard.config(bg=self.background_color)
        self._shown = False

    def add_plot(self, plot: Plot, row: int = None, column: int = None) -> None:
        if row is not None and column is not None:
            if self.layout.select(row=row, column=column, pandas_return_type=True) is None:
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

    def display(self, standalone: bool = True) -> tkinter.Frame or None:
        if self._shown:
            rows, columns = self.layout.shape
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
            if standalone:
                self.dashboard.protocol("WM_DELETE_WINDOW", self.on_closing)
                self.dashboard.grab_set()
                self.root.mainloop()
            else:
                return self.dashboard

    def on_closing(self):
        self.dashboard.grab_release()
        self.root.destroy()

    def add_row(self, columns: int = None) -> None:
        pass

    def add_column(self, rows: int = None) -> None:
        pass

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
        self._augment_layout()
        return self._next_available_position()

    def _augment_layout(self) -> None:
        if self.layout.column_count > self.layout.row_count:
            self.layout.append(row=Series(name=self.layout.row_count, data=[None] * self.layout.column_count), in_place=True)
        else:
            self.layout.append(column=Series(name=self.layout.column_count, data=[None] * self.layout.row_count), in_place=True)
