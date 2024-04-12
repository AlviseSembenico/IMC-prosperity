from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pyvista as pv
from pyvista import CellType

from statistics import NormalDist


from distributions import NormalDistribution, Sample


@dataclass
class Point:
    x: float
    y: float
    z: float


@dataclass
class Box:
    width: float
    height: float
    depth: float


def make_cell(position: Point, size: Box):
    offset = np.array([[position.x, position.y, position.z]]).astype(float)
    scale = np.array([[size.width, size.height, size.depth]]).astype(float)

    cell = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    ).astype(float)
    cell *= scale
    cell += offset

    return cell


@dataclass
class Cell:
    position: Point
    size: Box


def make_grid(cells: list[Cell]):
    np_cells = [make_cell(cell.position, cell.size) for cell in cells]
    points = np.vstack(np_cells).astype(float)
    cells_hex = np.arange(8 * len(cells)).reshape([len(cells), 8])
    return pv.UnstructuredGrid({CellType.HEXAHEDRON: cells_hex}, points)


Price = int
Volume = int


@dataclass
class Order:
    price: Price
    volume: Volume


@dataclass
class OrderDepth:
    buy_orders: Dict[Price, Volume]
    sell_orders: Dict[Price, Volume]


# def create_order_grid(timestep: int, orders: List[Order]):
def create_order_grid(
    time2orders: Dict[int, List[Order]], cell_size=0.7, extra_depth=0.0
) -> pv.UnstructuredGrid:
    all_cells = []
    for timestep, orders in time2orders.items():
        cells = [
            Cell(
                position=Point(x=timestep / 100, y=order.price, z=0),
                size=Box(
                    width=cell_size, height=cell_size, depth=order.volume + extra_depth
                ),
            )
            for order in orders
        ]
        all_cells.extend(cells)
    grid = make_grid(all_cells)
    return grid


def sliding_windows(lst: list, window_size: int, step=1):
    for i in range(window_size, len(lst), step):
        yield lst[i - window_size : i]


def create_probability_grid(time2orders: Dict[int, List[Order]]):
    time_and_orders = list(time2orders.items())
    time_and_orders.sort(key=lambda x: x[0])

    time_discount = 0.8

    for window in sliding_windows(time_and_orders, window_size=10):
        current_factor = 1.0

        samples = []
        for timestep, orders in window[::-1]:
            for order in orders:
                samples.append(
                    Sample(value=order.price, weight=current_factor * order.volume)
                )
            current_factor *= time_discount

        # calculate normal distribution based on samples
        dist = NormalDistribution.from_samples(samples)

        _, current_orders = window[-1]
        prices = np.array([order.price for order in current_orders])
        price_range = (prices.min(), prices.max())

        from visualize import draw_distribution


@dataclass
class Limits:
    min: float
    max: float


def create_grid(
    x: Limits, y: Limits, z: Limits, spacing: float = 10
) -> pv.StructuredGrid:
    xrng = np.arange(x.min, x.max + spacing, spacing, dtype=np.float32)
    yrng = np.arange(y.min, y.max + spacing, spacing, dtype=np.float32)
    zrng = np.arange(z.min, z.max + spacing, spacing, dtype=np.float32)
    mx, my, mz = np.meshgrid(xrng, yrng, zrng, indexing="ij")
    grid = pv.StructuredGrid(mx, my, mz)
    return grid


@dataclass
class Toggle:
    """Helper callback to keep a reference to the actor being modified."""

    pl: pv.Plotter
    actor: pv.PolyData
    t: bool = True

    def __call__(self, flag: Optional[bool] = None):
        if flag:
            self.t = flag
        else:
            self.t = not self.t
        self.actor.SetVisibility(self.t)
        self.pl.update(force_redraw=True)


@dataclass
class GridArgs:
    grid: pv.UnstructuredGrid
    color: Optional[str] = None
    scalars: Optional[str] = None
    show_edges: bool = True
    style: str = "surface"
    toggle_hotkey: Optional[str] = None


def visualize_grids(grids: List[GridArgs]):
    pv.set_plot_theme("dark")

    pl = pv.Plotter()

    for i, grid_args in enumerate(grids):
        mesh = pl.add_mesh(
            grid_args.grid,
            color=grid_args.color,
            scalars=grid_args.scalars,
            style=grid_args.style,
            show_edges=grid_args.show_edges,
        )

        toggle = Toggle(pl, mesh)

        size = 20
        padding = 5.0
        pl.add_checkbox_button_widget(
            toggle,
            value=True,
            position=(i * size + padding, padding),
            size=20,
            border_size=2,
            color_on=grid_args.color if grid_args.color else "white",
            color_off="gray",
        )
        if grid_args.toggle_hotkey:
            pl.add_key_event(grid_args.toggle_hotkey, toggle)

    pl.add_camera_orientation_widget(animate=True, n_frames=20)
    pl.enable_parallel_projection()

    # def cell_pick_callback(cells):
    #     print()
    #     print(cells)
    #
    # pl.enable_cell_picking(callback=cell_pick_callback)

    # pl.show_grid(font_size=1, n_ylabels=2)

    def set_camera_position(plane: str):
        assert plane == "xy" or plane == "yz" or plane == "zx"
        pl.camera_position = plane
        pl.zoom_camera(8)

    pl.add_key_event("x", lambda: set_camera_position("yz"))
    pl.add_key_event("y", lambda: set_camera_position("zx"))
    pl.add_key_event("z", lambda: set_camera_position("xy"))

    pl.add_key_event("s", lambda: pl.screenshot("screenshot.png"))

    pl.enable_fly_to_right_click()

    pl.show()


def order_visualizer(
    buy_time2orders: Dict[int, List[Order]],
    sell_time2orders: Dict[int, List[Order]],
    trades: Optional[Dict[int, List[Order]]],
):
    buy_grid = create_order_grid(buy_time2orders)
    sell_grid = create_order_grid(sell_time2orders)

    if trades is not None:
        trade_grid = create_order_grid(trades, cell_size=0.8, extra_depth=0.1)

    timestamps = list(buy_time2orders.keys())
    prices = []
    volumes = []
    for buy_orders, sell_orders in zip(
        buy_time2orders.values(), sell_time2orders.values()
    ):
        try:
            prices.extend([int(b.price) for b in buy_orders])
            volumes.extend([b.volume for b in buy_orders])
        except ValueError:
            pass

        try:
            prices.extend([int(s.price) for s in sell_orders])
            volumes.extend([s.volume for s in sell_orders])
        except ValueError:
            pass

    timestep = np.array(timestamps) / 100
    prices = np.array(prices)
    volumes = np.array(volumes)

    xlimits = Limits(timestep.min(), timestep.max())
    ylimits = Limits(prices.min(), prices.max())
    zlimits = Limits(volumes.min(), volumes.max())

    grid = create_grid(xlimits, ylimits, zlimits)

    grids = [
        GridArgs(grid, show_edges=True, style="wireframe", toggle_hotkey="g"),
        GridArgs(buy_grid, color="green", toggle_hotkey="b"),
        GridArgs(sell_grid, color="red", toggle_hotkey="a"),
    ]

    if trades is not None:
        grids.append(GridArgs(trade_grid, color="blue", toggle_hotkey="t"))

    visualize_grids(grids)

    return


def test():
    grid = make_grid(
        [
            Cell(Point(0, 0, 0), Box(1, 1, 1)),
            Cell(Point(0, 0, 2), Box(1, -2, 4)),
        ]
    )


if __name__ == "__main__":
    grids = []
    for timestep in range(10):
        grid = create_order_grid({timestep: [Order(1, 1), Order(2, 2)]})
        grids.append(grid)

    pl = pv.Plotter()
    for grid in grids:
        pl.add_mesh(grid, show_edges=True)
    pl.add_camera_orientation_widget(animate=True, n_frames=20)
    # pl.enable_joystick_style()
    # pl.enable_flight_to_right_click()
    # pl.enable_zoom_style()
    pl.show()
