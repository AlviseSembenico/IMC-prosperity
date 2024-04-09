from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pyvista as pv
from pyvista import CellType


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
def create_order_grid(time2orders: Dict[int, List[Order]]):
    all_cells = []
    for timestep, orders in time2orders.items():
        cells = [
            Cell(
                position=Point(x=timestep / 100, y=order.price, z=0),
                size=Box(width=0.7, height=0.7, depth=order.volume),
            )
            for order in orders
        ]
        all_cells.extend(cells)
    grid = make_grid(all_cells)
    return grid


def order_visualizer(
    buy_time2orders: Dict[int, List[Order]], sell_time2orders: Dict[int, List[Order]]
):
    buy_grid = create_order_grid(buy_time2orders)
    sell_grid = create_order_grid(sell_time2orders)

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

    xrng = np.arange(timestep.min(), timestep.max() + 10, 10, dtype=np.float32)
    yrng = np.arange(prices.min(), prices.max() + 10, 10, dtype=np.float32)
    zrng = np.arange(volumes.min(), volumes.max() + 10, 10, dtype=np.float32)
    x, y, z = np.meshgrid(xrng, yrng, zrng, indexing="ij")

    grid = pv.StructuredGrid(x, y, z)

    pv.set_plot_theme("dark")

    pl = pv.Plotter()

    buy_mesh = pl.add_mesh(buy_grid, color="green", show_edges=True)
    sell_mesh = pl.add_mesh(sell_grid, color="red", show_edges=True)

    pl.add_mesh(grid, show_edges=True, style="wireframe")

    class SetVisibilityCallback:
        """Helper callback to keep a reference to the actor being modified."""

        def __init__(self, actor):
            self.actor = actor

        def __call__(self, state):
            self.actor.SetVisibility(state)

    pl.add_checkbox_button_widget(
        SetVisibilityCallback(buy_mesh),
        value=True,
        position=(5.0, 5.0),
        size=20,
        border_size=2,
        color_on="green",
        color_off="gray",
    )

    pl.add_checkbox_button_widget(
        SetVisibilityCallback(sell_mesh),
        value=True,
        position=(25.0, 5.0),
        size=20,
        border_size=2,
        color_on="red",
        color_off="gray",
    )

    pl.add_camera_orientation_widget(animate=True, n_frames=20)
    pl.enable_parallel_projection()

    def set_camera_position(plane: str):
        assert plane == "xy" or plane == "yz" or plane == "zx"
        pl.camera_position = plane
        pl.zoom_camera(8)

    pl.add_key_event("x", lambda: set_camera_position("yz"))
    pl.add_key_event("y", lambda: set_camera_position("zx"))
    pl.add_key_event("z", lambda: set_camera_position("xy"))

    pl.show()


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
        grid = create_order_grid(timestep, [Order(1, 1), Order(2, 2)])
        grids.append(grid)

    pl = pv.Plotter()
    for grid in grids:
        pl.add_mesh(grid, show_edges=True)
    pl.add_camera_orientation_widget(animate=True, n_frames=20)
    # pl.enable_joystick_style()
    # pl.enable_flight_to_right_click()
    # pl.enable_zoom_style()
    pl.show()
