'''
Created on 2 Jan 2022

@author: ucacsjj
'''

from collections import deque
from math import sqrt
from queue import PriorityQueue
from typing import Optional, Tuple

from .occupancy_grid import OccupancyGrid
from .planner_base import PlannerBase
from .search_grid import SearchGridCell

class DijkstraPlanner(PlannerBase):

    # This implements Dijkstra. The priority queue is the path length
    # to the current position.
    
    def __init__(self, occupancy_grid: OccupancyGrid):
        PlannerBase.__init__(self, occupancy_grid)
        self.priority_queue = PriorityQueue()  # type: ignore
        self.cumulative_distances = []
    
        for _ in range(occupancy_grid.height()):
            self.cumulative_distances.append([float("inf")]*occupancy_grid.width())


    def push_cell_onto_queue(self, cell: SearchGridCell):
        x, y = cell.coords()

        if cell.is_start:
            self.cumulative_distances[y][x] = 0

        self.priority_queue.put((self.cumulative_distances[y][x], cell))


    def is_queue_empty(self) -> bool:
        return self.priority_queue.empty()
    

    def pop_cell_from_queue(self) -> SearchGridCell:
        return self.priority_queue.get()[1]


    def resolve_duplicate(self, cell: SearchGridCell, parent_cell: SearchGridCell):
        px, py = parent_cell.coords()
        x, y = cell.coords()

        tentative = self.cumulative_distances[py][px] + ((x - px)**2 + (y - py)**2)**0.5

        if tentative < self.cumulative_distances[y][x]:
            self.cumulative_distances[y][x] = tentative
