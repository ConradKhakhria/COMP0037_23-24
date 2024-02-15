'''
Created on 2 Jan 2022

@author: ucacsjj
'''

from .occupancy_grid import OccupancyGrid
from .dijkstra_planner import DijkstraPlanner
import math

class AStarPlanner(DijkstraPlanner):
    def __init__(self, occupancy_grid: OccupancyGrid):
        super().__init__(occupancy_grid)

    def push_cell_onto_queue(self, cell):
        # g(x)
        if cell.parent:
            cell.path_cost = cell.parent.path_cost + self.compute_l_stage_additive_cost(cell.parent, cell)
        else:
            cell.path_cost = 0

        # h(x)
        h_cost = self.euclidean_distance(cell, self.goal)

        # f(x) = g(x)+ h(x)
        f_cost = cell.path_cost + h_cost

        self.priority_queue.put((f_cost, cell))

        self._update_search_statistics()

    def euclidean_distance(self, cell1, cell2):
        x1, y1 = cell1.coords()
        x2, y2 = cell2.coords()
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _update_search_statistics(self):
        self.searchCount += 1
        current_queue_size = self.priority_queue.qsize()
        self.totalNodesStored += current_queue_size
        self.avgNodesStored = self.totalNodesStored / self.searchCount
        if current_queue_size > self.maxNodesStored:
            self.maxNodesStored = current_queue_size
