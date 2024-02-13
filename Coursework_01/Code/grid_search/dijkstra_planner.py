'''
Created on 2 Jan 2022

@author: ucacsjj
'''

from collections import deque
from math import sqrt, isinf
from queue import PriorityQueue
from typing import Optional, Tuple

from .occupancy_grid import OccupancyGrid
from .planner_base import PlannerBase
from .search_grid import SearchGrid, SearchGridCell
from .search_grid_drawer import SearchGridDrawer

class DijkstraPlanner(PlannerBase):

    # This implements Dijkstra. The priority queue is the path length
    # to the current position.
    
    def __init__(self, occupancy_grid: OccupancyGrid):
        PlannerBase.__init__(self, occupancy_grid)
        self.priority_queue = PriorityQueue()  # type: ignore
        self.searchCount = 0
        self.maxNodesStored = 0
        self.start_added_to_queue = False
        self.totalNodesStored = 0
        self.avgNodesStored = 0

    def reset_statistics(self):
        self.searchCount = 0
        self.maxNodesStored = 0
        self.totalNodesStored = 0
        self.avgNodesStored = 0


    def push_cell_onto_queue(self, cell: SearchGridCell):
        if cell.parent:
            cell.path_cost = cell.parent.path_cost + self.compute_l_stage_additive_cost(cell.parent, cell)
        else:
            cell.path_cost = 0

        if (node_len := self.priority_queue.qsize()) > self.maxNodesStored:
            self.maxNodesStored = node_len

        self.priority_queue.put((cell.path_cost, cell))
        self.searchCount += 1
        self.totalNodesStored += self.priority_queue.qsize()
        self.avgNodesStored = self.totalNodesStored/self.searchCount


    def is_queue_empty(self) -> bool:
        return self.priority_queue.qsize() == 0


    def pop_cell_from_queue(self) -> SearchGridCell:
        return self.priority_queue.get()[1]


    def resolve_duplicate(self, cell: SearchGridCell, parent_cell: SearchGridCell):
        tentative = parent_cell.path_cost + self.compute_l_stage_additive_cost(parent_cell, cell)

        if tentative < cell.path_cost:
            cell.parent = parent_cell
            if (cell.path_cost, cell) in self.priority_queue.queue:
                self.priority_queue.queue.remove((cell.path_cost, cell))
                cell.path_cost = tentative
                self.priority_queue.put((cell.path_cost, cell))
