from .planner_base import PlannerBase
from .occupancy_grid import OccupancyGrid
from .search_grid import SearchGridCell
from typing import List


class DepthFirstPlanner(PlannerBase):

    # This implements a simple LIFO search algorithm

    #

    def __init__(self, occupancyGrid: OccupancyGrid):
        PlannerBase.__init__(self, occupancyGrid)
        self.lifoQueue: List[SearchGridCell] = list()
        self.searchCount = 0
        self.maxNodesStored = 0
        self.totalNodesStored = 0
        self.avgNodesStored = 0

    def reset_statistics(self):
        self.searchCount = 0
        self.maxNodesStored = 0
        self.totalNodesStored = 0
        self.avgNodesStored = 0

    # Simply put on the end of the queue
    def push_cell_onto_queue(self, cell: SearchGridCell):
        self.lifoQueue.append(cell)
        self.searchCount += 1
        self.totalNodesStored += len(self.lifoQueue)
        self.avgNodesStored = self.totalNodesStored/self.searchCount

        node_len = len(self.lifoQueue)
        if node_len > self.maxNodesStored:
            self.maxNodesStored = node_len

    # Check the queue size is zero
    def is_queue_empty(self) -> bool:
        return not self.lifoQueue

    # Simply pull from the front of the list
    def pop_cell_from_queue(self) -> SearchGridCell:
        cell = self.lifoQueue.pop()
        return cell

    def resolve_duplicate(self, cell: SearchGridCell, parent_cell: SearchGridCell):
        # Nothing to do in this case
        pass
