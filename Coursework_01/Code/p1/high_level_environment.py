'''
Created on 25 Jan 2022

@author: ucacsjj
'''

from enum import Enum

import gymnasium
import time

# Import the planners
from grid_search.a_star_planner import AStarPlanner
from grid_search.breadth_first_planner import BreadthFirstPlanner
from grid_search.depth_first_planner import DepthFirstPlanner
from grid_search.dijkstra_planner import DijkstraPlanner

from .high_level_actions import HighLevelActionType


class PlannerType(Enum):
    BREADTH_FIRST = 0
    DEPTH_FIRST = 1
    DIJKSTRA = 2
    A_STAR = 3


class HighLevelEnvironment(gymnasium.Env):
    '''
    classdocs
    '''

    def __init__(self, airport_map, planner_type = PlannerType.DEPTH_FIRST):
        '''
        Constructor
        '''
        
        # Store the map
        self._airport_map = airport_map

        # store planner type
        self.queue_type = ""
        if (planner_type == PlannerType.DEPTH_FIRST):
                self.queue_type = "lifoQueue"
        if (planner_type == PlannerType.BREADTH_FIRST):
                self.queue_type = "fifoQueue"
        self.max_nodes_stored = 0
        
        # Create the planner which will be used to simulate the robot's travel
        planner_factory = {
            PlannerType.BREADTH_FIRST : BreadthFirstPlanner(self._airport_map),
            PlannerType.DEPTH_FIRST : DepthFirstPlanner(self._airport_map),
            PlannerType.DIJKSTRA : DijkstraPlanner(self._airport_map),
            PlannerType.A_STAR : AStarPlanner(self._airport_map),
            }
        self._planner = planner_factory.get(planner_type)

        # Disable the graphics by default; this can be enabled again
        self._planner.show_graphics(False)
                
        self._current_coords = None

    def reset(self):
        self._current_coords = None
        return self._current_coords
        
    def planner(self):
        return self._planner

    def search_grid_drawer(self):
        return self._planner.search_grid_drawer()

    def show_graphics(self, graphics):
        self._planner.show_graphics(graphics)

        if graphics is False:
            self._planner.update_graphics_each_iteration(graphics)
    
    def show_verbose_graphics(self, verbose_graphics):
        self._planner.update_graphics_each_iteration(verbose_graphics)

    def step(self, action):        
        # If the action is to teleport the robot to a new location,
        # do so instantly at no cost. If the robot  can't be 
        # transported to the new cell, return a reward of -infinity
        # and leave the robot as-is.

        if action[0] == HighLevelActionType.TELEPORT_ROBOT_TO_NEW_POSITION:
            new_coords = action[1]
            if self._airport_map.is_obstruction(new_coords[0], new_coords[1]):
                return self._current_coords, -float("inf"), False, False
            else:
                self._current_coords = action[1]
                return self._current_coords, 0, False, True
        
        # If the action is to plan a path to the new goal, fire up
        # our planner and get the path. Here we only care about the path cost.
        # If the path can be reached we return the negative of the path cost
        # (because we want to maximise reward and minimize the path length).
        # If the goal can't be reached, the reward is minus infinity
        if action[0] == HighLevelActionType.DRIVE_ROBOT_TO_NEW_POSITION:
            startTime = time.time()
            goal_coords = action[1]
            self._planner.plan(self._current_coords, goal_coords)
            plan = self._planner.extract_path_to_goal()
            endTime = time.time()

            duration = endTime - startTime

            print(f'(1) plan.path_travel_cost={plan.path_travel_cost}')
            print(f'plan.goal_reached={plan.goal_reached}')
            print(f'(2) The total number of nodes searched: {getattr(self._planner, "searchCount")}')
            print(f'(3) The maximum number of nodes stored in the queue at any point: {getattr(self._planner, "maxNodesStored")}')
            print(f'(4) The total number of nodes stored in the queue: {getattr(self._planner, "totalNodesStored")}')
            print(f'(5) The average number of nodes stored in the queue: {getattr(self._planner, "avgNodesStored")}')
            print(f'(6) The execution time taken to find the path: {duration}')

            self._planner.reset_statistics()

            if plan.goal_reached is True:
                self._current_coords = goal_coords
                return self._current_coords, -plan.path_travel_cost, False, plan
            else:
                return self._current_coords, -float("inf"), False, plan
            
