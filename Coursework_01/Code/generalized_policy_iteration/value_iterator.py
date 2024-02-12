'''
Created on 29 Jan 2022

@author: ucacsjj
'''

from .dynamic_programming_base import DynamicProgrammingBase
from p2.low_level_actions import LowLevelActionType


# This class ipmlements the value iteration algorithm

class ValueIterator(DynamicProgrammingBase):
    total_steps = 0

    def __init__(self, environment):
        DynamicProgrammingBase.__init__(self, environment)
        
        # The maximum number of times the value iteration
        # algorithm is carried out is carried out.
        self._max_optimal_value_function_iterations = 2000
   
    # Method to change the maximum number of iterations
    def set_max_optimal_value_function_iterations(self, max_optimal_value_function_iterations):
        self._max_optimal_value_function_iterations = max_optimal_value_function_iterations

    #    
    def solve_policy(self):
        ValueIterator.total_steps = 0
        # Initialize the drawers
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        self._compute_optimal_value_function()
 
        self._extract_policy()
        
        # Draw one last time to clear any transients which might
        # draw changes
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        return self._v, self._pi

    # Q3f:
    # Finish the implementation of the methods below.
    
    def _compute_optimal_value_function(self):

        # This method returns no value.
        # The method updates self._pi

        environment = self._environment
        map = environment.map()
        print("theta:", self._theta)

        iteration = 0

        while True:
            
            delta = 0

            # Sweep systematically over all the states            
            for x in range(map.width()):
                for y in range(map.height()):
                    if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                        continue

                    old_v = self._v.value(x, y)
                    best_v = -float('inf')
                    
                    for action_num in range(LowLevelActionType.NUMBER_OF_ACTIONS-2):
                        action = LowLevelActionType(action_num)
                        s_prime, r, p = environment.next_state_and_reward_distribution((x, y), action)

                        # Sum over the rewards
                        new_v = 0
                        for t in range(len(p)):
                            sc = s_prime[t].coords()
                            new_v = new_v + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))

                        best_v = max(best_v, new_v)

                    self._v.set_value(x, y, best_v)

                    delta = max(delta, abs(old_v-best_v))

            # print(delta, iteration)
            if delta < self._theta:
                break

            iteration += 1
            ValueIterator.total_steps += 1

            if iteration >= self._max_optimal_value_function_iterations:
                print('Maximum number of iterations exceeded')
                break
                    


    def _extract_policy(self):

        # This method returns no value.
        # The policy is in self._pi

        environment = self._environment
        map = environment.map()
            
        for x in range(map.width()):
            for y in range(map.height()):
                if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                    continue

                best_action = None
                best_v = -float('inf')
                
                for action_num in range(LowLevelActionType.NUMBER_OF_ACTIONS):
                    action = LowLevelActionType(action_num)
                    s_prime, r, p = environment.next_state_and_reward_distribution((x, y), action)

                    # Sum over the rewards
                    new_v = 0
                    for t in range(len(p)):
                        sc = s_prime[t].coords()
                        new_v = new_v + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))

                    if best_v < new_v:
                        best_v = new_v
                        best_action = action

                self._pi.set_action(x, y, best_action)