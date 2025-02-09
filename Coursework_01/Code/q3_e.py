#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

from common.scenarios import full_scenario
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer
import time
import matplotlib.pyplot as plt
import copy
import random
from common.airport_map import MapCellType
from p2.low_level_actions import LowLevelActionType
from generalized_policy_iteration.value_iterator import ValueIterator


driving_deltas=[
            (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

# get new coordinates by applying action a to state (x, y)
def get_new_coords(x, y, a):
    if a in [LowLevelActionType.TERMINATE, LowLevelActionType.NONE]:
        return x, y
    
    d = driving_deltas[a]
    return x + d[0], y + d[1]

# Calculates the average cost of policy. This is done by starting the robot at each open
# cell and moving it according to given policy until it reaches the goal. If the robot gets
# stuck, 0 is returned and it means that the policy is invalid. Otherwise, average cost
# to reach the goal is computed and the experiment is repeated n times for more accurate results.
def cost(pi, p, map):
    q = (1-p)/2
    # There is a probability be to go to nominal direction, and q to
    # go to either side to nominal direction. This is represented by
    # 2 thresholds t1 and t2
    t1 = p
    t2 = p + q
    add = 0

    total_cost = 0
    # Number of experiments to run
    n = 10
    for i in range(n):
        cost = 0
        cell_count = 0
        for x in range(map.width()):
            for y in range(map.height()):
                if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                    continue
                cell_count += 1
                current_cost = 0
                while True:
                    if current_cost < -10000:
                        # The robot is stuck in a loop, is trying to step outside of the grid
                        # or hit the obstructions
                        print("robot cannot reach the target")
                        return 0
                    a = pi.action(x, y)

                    rand = random.random()
                    # If action is movement, then apply probability. If action is none or
                    # terminate, probability is not applied 
                    if a < 8:
                        if rand < t1:
                            add = 0
                        elif rand < t2:
                            add = 1
                        else:
                            add = -1
                    else:
                        add = 0

                    a = a + add
                    if a == -1:
                        a = 7
                    elif a == 8:
                        a = 0
                    

                    new_x, new_y = get_new_coords(x, y, a)

                    # If the action would take the robot off the edge of the map,
                    # they can't move and so their position remains the same.
                    # There is a nominal cost associated with being stationary
                    if (new_x < 0) or (new_x >= map.width()) \
                        or (new_y < 0) or (new_y >= map.height()):
                        current_cost += -1
                        continue

                    new_cell = map.cell(new_x, new_y)
                    current_cell = map.cell(x, y)
                    if new_cell.is_obstruction():
                        if new_cell.cell_type() is MapCellType.BAGGAGE_CLAIM:
                            current_cost += -10
                        else:
                            current_cost += -1
                        continue
                    else:
                        current_cost += map.compute_transition_cost(current_cell.coords(), new_cell.coords())

                    x = new_x
                    y = new_y
                    if new_cell.cell_type() is MapCellType.ROBOT_END_STATION:
                        break
                cost += current_cost
        total_cost += cost/cell_count
    return total_cost / n
    
# Plots the average cost, execution time, and number of steps for the policy generated by
# each combination of parameters.
def plot_parameters(policy_solver, airport_map, benchmark_params, question):
    p = 0.8
    costs = []
    times = []
    params = []
    total_steps = []

    invalid_params = []
    counter = 1
    total_experiments = len(benchmark_params['thetas']) * len(benchmark_params['gammas'])
    for theta in benchmark_params['thetas']:
        for gamma in benchmark_params['gammas']:
            print(f"Experiment {counter}/{total_experiments}")
            counter += 1
            if gamma == 1 and theta < 10:
                if isinstance(policy_solver, PolicyIterator):
                    # Delta does not converge for policy iterator
                    continue
            policy_solver.set_theta(theta)
            policy_solver.set_gamma(gamma)
            if isinstance(policy_solver, PolicyIterator):
                policy_solver.set_max_policy_evaluation_steps_per_iteration(benchmark_params['N'])
            else:
                policy_solver.set_max_optimal_value_function_iterations(benchmark_params['N'])

            policy_solver.initialize()

            start = time.time()

            _, pi = policy_solver.solve_policy()
            c = cost(pi, p, airport_map)
            if c != 0:
                params.append(f"θ = {theta}\n γ = {gamma}")
                costs.append(c)
                times.append(time.time() - start)
                if isinstance(policy_solver, PolicyIterator):
                    total_steps.append(PolicyIterator.total_steps)
                else:
                    total_steps.append(ValueIterator.total_steps)
            else:
                invalid_params.append(f"θ = {theta}, γ = {gamma}")


    plt.rcParams["figure.figsize"] = (14,9)

    plt.plot(range(len(params)), costs, marker='s')
    plt.xticks(range(len(params)), params)
    plt.xlabel("Parameters")
    plt.ylabel("Average cost")

    plt.savefig(f"plots/{question}_costs.pdf")
    plt.savefig(f"plots/{question}_costs.jpg")

    plt.cla()

    plt.plot(range(len(params)), times, marker='s')
    plt.xticks(range(len(params)), params)
    plt.xlabel("Parameters")
    plt.ylabel("Execution Time (s)")
    plt.savefig(f"plots/{question}_times.pdf")
    plt.savefig(f"plots/{question}_times.jpg")

    plt.cla()

    plt.plot(range(len(params)), total_steps, marker='s')
    plt.xticks(range(len(params)), params)
    plt.xlabel("Parameters")
    plt.ylabel("Total Steps")
    plt.savefig(f"plots/{question}_steps.jpg")
    plt.savefig(f"plots/{question}_steps.pdf")

    if invalid_params:
        print("Robot could not reach the target with these params:")
        for p in invalid_params:
            print(p)

if __name__ == '__main__':
    
    # Get the map for the scenario
    #airport_map, drawer_height = three_row_scenario()
    airport_map, drawer_height = full_scenario()
    
    # Set up the environment for the robot driving around
    airport_environment = LowLevelEnvironment(airport_map)
    
    # Configure the process model
    p = 0.8
    airport_environment.set_nominal_direction_probability(p)

    # Create the policy iterator
    policy_solver = PolicyIterator(airport_environment)
    

    # Q3e:
    # Investigate different parameters
    params = {
        "thetas": [1e-6, 1e-3, 1, 11],
        "gammas": [0.985, 0.99, 0.995, 1],
        "N": 500,
    }
    plot_parameters(policy_solver, airport_map, params, "q3_e")        