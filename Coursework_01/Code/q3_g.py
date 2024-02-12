#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

from common.scenarios import *
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_iterator import ValueIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer
import time
from q3_e import plot_parameters, cost

def evaluate(type, n, theta, gamma, N, env, drawer_height):
    execution_time = 0
    total_steps = 0
    total_cost = 0

    policy_solver = None
    if type == "Value iteration":
        policy_solver = ValueIterator(env)
    else:
        policy_solver = PolicyIterator(env)

    for i in range(n):
        policy_solver.set_theta(theta)
        policy_solver.set_gamma(gamma)

        if type == "Value iteration":
            policy_solver.set_max_optimal_value_function_iterations(N)
        else:
            policy_solver.set_max_policy_evaluation_steps_per_iteration(N)

        policy_solver.initialize()

        value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
        policy_solver.set_value_function_drawer(value_function_drawer)

        policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
        policy_solver.set_policy_drawer(policy_drawer)

        start = time.time()
        _, pi = policy_solver.solve_policy()
        execution_time += time.time() - start
        policy_drawer.save_screenshot(f"plots/q3_g_optimal_{type.split()[0]}_iteration_policy.pdf")
        policy_drawer.save_screenshot(f"plots/q3_g_optimal_{type.split()[0]}_iteration_policy.jpg")
        value_function_drawer.save_screenshot(f"plots/q3_g_optimal_{type.split()[0]}_iteration_value.pdf")
        value_function_drawer.save_screenshot(f"plots/q3_g_optimal_{type.split()[0]}_iteration_value.jpg")
        if type == "Value iteration":
            total_steps += ValueIterator.total_steps
        else:
            total_steps += PolicyIterator.total_steps
        total_cost += cost(pi, p, airport_map)

    print(f"{type} benchmark:")
    print("Average execution time:", round(execution_time/n, 2), "s")
    print("Average steps:", round(total_steps/n, 2))
    print("Average cost:", round(total_cost/n, 2))

if __name__ == '__main__':
    
    # Get the map for the scenario
    #airport_map, drawer_height = three_row_scenario()
    airport_map, drawer_height = full_scenario()
    
    # Set up the environment for the robot driving around
    airport_environment = LowLevelEnvironment(airport_map)
    p = 0.8
    N = 500
    gamma = 0.99
    theta = 1e-3
    # Configure the process model
    airport_environment.set_nominal_direction_probability(p)
    
    # Q3i: Add code to evaluate value iteration down here.
    
    # policy_solver = ValueIterator(airport_environment)
    # params = {
    #     "thetas": [0.001, 0.01, 0.1],
    #     "gammas": [0.985, 0.99, 0.995, 1],
    #     "N": 3000,
    # }
    
    # plot_parameters(policy_solver, airport_map, params, "q3_g", drawer_height)

    
    # Run policy iteration and value iteration with optimal parameters
    evaluate("Policy iteration", 1, 10, 0.99, 500, airport_environment, drawer_height)
    evaluate("Value iteration", 1, 0.001, 1, 2000, airport_environment, drawer_height)