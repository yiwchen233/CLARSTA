import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint
import math
import CLARSTA
from CLARSTA.tools import dykstra

# Objective function
def chainrosenbrock(x):
    return sum(10 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

# Define a box constraint [lb, ub]
def box_constraint(lb, ub):
    proj_C = [lambda x: np.minimum(np.maximum(x, lb), ub)]
    bounds = Bounds(lb, ub)
    return proj_C, bounds

# Define a ball constraint centered at c with radius r
def ball_constraint(c, r):
    proj_C = [lambda x: c + r/max(np.linalg.norm(x-c), r) * (x-c)]
    constraints = NonlinearConstraint(lambda x: np.linalg.norm(x-c), -np.inf, r)
    return proj_C, constraints

# Define a constraint that is the intersection of a box and a ball
def ballbox_constraint(c, r, lb, ub):
    proj_C_ball, constraints = ball_constraint(c, r)
    proj_C_box, bounds = box_constraint(lb, ub)
    proj_C = [proj_C_ball[0], proj_C_box[0]]
    return proj_C, bounds, constraints

# Example setup
n = 5

# Intersection of ball constraint B(0,\sqrt(n)) and box constraint [-2, -1]
ball_c = np.zeros(n); ball_r = math.sqrt(n); lb = np.ones((n,)) * -2; ub = np.ones((n,)) * -1; proj_C, bounds, constraints = ballbox_constraint(ball_c, ball_r, lb, ub)

# Set a feasible x0
x0 = dykstra(proj_C, np.zeros((n,)))

np.random.seed(0)
sol = CLARSTA.solve(chainrosenbrock, x0, p = 2, prand = 1, model_type = "underdetermined quadratic", proj_C=proj_C)
print(sol)