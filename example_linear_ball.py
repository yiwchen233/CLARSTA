import numpy as np
from scipy.optimize import NonlinearConstraint
import math
import CLARSTA

# Objective function
def chainrosenbrock(x):
    return sum(10 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

# Define a ball constraint centered at c with radius r
def ball_constraint(c, r):
    proj_C = [lambda x: c + r/max(np.linalg.norm(x-c), r) * (x-c)]
    constraints = NonlinearConstraint(lambda x: np.linalg.norm(x-c), -np.inf, r)
    return proj_C, constraints

# Example setup
n = 5

# Ball constraint B(0,\sqrt(n))
ball_c = np.zeros(n); ball_r = math.sqrt(n); proj_C, constraints = ball_constraint(ball_c, ball_r)

# Set a feasible x0
x0 = proj_C[0](np.zeros((n,)))

np.random.seed(0)
sol = CLARSTA.solve(chainrosenbrock, x0, p = 2, prand = 1, model_type = "linear", proj_C=proj_C)
print(sol)