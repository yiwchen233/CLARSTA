import numpy as np
from scipy.optimize import LinearConstraint
import CLARSTA

# Objective function
def chainrosenbrock(x):
    return sum(10 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

# Define a half-space constraint h^Tx <= c
def halfspace_constraint(h, c):
    proj_C = [lambda x: x-max(np.dot(h.T,x)-c,0)/np.linalg.norm(h)**2 * h]
    constraints = LinearConstraint(h.T, -np.inf, c)
    return proj_C, constraints

# Example setup
n = 5

# Halfspace constraint -1^\top x <= 0
hs_h = np.ones(n) * -1; hs_c = 0; proj_C, constraints = halfspace_constraint(hs_h, hs_c)

# Set a feasible x0
x0 = proj_C[0](np.zeros((n,)))

np.random.seed(0)
sol = CLARSTA.solve(chainrosenbrock, x0, p = 2, prand = 1, model_type = "quadratic", proj_C=proj_C)
print(sol)