import numpy as np
from scipy.optimize import Bounds
import CLARSTA

# Objective function
def testfnc(x):
    return np.dot(x.T,x)

# Objective function residues
def testfnc_res(x):
    return x

# Define a box constraint lb <= x <= ub
def box_constraint(lb, ub):
    proj_C = [lambda x: np.minimum(np.maximum(x, lb), ub)]
    bounds = Bounds(lb, ub)
    return proj_C, bounds

# Example setup
n = 5

# Box constraint [1, 2]
lb = np.ones((n,)) * 1; ub = np.ones((n,)) * 2; proj_C, bounds = box_constraint(lb, ub)

# Set a feasible x0
x0 = proj_C[0](np.zeros((n,)))

np.random.seed(0)
sol = CLARSTA.solve(testfnc, x0, p = 2, prand = 1, model_type = "square of linear", resfuns = testfnc_res, resfun_num = n, proj_C=proj_C)
print(sol)