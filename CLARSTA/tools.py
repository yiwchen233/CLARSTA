import numpy as np
import math

__all__ = ['sumsq', 'eval_objective', 'gen_random_directions', 'dykstra', 'pball', 'pbox', 'check_feasibility']


def sumsq(x):
    return np.dot(x, x)

# Evaluate objective function at x
def eval_objective(objfun, x):
    fval = objfun(x)
    return fval

# Generate random sample directions
def gen_random_directions(n, num_pts, delta, prand, Q=None):
    if Q is not None:
        p = Q.shape[1]
        assert Q.shape == (n, p), "Q must have n rows"
    else:
        p = 0
    assert delta > 0, "delta must be strictly positive"
    assert num_pts > 0, "num_pts must be strictly positive"
    assert num_pts <= n - p, "num_pts must be <= n-p (p=number of columns of Q)"

    results = np.zeros((num_pts, n))  # save space for results

    A = np.random.normal(size=(n, num_pts)) / math.sqrt(prand)
    if Q is not None:
        A = A - np.dot(Q, np.dot(Q.T, A))  # make orthogonal to columns of Q
    A_Q, A_R = np.linalg.qr(A, mode='reduced')  # make directions orthonormal

    # Construct transfomation matrix to make the results satisfy Haar distribution
    A_R_diag = A_R.diagonal()
    Mtrans = np.diag(A_R_diag/np.absolute(A_R_diag))
    A_Q = np.dot(A_Q, Mtrans)

    # The results are the columns of A_Q * delta
    for i in range(num_pts):
        results[i, :] = delta * A_Q[:, i]

    return results

# Dykstra's algorithm for computing the projection of x0 into the intersection of several convex sets, each with its own projection operator.
def dykstra(P, x0, max_iter=100, tol=1e-10):
    x = x0.copy()
    p = len(P)
    y = np.zeros((p, x0.shape[0]))

    n = 0
    cI = float('inf')
    while n < max_iter and cI >= tol:
        cI = 0.0
        for i in range(p):
            # Update iterate
            prev_x = x.copy()
            x = P[i](prev_x - y[i,:])

            # Update increment
            prev_y = y[i, :].copy()
            y[i, :] = x - (prev_x - prev_y)

            # Stop condition
            cI += np.linalg.norm(prev_y - y[i, :])**2

        n += 1

    return x

# Projection operator for a Euclidean ball with center c and radius r
def pball(x, c, r):
    return c + r/max(np.linalg.norm(x-c), r) * (x-c)

# Projection operator for box constraints, l <= x <= u
def pbox(x, l, u):
    return np.minimum(np.maximum(x, l), u)

# Check if x0 is feasible, return True of False
def check_feasibility(proj_C, x0, tol=1e-10):
    if proj_C is None:
        return True
    else:
        x = dykstra(proj_C, x0)
        if np.linalg.norm(x-x0) <= tol:
            return True
        else:
            return False