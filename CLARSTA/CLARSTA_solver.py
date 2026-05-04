"""
CLARSTA: Convex-constrained Linear Approximation Random Subspace Trust-region Algorithm
Given a blackbox objective function f defined over an n-dimensional variable x,
and a convex, closed set C with non-empty interior,
this code solves the convex-constrained blackbox optimization problem
min_C f(x).
"""

import numpy as np

from .CLARSTA_exit_info import *
from .CLARSTA_model import *
from .CLARSTA_parameters import ParameterList
from .trust_region import ctrsbox
from .tools import eval_objective, gen_random_directions, dykstra, check_feasibility


__all__ = ['solve']


# Approximatly calculate pi^m(xk) by finding projected gradient path
def cal_pim(proj_C, xk, gk, Q, gtol=1e-12):
    n = len(xk)
    norm_gk = np.linalg.norm(gk)
    dk = np.zeros((n,))
    if norm_gk > gtol:
        g_full = Q.dot(gk)
        if proj_C is None or len(proj_C) == 0:
            dk = -g_full/norm_gk
        else:
            dk = dykstra(proj_C, xk - g_full/norm_gk) - xk
        return abs(np.dot(g_full.T, dk))
    return 0

# Trust-region radius update procedure
def update_tr(delta, ratio, params):
    if ratio < params("tr_radius.eta1"):  # ratio < 0.1
        delta = params("tr_radius.gamma_dec") * delta
    elif ratio <= params("tr_radius.eta2"):  # 0.1 <= ratio <= 0.7
        delta = delta  
    else:  # ratio >= 0.7
        delta = min(params("tr_radius.gamma_inc") * delta, params("tr_radius.delta_max"))

    return delta

# Construct sample set
def construct_sample_set(model, p, delta, objfun, nf, maxfun, params, fmin_true):
    exit_info = None

    if model.p > p:
        # If we somehow have more points, remove until correct
        while model.p > p:
            k = np.argmax(model.distances_to_xiter())
            model.remove_point(k)

    if model.p < p:
        # Generate new directions orthogonal to current directions
        model.factorise_system()
        dirns = gen_random_directions(model.n, p-model.p, delta, model.prand, Q=model.Q)
        
        for i in range(dirns.shape[0]):
            d = dirns[i, :]
            xnew = model.xiter() + d

            # Evaluate objective at xnew
            nf += 1
            fnew = eval_objective(objfun, xnew)
            fea_flag = check_feasibility(model.proj_C, xnew)

            if fmin_true is not None and fnew <= fmin_true + params("model.rel_tol") * (model.fbeg - fmin_true) and fea_flag == True:
                model.save_point(xnew, fnew)
                exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
                break  # quit

            if nf >= maxfun:
                if fea_flag == True:
                    model.save_point(xnew, fnew) 
                exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                break  # quit

            # Append xnew to model
            model.append_point(xnew, fnew, fea_flag)

    return exit_info, model, nf


def solve_main(objfun, x0, deltabeg, deltaend, maxfun, params, p, prand, fmin_true=None, model_type="linear", resfuns=None, resfun_num=None, proj_C=None, gtol=1e-12):
    exit_info = None
    
    # Start with evaluating f(x0)
    nf = 1
    f0 = eval_objective(objfun, x0)
    
    if fmin_true is not None and f0 <= fmin_true + params("model.rel_tol") * (f0 - fmin_true):
        # If f(x0) is already good enough, declare success and exit
        exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
        return x0, f0, nf, 0, exit_info

    # Initialize model
    delta = deltabeg
    model = SampleSet(p, prand, x0, f0, proj_C, rel_tol=params("model.rel_tol"))
    exit_info, nf = model.initialise_sample_set(delta, objfun, nf, maxfun, fmin_true)

    if exit_info is not None:
        xiter, fiter = model.get_final_results()
        return xiter, fiter, nf, 0, exit_info

    # Start iterating
    current_iter = 0
    while True:
        current_iter += 1
        # Construct sample set
        exit_info, model, nf = construct_sample_set(model, p, delta, objfun, nf, maxfun, params, fmin_true)
        if exit_info is not None:
            break  # quit

        # Build model according to the specified model type
        if model_type == "quadratic":
            interp_ok, ck, gk, Hk, exit_info, nf = model.interpolate_quadratic_model(objfun, nf, maxfun, fmin_true)
        elif model_type == "underdetermined quadratic":
            interp_ok, ck, gk, Hk, exit_info, nf  = model.interpolate_underdetermined_quadratic_model(objfun, nf, maxfun, fmin_true)
        elif model_type == "linear":
            interp_ok, ck, gk, Hk, exit_info = model.interpolate_linear_model()
        elif model_type == "square of linear":
            interp_ok, ck, gk, Hk, exit_info = model.interpolate_linear_model_square(resfuns, resfun_num)
        xk = model.xiter()
        fk = model.fiter()

        if exit_info is not None:
            break
        if not interp_ok:
            exit_info = ExitInformation(EXIT_LINALG_ERROR, "Failed to build interpolation model")
            break  # quit

        # Criticality step
        pimk = cal_pim(proj_C, xk, gk, model.Q)
        if params("general.criticality_step_mu") * pimk < delta:
            delta = params("tr_radius.gamma_dec") * delta
            if delta <= deltaend:
                exit_info = ExitInformation(EXIT_SUCCESS, "delta has reached deltaend")
                break  # quit

            # Shrink all directions
            dirns = model.directions_from_xiter()
            for i in range(dirns.shape[0]):
                d = dirns[i, :]
                xnew = model.xiter() + params("tr_radius.gamma_dec") * d

                # Evaluate objective at xnew
                nf += 1
                fnew = eval_objective(objfun, xnew)
                fea_flag = check_feasibility(model.proj_C, xnew)

                if fmin_true is not None and fnew <= fmin_true + params("model.rel_tol") * (model.fbeg - fmin_true) and fea_flag == True:
                    model.save_point(xnew, fnew)
                    exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
                    break  # quit

                if nf >= maxfun:
                    if fea_flag == True:
                        model.save_point(xnew, fnew) 
                    exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                    break  # quit

                if i < model.kiter:
                    model.points[i, :] = xnew.copy()
                else:
                    model.points[i + 1, :] = xnew.copy()

            continue

        # Calculate tentative step
        pred_reduction = 0
        xnew = xk.copy()
        if np.linalg.norm(gk) > gtol: # only do trust-region subproblem when model gradient is not ~0
            proj_C_red = []
            if not (proj_C is None or len(proj_C) == 0):
                for proj_idx in range(len(proj_C)):
                    proj_fnc = proj_C[proj_idx]
                    proj_C_red.append(lambda x: np.dot(model.Q.T, proj_fnc(xk + np.dot(model.Q, x)) - xk))
            sk_red, _, _ = ctrsbox(np.zeros((model.p,)), np.zeros((model.p,)), gk, Hk, -1e20 * np.ones((model.p,)), 1e20 * np.ones((model.p,)), proj_C_red, delta)
            pred_reduction =  -(np.dot(gk, sk_red) + 0.5 * np.dot(sk_red, Hk.dot(sk_red)))
            
            sk_full = model.Q.dot(sk_red)
            if proj_C is None or len(proj_C) == 0:
                xnew = xk + sk_full 
            else:
                xnew = dykstra(proj_C, xk + sk_full)

        # Evaluate objective at xnew
        nf += 1
        fnew = eval_objective(objfun, xnew)
        fea_flag = check_feasibility(model.proj_C, xnew)

        if fmin_true is not None and fnew <= fmin_true + params("model.rel_tol") * (model.fbeg - fmin_true) and fea_flag == True:
            model.save_point(xnew, fnew)
            exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
            break  # quit

        if nf >= maxfun:
            if fea_flag == True:
                model.save_point(xnew, fnew)
            exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
            break  # quit

        # Decide on type of step
        actual_reduction = fk - fnew
        if abs(pred_reduction) > gtol:
            ratio = actual_reduction / pred_reduction
        else:
            ratio = actual_reduction / gtol

        # Update trust region radius
        delta = update_tr(delta, ratio, params)

        if delta <= deltaend:
            exit_info = ExitInformation(EXIT_SUCCESS, "delta has reached deltaend")
            break  # quit

        # Add xnew to sample set
        if model.p < model.n:
            model.append_point(xnew, fnew, fea_flag)
            xnew_appended = True
        else:
            # If the model is full, replace the worst point with xnew
            try:
                sigmas = model.sigmamin_corresponding_to_each_point()
                sqdists = np.square(model.distances_to_xiter())  # ||yt-xk||^2
                vals = sigmas * np.maximum(sqdists ** 2 / delta ** 4, 1)  # BOBYQA point to remove criterion
                vals[model.kiter] = -1.0  # make sure kiter is never selected
                knew = np.argmax(vals)
            except np.linalg.LinAlgError:
                # If poisedness calculation fails, revert to dropping furthest points
                sqdists = np.square(model.distances_to_xiter())  # ||yt-xk||^2
                knew = np.argmax(sqdists)
            model.change_point(knew, xnew, fnew, fea_flag)  # updates xiter
            xnew_appended = False

        # Update kiter (xiter)
        model.update_kiter(model_type)

        # Remove at least 1 direction (if xnew appended) and prand to make space for new directions
        min_npt_to_drop = model.prand + (1 if xnew_appended else 0)
        ndirs_to_keep = max(0, model.p - min_npt_to_drop)
        ndirs_to_drop = model.p - ndirs_to_keep

        # Criteria of directions to remove:
        for i in range(ndirs_to_drop):
            try:
                sigmas = model.sigmamin_corresponding_to_each_point()
                sqdists = np.square(model.distances_to_xiter())  # ||yt-xk||^2
                vals = sigmas * np.maximum(sqdists**2 / delta**4, 1)  # BOBYQA point to remove criterion
                vals[model.kiter] = -1.0  # make sure kiter is never selected
            except np.linalg.LinAlgError:
                # If poisedness calculation fails, revert to dropping furthest points
                vals = np.square(model.distances_to_xiter())  # ||yt-xk||^2
                vals[model.kiter] = -1.0  # make sure kiter is never selected
                
            k = np.argmax(vals)
            vals = np.delete(vals, k)  # keep vals indices in line with indices of model.points
            model.remove_point(k)

        # Geometry management
        if model.p > 1:
            dists = model.distances_to_xiter()
            while model.p > 1 and np.max(dists) > params("geometry.sample_set_radius_tol") * delta:
                k = np.argmax(dists)
                dists = np.delete(dists, k)
                model.remove_point(k)
        if model.p > 1:
            dirns = model.directions_from_xiter()
            current_norm = np.linalg.norm(dirns, ord=2)
            while model.p > 1 and current_norm > 1 / params("geometry.tol") and (model.points.shape)[0] > 1:
                sigmas = model.sigmamin_corresponding_to_each_point()
                sqdists = np.square(model.distances_to_xiter())  # ||yt-xk||^2
                vals = sigmas * np.maximum(sqdists**2 / delta**4, 1)  # BOBYQA point to remove criterion
                vals[model.kiter] = -1.0  # make sure kiter is never selected

                k = np.argmax(vals)
                vals = np.delete(vals, k)  # keep vals indices in line with indices of model.points
                model.remove_point(k)

                dirns = model.directions_from_xiter()
                current_norm = np.linalg.norm(dirns, ord=2)

        continue

    xiter, fiter = model.get_final_results()
    
    return xiter, fiter, nf, current_iter, exit_info


def solve(objfun, x0, p, prand, deltabeg=None, deltaend=1e-8, maxfun=None, fmin_true=None, model_type="linear", resfuns=None, resfun_num=None, proj_C=None):

    n = len(x0)
    assert model_type == "quadratic" or model_type == "underdetermined quadratic" or model_type == "linear" or model_type == "square of linear", "Model type must be quadratic/underdetermined quadratic/linear/square of linear"
    if model_type == "square of linear":
        assert resfuns is not None and resfun_num is not None, "For square of linear models, resfuns and resfun_num must be specified"
    assert 1 <= p <= n, "p must be in [1..n]"
    assert 1 <= prand <= p, "prand must be in [1..p]"

    if deltabeg is None:
        deltabeg = 0.1 * max(np.max(np.abs(x0)), 1.0)
    if maxfun is None:
        maxfun = 1e5

    # Set parameters
    params = ParameterList(n)

    exit_info = None
    # Input & parameter checks
    if exit_info is None and deltabeg < 0.0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "deltabeg must be strictly positive")
    if exit_info is None and deltaend < 0.0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "deltaend must be strictly positive")
    if exit_info is None and deltabeg <= deltaend:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "deltabeg must be > deltaend")
    if exit_info is None and maxfun <= 0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "maxfun must be strictly positive")
    if exit_info is None and np.shape(x0) != (n,):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "x0 must be a vector")
    if exit_info is None and check_feasibility(proj_C, x0) == False:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "x0 must be feasible")

    # Check invalid parameter values
    all_ok, bad_keys = params.check_all_params()
    if exit_info is None and not all_ok:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "Bad parameters: %s" % str(bad_keys))

    # If we had an input error, quit gracefully
    if exit_info is not None:
        exit_flag = exit_info.flag
        exit_msg = exit_info.message(with_stem=True)
        results = OptimResults(None, None, 0, 0, exit_flag, exit_msg)
        return results

    # Call main solver
    xmin, fmin, nf, niter, exit_info = solve_main(objfun, x0, deltabeg, deltaend, maxfun, params, p, prand, fmin_true=fmin_true, model_type=model_type, resfuns=resfuns, resfun_num=resfun_num, proj_C=proj_C)

    # Process final return values & package up
    exit_flag = exit_info.flag
    exit_msg = exit_info.message(with_stem=True)

    results = OptimResults(xmin, fmin, nf, niter, exit_flag, exit_msg)

    return results
