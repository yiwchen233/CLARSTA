# CLARSTA -- Convex-constrained Linear Approximation Random Subspace Trust-region Algorithm
![GitHub](https://img.shields.io/badge/License-GPL%20v3-blue.svg)

This repository contains the source code for the Convex-constrained Linear Approximation Random Subspace Trust-region Algorithm (CLARSTA), introduced in our [paper](https://arxiv.org/abs/2506.20335).  

CLARSTA is a Python package designed for large-scale convex-constrained optimization problems where derivative information is unavailable. The feasible set is assumed to be convex, closed, and to have a nonempty interior. The only requirement is access to a projection operator onto the constraint set.

The algorithm makes no structural assumptions about the objective function and currently supports four types of surrogate models:
* determined quadratic interpolation model (using $\frac{(n+1)(n+2)}{2}$ sample points)  
* underdetermined quadratic interpolation model (using $2n+1$ sample points)  
* linear interpolation model (using $n+1$ sample points)  
* square of linear interpolation model (using $n+1$ sample points, can only be constructed when the objective function has the structure of sum-of-square)

For a detailed explanation, please see: Y. Chen, W. Hare, and A. Wiebe, CLARSTA: A random subspace trust-region algorithm for convex-constrained derivative-free optimization, https://arxiv.org/abs/2506.20335 (2025)


## Citation
If you use our code in your research, then please cite:
```
@misc{chen2025clarsta,
  title={{CLARSTA: A} random subspace trust-region algorithm for convex-constrained derivative-free optimization}, 
  author={Yiwen Chen and Warren Hare and Amy Wiebe},
  year={2025},
  eprint={2506.20335},
  archivePrefix={arXiv},
  primaryClass={math.OC},
  url={https://arxiv.org/abs/2506.20335}
}
```


## Requirements
CLARSTA requires Python 3.11.6 or above, with the following python packages:
```
NumPy >= 1.24.2
SciPy >= 1.10.1
```


## Installation & Updating
To install CLARSTA, please download from Github by either downloading the ZIP file or using the follwing command:
```sh
git clone https://github.com/yiwchen233/CLARSTA
```

To update to the latest version, please go to the top-level directory and do the following:
```sh
git pull
```


## Using CLARSTA
The API of CLARSTA is:
```sh
sol = CLARSTA.solve(obj, x0, p, prand, deltabeg, deltaend, maxfun, fmin_true, model_type, resfuns, resfun_num, proj_C)
```


### Inputs
```
obj         (required)  objective function
x0          (required)  starting point
p           (required)  full subspace dimension
prand       (required)  minimum randomized subspace dimension
deltabeg    (optional, default 0.1\max(\|x0\|_\infty, 1.0))  initial trust-region radius
deltaend    (optional, default 10^{-8})  minimum trust-region radius
maxfun      (optional, default 10^5)  maximum number of function evaluations
model_type  (optional, default "quadratic")  model construction technique (must be one of "quadratic", "underdetermined quadratic", "linear", or "square of linear")
resfuns     (required if model_type == "square of linear", default None)  residue functions
resfun_num  (required if model_type == "square of linear", default None)  number of residue functions
proj_C      (optional, default None) list of projection functions onto the constraint set C
```


### Output
A class that contains the results of CLARSTA and can be called by:
```
sol.x      minimizer obtained by CLARSTA
sol.f      minimum function value obtained by CLARSTA
sol.nf     number of function evaluations used
sol.niter  number of iterations used
```


### Examples
The files in the format of example_XXX_YYY.py are some examples of how to use CLARSTA, where XXX corresponds to the model construction technique, and YYY corresponds to the constraint set used in the example. 


## License 
All code in CLARSTA is released under the GNU GPL [license](/LICENSE).  


## Contact
Please contact us via email to report any issues:
```
yiwchen@student.ubc.ca
```
