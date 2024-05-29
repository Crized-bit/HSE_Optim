from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from time import time
import scipy as sp
import datetime
from oracles import LassoOracle


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary 
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'Computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    msg = 'iterations_exceeded'
    oracle = LassoOracle(A, b, reg_coef, t_0)
    start_time = time()

    z_k = np.hstack((x_0, u_0))
    for t in range(max_iter):
        grad_k = oracle.grad(z_k)
        hess_k = oracle.hess(z_k)
        stop_val = tolerance_inner * np.linalg.norm(grad_k) ** 2
        for k in range(max_iter_inner):
            d_k = np.linalg.solve(hess_k, -grad_k)
            line_search_tool = get_line_search_tool({"c1": c1, "method": "Armijo",
                                                     "alpha_0": min(1.0, oracle.find_max_alpha(d_k=d_k, z_k=z_k))})
            alpha = line_search_tool.line_search(x_k=z_k, d_k=d_k, oracle=oracle)
            z_k += alpha * d_k

            grad_k = oracle.grad(z_k)
            hess_k = oracle.hess(z_k)

            if np.linalg.norm(grad_k) ** 2 <= stop_val:
                break

        gap = oracle.lasso_duality_gap(z_k[:len(z_k) // 2])

        if isinstance(history, defaultdict):
            history['func'].append(oracle.init_func(z_k))
            if z_k.size <= 2:
                history['x'].append(z_k)
            history['time'].append((time() - start_time))
            history['duality_gap'].append(gap)

        if gap < tolerance:
            msg = 'success'
            break
        oracle.t *= gamma
    return z_k, msg, history


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """

    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """

        def armijo(fun, const):
            alpha_iter = self.alpha_0

            if np.isnan(fun.grad_directional(x_k, d_k, alpha_iter)):
                return None

            if fun.func_directional(x_k, d_k, alpha_iter) <= \
                    fun.func_directional(x_k, 0, 0) + const * alpha_iter * fun.grad_directional(x_k, d_k, 0):
                return alpha_iter
            else:
                while fun.func_directional(x_k, d_k, alpha_iter) > \
                        fun.func_directional(x_k, 0, 0) + const * alpha_iter * fun.grad_directional(x_k, d_k, 0):
                    alpha_iter /= 2

            return alpha_iter

        if self._method == 'Wolfe':
            alpha = sp.optimize.line_search(f=oracle.func, myfprime=oracle.grad, xk=x_k, pk=d_k, c1=self.c1,
                                            c2=self.c2)
            if alpha[0] is None:
                return armijo(oracle, self.c1)
            else:
                return alpha[0]

        elif self._method == 'Armijo':
            return armijo(oracle, self.c1)

        elif self._method == 'Constant':
            return self.c


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()
