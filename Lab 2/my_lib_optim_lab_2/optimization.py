import collections
from datetime import datetime
import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from .utils import get_line_search_tool
from collections.abc import Callable


def conjugate_gradients(matvec: Callable, b: np.ndarray, x_0: np.ndarray, tolerance: float = 1e-4,
                        max_iter: int | None = None,
                        trace: bool = False, display: bool = False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.ndarray
        Vector b for the system.
    x_0 : 1-dimensional np.ndarray
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise, None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.ndarray
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of torch.Tensor's, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    stop_value = np.linalg.norm(b) * tolerance

    if max_iter is None:
        max_iter = x_0.shape[0]

    x_k = x_0.copy()
    r_k = b - matvec(x_0)
    d_k = r_k.copy()
    start_time = datetime.now()
    for _ in range(max_iter):
        if isinstance(history, defaultdict):
            history['residual_norm'].append(np.linalg.norm(r_k))
            history['x'].append(x_k)
            history['time'].append((datetime.now() - start_time).total_seconds())
        if np.linalg.norm(r_k) < stop_value:
            return x_k, 'success', history

        A_dk = matvec(d_k)
        alpha = (r_k.T @ r_k) / (d_k.T @ A_dk)
        x_k = x_k + alpha * d_k
        r_k_1 = r_k - alpha * A_dk
        betha = (r_k_1.T @ r_k_1) / (r_k.T @ r_k)

        r_k = r_k_1
        d_k = r_k + betha * d_k
    return x_k, 'iterations_exceeded', history


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradient descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)

    iter_num = 0
    x_k = np.copy(x_0)
    d_k = oracle.grad(x_k)
    initial_grad = d_k.copy()
    stop_val = tolerance * np.linalg.norm(initial_grad) ** 2
    message = 'iterations_exceeded'
    prev_alpha = None
    start_time = datetime.now()

    while iter_num < max_iter:

        if isinstance(history, defaultdict):
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(d_k))
            history['x'].append(x_k)
            history['time'].append((datetime.now() - start_time).total_seconds())

        if prev_alpha is not None:
            best_alpha = line_search_tool.line_search(oracle, x_k, -d_k, previous_alpha=prev_alpha)
        else:
            best_alpha = line_search_tool.line_search(oracle, x_k, -d_k)

        if best_alpha is None:
            message = 'computational_error'
            break

        prev_alpha = best_alpha
        x_k = x_k - best_alpha * d_k

        d_k = oracle.grad(x_k)

        if np.linalg.norm(d_k) ** 2 <= stop_val:
            message = 'success'

            if isinstance(history, defaultdict):
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(np.linalg.norm(np.linalg.norm(d_k)))
                history['x'].append(x_k)
                history['time'].append((datetime.now() - start_time).total_seconds())
            break

        iter_num += 1

    return x_k, message, history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    my_queue = deque(maxlen=memory_size)
    history = defaultdict(list) if trace else None

    stop_val = tolerance * np.linalg.norm(oracle.grad(x_0)) ** 2
    x_k = x_0.copy()
    line_search_tool = get_line_search_tool(line_search_options)

    def LBFGS_multiply(v: np.ndarray, curr_queue: collections.deque, gamma: float):
        if not curr_queue:
            return gamma * v
        s, y = curr_queue.pop()
        v_new = v - np.dot(s, v) / np.dot(y, s) * y
        z = LBFGS_multiply(v_new, curr_queue, gamma)
        return z + (np.dot(s, v) - np.dot(y, z)) / np.dot(y, s) * s

    def LBFGS_direction(x, curr_queue: collections.deque):
        if curr_queue:
            s, y = curr_queue[-1]
            gamma = np.dot(y, s) / np.dot(y, y)
        else:
            gamma = 1.0

        return LBFGS_multiply(-oracle.grad(x), curr_queue, gamma)

    start_time = datetime.now()
    for _ in range(max_iter):
        d_k = LBFGS_direction(x_k, my_queue.copy())
        grad_k = oracle.grad(x_k)
        best_alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1)

        if isinstance(history, defaultdict):
            history['func'].append(oracle.func(x_k))
            if x_k.size <= 2:
                history['x'].append(x_k)
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['grad_norm'].append(np.linalg.norm(grad_k))

        if np.linalg.norm(grad_k) ** 2 <= stop_val:
            return x_k, 'success', history

        x_k += best_alpha * d_k
        if memory_size > 0:
            if len(my_queue) >= memory_size:
                my_queue.popleft()

            my_queue.append((best_alpha * d_k, oracle.grad(x_k) - grad_k))

    return x_k, 'iterations_exceeded', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None

    stop_val = tolerance * np.linalg.norm(oracle.grad(x_0)) ** 2
    x_k = x_0.copy()
    line_search_tool = get_line_search_tool(line_search_options)

    start_time = datetime.now()
    for _ in range(max_iter):
        grad_k = oracle.grad(x_k)
        matvec_k = lambda x: np.squeeze(oracle.hess_vec(x_k, x))
        ehta_k = min(0.5, np.sqrt(np.linalg.norm(grad_k)))

        d_k, _, _ = conjugate_gradients(matvec_k, -grad_k, -grad_k, tolerance=ehta_k)
        while np.dot(d_k, grad_k) >= 0:
            ehta_k /= 10
            d_k, _, _ = conjugate_gradients(matvec_k, -grad_k, d_k, tolerance=ehta_k)

        if isinstance(history, defaultdict):
            history['func'].append(oracle.func(x_k))
            if x_k.size <= 2:
                history['x'].append(x_k)
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['grad_norm'].append(np.linalg.norm(grad_k))

        if np.linalg.norm(grad_k) ** 2 <= stop_val:
            return x_k, 'success', history

        best_alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1)
        x_k += best_alpha * d_k

    return x_k, 'iterations_exceeded', history
