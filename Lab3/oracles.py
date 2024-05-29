import numpy as np
import scipy as sp


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class LassoOracle(BaseSmoothOracle):
    def __init__(self, a, b, reg_coef, t):
        self.A = a
        self.b = b
        self.n_dim = self.A.shape[1]
        self.reg_coef = reg_coef

        self.Ax_b = lambda x: self.A @ x.reshape(-1, 1) - self.b
        self.ATAx_b = lambda x: self.A.T @ (self.A @ x.reshape(-1, 1) - self.b)
        self.t = t

        self.I_matrix = np.zeros((2 * self.n_dim, 2 * self.n_dim))
        for i in range(self.n_dim):
            self.I_matrix[i, i] = -1
            self.I_matrix[i, self.n_dim + i] = 1
            self.I_matrix[self.n_dim + i, i] = -1
            self.I_matrix[self.n_dim + i, self.n_dim + i] = -1

    def init_func(self, z):
        x, u = z[:z.shape[0] // 2], z[z.shape[0] // 2:]
        return 1 / 2 * np.linalg.norm(self.Ax_b(x)) ** 2 + self.reg_coef * np.dot(np.ones_like(u), u)

    def func(self, z):
        x, u = z[:z.shape[0] // 2], z[z.shape[0] // 2:]
        return self.t * self.init_func(z) - np.dot(np.ones(self.n_dim), (np.log(u + x) + np.log(u - x)))

    def grad(self, z):
        x, u = z[:z.shape[0] // 2], z[z.shape[0] // 2:]
        grad_x = self.t * self.ATAx_b(x).squeeze() - (1 / (u + x) - 1 / (u - x))
        grad_u = self.t * self.reg_coef * np.ones(self.n_dim) - (1 / (u + x) + 1 / (u - x))
        return np.concatenate((grad_x, grad_u))

    def hess(self, z):
        x, u = z[:z.shape[0] // 2], z[z.shape[0] // 2:]
        hess_xx = self.t * self.A.T @ self.A + np.diag(1 / (u + x) ** 2 + 1 / (u - x) ** 2)
        hess_uu = np.diag(1 / (u + x) ** 2 + 1 / (u - x) ** 2)
        hess_ux = np.diag(1 / (u + x) ** 2 - 1 / (u - x) ** 2)

        top, bottom = np.hstack((hess_xx, hess_ux)), np.hstack((hess_ux, hess_uu))

        return np.vstack((top, bottom))

    def lasso_duality_gap(self, x):
        """
        Estimates f(x) - f* via duality gap for
            f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        """
        f = 0.5 * np.linalg.norm(self.Ax_b(x)) ** 2 + self.reg_coef * np.linalg.norm(x, ord=1)
        mu = min(1, self.reg_coef / np.linalg.norm(self.ATAx_b(x), ord=np.inf)) * self.Ax_b(x)
        f_star = 1 / 2 * np.linalg.norm(mu) ** 2 + np.dot(self.b.squeeze(), mu.squeeze())
        return f + f_star

    def find_max_alpha(self, d_k, z_k, theta=0.99) -> float:
        binary_check = (d_k.reshape(1, -1) @ self.I_matrix).squeeze() > 0
        matrix = self.I_matrix[:, binary_check]
        if matrix.size == 0:
            return 1
        return theta * np.min(-(z_k.reshape(1, -1) @ matrix) / (d_k.reshape(1, -1) @ matrix))

