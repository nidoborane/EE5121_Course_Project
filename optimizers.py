import numpy as np

class RACDM:
    def __init__(self, p_grad, x0, alpha, L0, eps=1e-6, max_iters=10000):
        self.p_grad = p_grad
        self.x0 = x0.astype(float).copy()
        self.alpha = alpha
        self.L = L0.copy()
        self.probs = self.L ** alpha / np.sum(self.L ** alpha)
        self.eps = eps
        self.max_iters = max_iters

    def update(self, x):
        i = np.random.choice(len(self.probs), p=self.probs)
        grad_i = self.p_grad(x, i)
        L_hat_i = self.L[i]
        x_new = x.copy()
        x_new[i] -= grad_i / L_hat_i
        grad_i_new = self.p_grad(x_new, i)
        while grad_i * grad_i_new < 0:
            L_hat_i *= 2
            x_new = x.copy()
            x_new[i] -= grad_i / L_hat_i
            grad_i_new = self.p_grad(x_new, i)
        self.L[i] = L_hat_i / 2
        self.probs = self.L ** self.alpha / np.sum(self.L ** self.alpha)
        return x_new

    def solve(self):
        x = self.x0.copy()
        for k in range(self.max_iters):
            x = self.update(x)
            # if (k + 1) % 100 == 0:
            #     full_grad = np.array([self.p_grad(x, i) for i in range(len(x))])
            #     if np.linalg.norm(full_grad) < self.eps:
            #         break
        return x

class ACDM:
    def __init__(self):
        pass

class GradientDescent:
    def __init__(self, p_grad, x0, eta=1e-3, eps=1e-6, max_iters=10000):
        self.p_grad = p_grad
        self.x0 = x0.astype(float).copy()
        self.eta = eta
        self.eps = eps
        self.max_iters = max_iters

    def update(self, x):
        x_prev = x.copy()
        for i in range(x.shape[0]):
            x[i] -= self.eta * self.p_grad(x_prev, i)
        return x

    def solve(self):
        x = self.x0.copy()
        for k in range(self.max_iters):
            x = self.update(x)
            #if (k + 1) % 100 == 0:
            #    full_grad = np.array([self.p_grad(x, i) for i in range(len(x))])
            #    if np.linalg.norm(full_grad) < self.eps:
            #        break
        return x    

# Q = np.array([
#     [10.,  2.,  1.,  0.],
#     [ 2.,  8.,  2.,  1.],
#     [ 1.,  2.,  6.,  2.],
#     [ 0.,  1.,  2.,  4.]
# ])

# c = np.array([10., -5., 20., -10.])

# def p_grad(x, i):
#     return (Q @ x - c)[i]

# L = np.ones(Q.shape[0]) 

# x0 = np.array([100., -100., 100., -100.])

# alpha = 0.5
# eps = 1e-6

# racdm = RACDM(p_grad, x0, alpha, L, eps)
# x_opt = racdm.solve()

# x_exact = np.linalg.solve(Q, c)

# print("RACDM:", x_opt)
# print("Exact: ", x_exact)
# print("Error: ", np.linalg.norm(x_opt - x_exact))