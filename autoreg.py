class VectAutoReg:
    def __init__(self, coeffs, bias=None, init=None):
        self.coeffs = coeffs
        self.bias = bias
        self.dim = len(coeffs[0])
        self.order = len(coeffs)
        if init is None:
            self.init = np.zeros((self.order, self.dim))
        else:
            self.init = init
        self.series = None
        self.length = None
        self.companion = None
        
    def __call__(self, length):
        self.series = [*self.init]
        self.length = length
        for i in range(length):
            self.series.append(self._next())
        return np.array(self.series).transpose()

    def __len__(self):
        return self.length

    def _generate_companion(self):
        self.companion = self.stability_companion(self.coeffs)
        return self.companion
    
    def _compute_stability(self):
        if self.companion is not None:
            return self.is_stable(self.companion)
        else:
            raise Exception("Companion matrix not generated.")

    def _next(self):
        next_vec = np.zeros(self.dim)
        for i in range(self.order):
            next_vec += self.coeffs[i] @ self.series[-i-1] + np.random.normal(0, 1, self.dim)
        if self.bias is not None:
            next_vec += self.bias
        return next_vec

    @staticmethod
    def stability_companion(coeffs):
        d = len(coeffs[0])
        p = len(coeffs)
        first_column = [np.eye(d)]
        for i in range(1, p):
            first_column.append(np.zeros((d, d)))
        first_column = np.concatenate(first_column, axis=0)

        companion_matrix = np.zeros((d * p, d * p))
        companion_matrix[:d, :] = np.hstack(coeffs)
        companion_matrix[d:, :d*(p-1)] = first_column[:-d]
        return companion_matrix

    @staticmethod
    def is_stable(mat):
        return np.all(np.abs(np.linalg.eigvals(mat)) < 1)
    
# USAGE EXAMPLE:
# import numpy as np
# import matplotlib.pyplot as plt
# A1 = np.array([[0.5, 0.1], [0.1, 0.5]])
# A2 = np.array([[0.3, 0.0], [0.0, 0.3]])
# autoreg = VectAutoReg([A1, A2])
# series = autoreg(1000)
# for s in series:
#     plt.plot(s)
# plt.show()
# comp = autoreg._generate_companion()
# print(autoreg._compute_stability())
