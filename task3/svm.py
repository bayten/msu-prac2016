import numpy as np
from cvxopt import matrix, solvers
import sklearn.svm as sk_svm
from math import exp
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import time as timer


class SVM:
    C = 0.5  # regularisation parameter
    method = 'primal'
    # method in ['primal', 'dual', 'subgradient', 'stoch_subgradient',
    # 'liblinear', 'libsvm']

    w = None  # weights vector (only for primal task)
    w0 = None  # constant for decision function (only for primal task)
    A = None  # values of dual variables (only for dual task)
    sv = None  # support vectors with classes (only for dual task)

    X_train = None  # memorised matrix for kernel trick
    y_train = None  # memorised vector for kernel trick

    kernel = 'rbf'  # kernel in ['linear', 'rbf']
    gamma = 1.0  # width of RBF kernel (if kernel = 'rbf', of course)

    inner_svm = None  # storage for external SVM classes

    def __init__(self, C=0.5, method='primal', kernel='rbf', gamma=1.0):
        self.C = C
        self.method = method
        self.kernel = kernel
        self.gamma = gamma

    def compute_primal_objective(self, X, y):
        if self.w is None or self.w0 is None:
            raise ValueError('''Weights weren't computed''')
        N, D = X.shape
        bound_term = 0.0
        for i in range(N):
            bound_term += max(0, 1 - y[i]*(X[i] @ self.w + self.w0))

        return 0.5*norm(np.hstack((self.w.ravel(), self.w0)))**2 + \
            self.C*bound_term

    def compute_dual_objective(self, X, y):
        if self.A is None:
            raise ValueError('''Support vector coeffs weren't computed''')

        N, D = X.shape
        tmp_sum = 0.0

        if self.kernel == 'linear':
            tmp_sum = self.kmfunc_linear(X, X) @ self.A @ self.A.T * y

        elif self.kernel == 'rbf':
            tmp_sum = self.kmfunc_rbf(X, X) @ self.A @ self.A.T * y

        return np.sum(self.A)-0.5*np.sum(tmp_sum)

    def score(self, X, y):
        if self.method in ['primal', 'subgradient', 'stoch_subgradient',
                           'liblinear']:
            return self.compute_primal_objective(X, y)
        elif self.method in ['dual', 'libsvm']:
            return self.compute_dual_objective(X, y)
        else:
            raise AttributeError('Unknown method')

    def fit(self, X, y, tol=1e-4, max_iter=2000, verbose=True,
            stop_criterion='objective', batch_size=1, lamb=0.01, alpha=1.0,
            beta=0.0):
        N, D = X.shape
        if self.method == 'primal':
            solvers.options['show_progress'] = verbose
            solvers.options['maxiters'] = max_iter
            solvers.options['feastol'] = tol
            P = np.zeros((D+1+N, D+1+N))
            P[:D, :D] = np.eye(D)

            q = np.zeros(D+1+N)
            q[D+1:] = self.C

            G = np.zeros((2*N, D+1+N))
            G[:N, D+1:] = -np.eye(N)
            G[N:, D+1:] = -np.eye(N)
            G[N:, :D] = -y.reshape((N, 1))*X
            G[N:, D] = -y
            h = np.zeros(2*N)
            h[N:] = -1

            P = matrix(P, tc='d')
            q = matrix(q, tc='d')
            G = matrix(G, tc='d')
            h = matrix(h, tc='d')
            start_time = timer.time()
            sol_obj = solvers.qp(P, q, G, h)
            fit_time = timer.time()-start_time

            solution = np.array(sol_obj['x'])
            status = 0 if sol_obj['status'] == 'optimal' else 1

            self.w = solution[:D].reshape((D, 1))
            self.w0 = solution[D]
            return {'status': status,
                    'time': fit_time}

        elif self.method == 'dual':
            solvers.options['show_progress'] = verbose
            solvers.options['maxiters'] = max_iter
            solvers.options['feastol'] = tol
            y = y.reshape((N, 1))

            P = None
            if self.kernel == 'rbf':
                P = self.kmfunc_rbf(X, X)
            elif self.kernel == 'linear':
                P = self.kmfunc_linear(X, X)
            P = P * y * y.T

            q = -np.ones(N)

            G = np.zeros((2*N, N))
            G[:N] = -np.eye(N)
            G[N:] = np.eye(N)
            h = np.zeros(2*N)
            h[N:] = self.C

            A = y.T
            b = 0

            P = matrix(P, tc='d')
            q = matrix(q, tc='d')
            G = matrix(G, tc='d')
            h = matrix(h, tc='d')
            A = matrix(A, tc='d')
            b = matrix(b, tc='d')

            start_time = timer.time()
            sol_obj = solvers.qp(P, q, G, h, A, b)
            fit_time = timer.time()-start_time

            solution = np.array(sol_obj['x'])
            status = 0 if sol_obj['status'] == 'optimal' else 1
            self.A = solution.reshape((N, 1))
            self.X_train = X.copy()
            self.y_train = y.copy().reshape((N, 1))

            sv_inds = np.where((self.A > 0.0) * (self.A < self.C))[0]
            self.sv = list(zip(X[sv_inds], y[sv_inds]))
            return {'time': fit_time,
                    'status': status}

        elif self.method == 'subgradient':
            # initialising all vars
            self.w = np.zeros(D)
            prev_w = self.w.copy()
            self.w0 = np.zeros(1)
            prev_w0 = self.w0.copy()
            func_err = self.compute_primal_objective(X, y)

            graph_data = []
            iter_count = 0
            status = 1
            start_time = timer.time()
            for iter_n in range(1, max_iter+1):
                # computing one more step of gradient
                iter_count += 1
                graph_data.append(self.compute_primal_objective(X, y))
                eta_coef = alpha/(iter_n**beta)

                if verbose is True:
                    print("< ITER", iter_n, ">")
                    print("w vector:", self.w)
                    print("w0:", self.w0)

                self.w -= eta_coef * prev_w
                for n in range(N):
                    if 1.0 - y[n]*(X[n] @ prev_w + prev_w0) > 0.0:
                        self.w += eta_coef * self.C * y[n] * X[n]  # doubled -
                        self.w0 += eta_coef * self.C * y[n]

                # checking stop criterion
                if stop_criterion == 'argument':
                    if norm(np.hstack((self.w, self.w0)) -
                            np.hstack((prev_w, prev_w0))) <= tol:
                        status = 0
                        break

                elif stop_criterion == 'objective':
                    new_err = self.compute_primal_objective(X, y)
                    if abs(func_err-new_err) <= tol:
                        status = 0
                        break

                    func_err = new_err
                else:
                    raise ValueError('Unknown stop criterion')

                prev_w = self.w.copy()
                prev_w0 = self.w0.copy()
            fit_time = timer.time()-start_time
            self.w = self.w.reshape((D, 1))
            return {'objective_curve': graph_data,
                    'status': status,
                    'iteration': iter_count,
                    'time': fit_time}

        elif self.method == 'stoch_subgradient':
            self.w = np.zeros(D)
            prev_w = self.w.copy()
            self.w0 = np.random.rand(1)
            prev_w0 = self.w0.copy()

            func_err = self.compute_primal_objective(X, y)
            graph_data = []
            iter_count = 0
            status = 1
            start_time = timer.time()
            for iter_n in range(1, max_iter+1):
                # computing one more step of gradient
                eta_coef = alpha/(iter_n**beta)
                iter_count += 1
                graph_data.append(self.compute_primal_objective(X, y))
                # preparing batch indices
                batch_indices = np.arange(N)
                np.random.shuffle(batch_indices)
                batch_indices = batch_indices[:batch_size]

                self.w -= eta_coef * prev_w
                for n in batch_indices:
                    if 1.0-y[n]*(X[n] @ prev_w + prev_w0) >= 0.0:
                        self.w += eta_coef * self.C * y[n] * X[n]
                        self.w0 += eta_coef * self.C * y[n]

                # checking stop criterion
                if stop_criterion == 'argument':
                    if norm(np.hstack((self.w, self.w0)) -
                            np.hstack((prev_w, prev_w0))) <= tol:
                        status = 0
                        break
                elif stop_criterion == 'objective':
                    func_err = (1-lamb)*func_err + \
                               lamb*self.compute_primal_objective(X, y)
                    if func_err <= tol:
                        status = 0
                        break
                else:
                    raise ValueError('Unknown stop criterion')
                prev_w = self.w.copy()
                prev_w0 = self.w0.copy()

            fit_time = timer.time() - start_time
            self.w = self.w.reshape((D, 1))
            return {'objective_curve': graph_data,
                    'status': status,
                    'iteration': iter_count,
                    'time': fit_time}

        elif self.method == 'liblinear':
            self.inner_svm = sk_svm.LinearSVC(C=self.C, tol=tol,
                                              verbose=verbose,
                                              max_iter=max_iter)
            start_time = timer.time()
            self.inner_svm.fit(X, y)
            fit_time = timer.time() - start_time
            self.w = self.inner_svm.coef_.T
            self.w0 = self.inner_svm.intercept_
            return {'time': fit_time}

        elif self.method == 'libsvm':
            self.A = np.zeros((X.shape[0], 1))
            self.inner_svm = sk_svm.SVC(self.C, self.kernel, gamma=self.gamma,
                                        tol=tol, verbose=verbose,
                                        max_iter=max_iter, probability=True)
            start_time = timer.time()
            self.inner_svm.fit(X, y)
            fit_time = timer.time()-start_time
            self.sv = list(zip(self.inner_svm.support_vectors_,
                               y[self.inner_svm.support_]))
            self.A[self.inner_svm.support_] = self.inner_svm.dual_coef_.T
            self.X_train = X.copy()
            self.y_train = y.copy()
            return {'time': fit_time}

        else:
            raise AttributeError('Unknown method:', method)

    def predict(self, X_test, return_classes=False):
        N, D = X_test.shape

        if self.method == 'dual':
            myN = self.X_train.shape[0]

            w_matrix = None
            w0_matrix = None

            sv_inds = np.where((self.A > 0.0) * (self.A < self.C))[0]

            if self.kernel == 'rbf':
                w_matrix = self.kmfunc_rbf(X_test, self.X_train[sv_inds])
                w0_matrix = self.kmfunc_rbf(self.X_train,
                                            self.X_train[sv_inds])

            elif self.kernel == 'linear':
                w_matrix = self.kmfunc_linear(X_test, self.X_train)
                w0_matrix = self.kmfunc_linear(self.X_train,
                                               self.X_train[sv_inds])
            w_matrix = w_matrix @ (self.y_train * self.A)
            w0_matrix = w0_matrix @ (self.y_train * self.A)

            kernel_w0 = (np.sum(self.y_train[sv_inds]) -
                         np.sum(w0_matrix)) / sv_inds.shape[0]
            y_pred = w_matrix + kernel_w0

            if return_classes is True:
                return 2*(y_pred >= 0).astype(np.int64)-1
            else:
                return y_pred

        elif self.method in ['primal', 'subgradient', 'stoch_subgradient']:
            y_pred = X_test @ self.w + self.w0
            if return_classes is True:
                return 2*(y_pred > 0).astype(np.int64)-1
            else:
                return y_pred

        elif self.method in ['libsvm', 'liblinear']:
            if return_classes is True:
                return self.inner_svm.predict(X_test)
            else:
                return self.inner_svm.decision_function(X_test)
        else:
            raise AttributeError('Unknown method:', self.method)

    def compute_support_vectors(self, with_labels=False):
        if self.sv is not None:
            if with_labels is True:
                return self.sv
            else:
                return [elem[0] for elem in self.sv]
        else:
            raise ValueError('''Support vectors weren't computed''')

    def compute_w(self):
        if kernel == 'rbf':
            raise AttributeError('''Can't use compute_w with RBF kernel''')
        sv_num = len(self.sv)
        sv = np.array([x[0] for x in self.sv]).reshape((sv_num, 1))
        sa = np.array([x[1] for x in self.sv]).reshape((sv_num, 1))
        return self.A[self.A != 0] * sa * np.sum(sv, axis=0)

    def kmfunc_rbf(self, fstX, secX):
        return np.exp(-self.gamma*np.power(cdist(fstX, secX, 'euclidean'), 2))

    def kmfunc_linear(self, fstX, secX):
        return fstX @ secX.T


def visualize(X, y, alg_svm, show_vectors=False):
    N, D = X.shape

    if D != 2:
        raise ValueError('''Can't visualize multidimensional data''')

    # following code was inspired by SVM visualiser example:
    # http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    y_pred = alg_svm.predict(np.c_[xx.ravel(), yy.ravel()], True)
    y_pred = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.coolwarm, alpha=0.8)

    if show_vectors is True:
        supp_inds = ~(alg_svm.A == 0).ravel()
        colors = []
        for y_elem in y[supp_inds]:
            if y_elem == 1:
                colors.append('c')
            else:
                colors.append('y')
        area = np.pi * (np.ones(len(colors))*4.5)**2
        plt.scatter(X[~supp_inds, 0], X[~supp_inds, 1], c=y[~supp_inds])
        plt.scatter(X[supp_inds, 0], X[supp_inds, 1], s=area, c=y[supp_inds],
                    marker="*", alpha=1.0)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()
    pass
