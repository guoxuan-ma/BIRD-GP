import numpy as np
from tqdm import tqdm

class FastHorseshoeLM:
    def __init__(self, mcmc_sample = 500, burnin = 500, thinning = 1, 
                 a_sigma = 0, b_sigma = 0, A_tau = 1, A_lambda = 1, 
                 progression_bar = True, track_loglik = True) -> None:
        self.mcmc_sample, self.burnin, self.thinning = mcmc_sample, burnin, thinning
        self.a_sigma, self.b_sigma = a_sigma, b_sigma
        self.A_tau, self.A_lambda = A_tau, A_lambda
        self.progression_bar, self.track_loglik = progression_bar, track_loglik
        self.chain = []
    
    def process_data(self, y, Z):
        self.y, self.Z = y, Z
        self.U, self.d, Vt = np.linalg.svd(self.Z, full_matrices = False)
        self.V = Vt.T
        
        self.n, self.K1 = self.Z.shape
        self.short_Z = self.n < self.K1
        
        self.y_star = self.U.T @ y
        self.VD = self.V * self.d
        if self.short_Z:
            self.d2 = 0
        else:
            self.DY_star = self.d * self.y_star
            
            self.d2 = np.sum(np.square(self.U @ self.y_star - self.y))
            self.ZtY = self.Z.T @ self.y
            self.VD2invVt = (self.V * (1 / np.square(self.d))) @ self.V.T
            self.ZtZ_inv = np.linalg.inv(self.Z.T @ self.Z)
    
    def initialize(self):
        self.beta = np.zeros(self.K1)
        self.lambda2 = np.ones(shape = self.K1)
        self.b_lambda = np.ones(shape = self.K1)
        self.sigma2 = 1
        self.tau2 = 1 / self.K1
        self.b = 1
        
        self.remember()
    
    def fit(self, X, y):
        self.process_data(y, X)
        self.initialize()
        self.update(round(self.burnin + self.mcmc_sample * self.thinning))
        
    def remember(self):
        current = {
            'beta': self.beta,
            'lambda2': self.lambda2,
            'b_lambda': self.b_lambda,
            'sigma2': self.sigma2,
            'tau2': self.tau2,
            'b': self.b
        }
        if self.track_loglik:
            current['loglik'] = self.loglik()
        self.chain.append(current)
    
    def update_beta(self):
        if self.short_Z:
            alpha1 = np.random.standard_normal(size = self.K1) * np.sqrt(self.sigma2) * np.sqrt(self.tau2) * np.sqrt(self.lambda2)
            alpha2 = np.random.standard_normal(size = self.n) * np.sqrt(self.sigma2)
            
            LambdaVD = np.sqrt(self.lambda2)[:, None] * self.VD
            A = LambdaVD.T @ LambdaVD
            A[np.diag_indices(self.n)] += 1 / self.tau2
            b = self.y_star - self.VD.T @ alpha1 - alpha2
            alpha = np.linalg.solve(A, b)
            
            self.beta = alpha1 + self.lambda2 * (self.VD @ alpha)
        else:
            # Omega = (self.V.T / self.lambda2) @ self.V / self.tau2
            # Omega[np.diag_indices(self.K1)] += np.square(self.d)
            # R = np.linalg.cholesky(Omega).T
            # b = np.linalg.solve(R.T, self.DY_star / np.sqrt(self.sigma2))
            # z = np.random.normal(size = self.K1)
            # alpha = np.linalg.solve(R, z + b)
            # self.beta = np.sqrt(self.sigma2) * (self.V @ alpha)
            
            t_star = self.tau2 * self.lambda2 * self.ZtY
            alpha1 = np.random.normal(scale = np.sqrt(self.sigma2) / self.d)
            alpha2 = np.random.normal(scale = np.sqrt(self.sigma2 * self.tau2 * self.lambda2))
            
            A = self.sigma2 * self.VD2invVt
            A[np.diag_indices(self.K1)] += self.sigma2 * self.tau2 * self.lambda2
            b = t_star - self.V @ alpha1 - alpha2
            alpha = np.linalg.solve(A, b)
            
            self.beta = self.V @ alpha1 + self.sigma2 * (self.ZtZ_inv @ alpha)
    
    def update_lambda2(self):
        rate = self.b_lambda + np.square(self.beta / (np.sqrt(self.sigma2) * np.sqrt(self.tau2))) / 2
        self.lambda2 = 1 / np.random.gamma(shape = 1, scale = 1 / rate)
    
    def update_b_lambda(self):
        # rate = 1 / np.square(self.A_lambda) + np.sum(1 / self.lambda2)
        # self.b_lambda = np.random.gamma(shape = 0.5 * (self.K1 + 1), scale = 1 / rate)
        rate = 1 / np.square(self.A_lambda) + 1 / self.lambda2
        self.b_lambda = np.random.gamma(shape = 1, scale = 1 / rate)
    
    def update_sigma2(self):
        rate = self.b_sigma + 0.5 / self.tau2 * np.sum(np.square(self.beta) / self.lambda2)
        # rate += 0.5 * np.sum(np.square(self.y - self.Z @ self.beta))
        rate += 0.5 * (np.sum(np.square(self.y_star - self.VD.T @ self.beta)) + self.d2)
        shape = self.a_sigma + 0.5 * (self.K1 + self.n)
        self.sigma2 = 1 / np.random.gamma(shape = shape, scale = 1 / rate)
        
        
        # rate = self.b_sigma + 0.5 / self.tau2 * np.sum(np.square(self.beta) / self.lambda2)
        # rate += 0.5 * np.sum(np.square(self.y_star - self.VD.T @ self.beta))
        # if self.short_Z:
        #     shape = self.a_sigma + 0.5 * (self.K1 + self.n)
        # else:
        #     shape = self.a_sigma + self.K1
        # self.sigma2 = rate / np.random.gamma(shape = shape)
    
    def update_tau2(self):
        rate = self.b + 0.5 / self.sigma2 * np.sum(np.square(self.beta / np.sqrt(self.lambda2)))
        self.tau2 = 1 / np.random.gamma(shape = 0.5 + self.K1 / 2, scale = 1 / rate)
    
    def update_b(self):
        rate = 1 / np.square(self.A_tau) + 1 / self.tau2
        self.b = np.random.gamma(shape = 1, scale = 1 / rate)
    
    def update(self, n_iter = 1):
        if self.progression_bar:
            iterator = tqdm(range(n_iter))
        else:
            iterator = range(n_iter)
        for _ in iterator:
            self.update_beta()
            self.update_lambda2()
            self.update_b_lambda()
            self.update_tau2()
            self.update_b()
            self.update_sigma2()
            
            self.remember()
    
    @property
    def coef_(self):
        return self.posterior_mean('beta')
    
    @property
    def lambdas(self):
        start = - round(self.mcmc_sample * self.thinning) - 1
        mat = [param['lambda2'] for param in self.chain[start:self.thinning:-1]]
        mat = np.sqrt(np.stack(mat))
        return np.mean(mat, axis = 0)
    
    def posterior_mean(self, key):
        start = - round(self.mcmc_sample * self.thinning) - 1
        mat = [param[key] for param in self.chain[start:-1:self.thinning]]
        mat = np.stack(mat)
        return np.mean(mat, axis = 0)
    
    def loglik(self):
        lik = - 0.5 * self.n * np.log(self.sigma2) - 0.5 * np.sum(np.square(self.y - self.Z @ self.beta)) / self.sigma2
        
        lik -= 0.5 * self.K1 * (np.log(self.sigma2) + np.log(self.tau2))
        lik -= 0.5 * np.sum(np.log(self.lambda2))
        lik -= 0.5 * np.sum(np.square(self.beta) / self.lambda2 / self.tau2 / self.sigma2)
        
        lik -= 1.5 * np.sum(np.log(self.lambda2)) + np.sum(self.b_lambda / self.lambda2)
        lik -= np.sum(1 / np.square(self.A_lambda) * self.b_lambda)
        
        lik -= 1.5 * np.log(self.tau2) + self.b / self.tau2 + 1 / np.square(self.A_tau) * self.b
        
        lik -= (self.a_sigma + 1) * np.log(self.sigma2) + self.b_sigma / self.sigma2
        return lik
        