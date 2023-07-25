import numpy as np
import torch
import bfnn
from fastBayesReg import FastHorseshoeLM
import svgd
from tqdm import tqdm
import itertools


class BIRD_GP:
    '''
    A class for the model Bayesian Image-on-image Regression via Deep kernel learning based Gaussian Processes (BIRD-GP) model.

    Attributes
    ----------
    grids_in : 2d array
        the grid point coordinates of all voxels on the predictor image; of size V_in * d_in
    grids_out : 2d array
        the grid point coordinates of all voxels on the outcome image; of size V_out * d_out
    L_in : int
        the number of basis functions used to fit the predictor image
    L_out : int
        the number of basis functions used to fit the outcome image
    d_in : int
        the number of dimensions of the predictor image
    d_out : int
        the number of dimensions of the outcome image
    V_in : int
        the number of voxels of the predictor image
    V_out : int
        the number of voxels of the outcome image
    bf_predictor_lr : double
        the learning rate of the Adam optimizer for training the basis fitting neural network of predicor images
    bf_predictor_steps : int
        the number of training steps of the Adam optimizer for training the basis fitting neural network of predictor images
    bf_outcome_lr : double
        the learning rate of the Adam optimizer for training the basis fitting neural network of outcome images
    bf_outcome_steps : int
        the number of training steps of the Adam optimizer for training the basis fitting neural network of outcome images
    hs_lm_mcmc_burnin : int
        the number of burin samples in the horseshoe-prior linear regression fitting the basis coefficients; shared by fitting both predictors and outcomes
    hs_lm_mcmc_samples : int
        the number of mcmc samples for estimation in the horseshoe-prior linear regression fitting the basis coefficients; shared by fitting both predictors and outcomes
    hs_lm_thinning : int
        the thining parameters in the horseshoe-prior linear regression fitting the basis coefficients; shared by fitting both predictors and outcomes
    hs_lm_a_sigma : double
        to do
    hs_lm_b_sigma : double
        to do
    hs_lm_A_tau : double
        to do
    hs_lm_A_lambda : double
        to do
    svgd_a_gamma : double
        the shape parameter in the gamma prior for the i.i.d. errors precision (gamma) in the Bayeian neural network
    svgd_b_gamma : double
        the scale parameter in the gamma prior for the i.i.d. errors precision (gamma) in the Bayeian neural network
    svgd_a_lambda : double
        the shape paramter in the gamma prior for the weights and biases precision (lambda) in the Bayeian neural network
    svgd_b_lambda : double
        the scale paramter in the gamma prior for the weights and biases precision (lambda) in the Bayeian neural network
    svgd_batch_size : int
        the batch size when training the Bayeian neural network by SVGD
    svgd_epochs : int
        the number of epochs when training the Bayeian neural network by SVGD
    device : string
        the device on which basis coefficients are fitted
    Psi_predictors : 2d array
        the fitted basis functions for predictors; of size V_in * L_in
    Psi_outcomes : 2d array
        the fitted basis functions for outcomes; of size V_out * L_out
    theta_train_predictors : 2d array
        the fitted basis coefficients for predictors in the training set; of size n_train * L_in
    theta_train_outcomes : 2d array
        the fitted basis coefficients for outcomes in the training set; of size n_train * L_out
    svgd_nn : a svgd_bnn class object
        the Stein variantional gradient descent Bayesian neural network model
    theta_train_outcomes_pred : 2d array
        the predicted basis coefficients for outcomes in the training set; of size n_train * L_out
    theta_train_outcomes_samples : 3d array
        the svgd samples of basis coefficients for outcomes in the training set; of size M * n_train * L_out
    train_outcomes_pred : 2d array
        the predicted outcome images in the training set; of size n_train * V_out

    Methods
    -------
    fit(predictors, outcomes)
        fit the BIRD-GP model
    predict_train()
        make prediction of outcomes on training data after fitting the model
    predict_test(test_predictors)
        make prediction of outcomes on testing data after fitting the model
    fit_basis(grids, images, lr, steps, d, V, L)
        fit basis functions by neural network
    fit_coefficients(images, Psi, hs_lm_mcmc_burnin, hs_lm_mcmc_samples, hs_lm_thinning, hs_lm_a_sigma, hs_lm_b_sigma, hs_lm_A_tau, hs_lm_A_lambda)
        fit basis coefficients by horseshoe-prior linear regression
    orthogonalization(Psi)
        orthogonalize basis functions by SVD
    '''

    def __init__(self,
                 predictor_grids,
                 outcome_grids,
                 predictor_L,
                 outcome_L,
                 bf_predictor_lr = 1e-3,
                 bf_predictor_steps = 10000, 
                 bf_outcome_lr = 1e-3,
                 bf_outcome_steps = 10000, 
                 hs_lm_mcmc_burnin = 500,
                 hs_lm_mcmc_samples = 500, 
                 hs_lm_thinning = 1, 
                 hs_lm_a_sigma = 0, 
                 hs_lm_b_sigma = 0, 
                 hs_lm_A_tau = 1, 
                 hs_lm_A_lambda = 1,
                 svgd_num_particles = 20,
                 svgd_a_gamma = 1, 
                 svgd_b_gamma = 1, 
                 svgd_a_lambda = 1, 
                 svgd_b_lambda = 1, 
                 svgd_batch_size = 64, 
                 svgd_epochs = 30,
                 device = None,
                 disable_message = True):
        '''
        initialization

        Parameters
        ----------
        predictor_grids : 2d array
            the grid point coordinates of all voxels on the predictor image; of size V_in * d_in
        outcome_grids : 2d array
            the grid point coordinates of all voxels on the outcome image; of size V_out * d_out
        predictor_L : int
            the number of basis functions used to fit the predictor image
        outcome_L : int
            the number of basis functions used to fit the outcome image
        bf_predictor_lr : double
            the learning rate of the Adam optimizer for training the basis fitting neural network of predicor images
        bf_predictor_steps : int
            the number of training steps of the Adam optimizer for training the basis fitting neural network of predictor images
        bf_outcome_lr : double
            the learning rate of the Adam optimizer for training the basis fitting neural network of outcome images
        bf_outcome_steps : int
            the number of training steps of the Adam optimizer for training the basis fitting neural network of outcome images
        hs_lm_mcmc_burnin : int
            the number of burin samples in the horseshoe-prior linear regression fitting the basis coefficients; shared by fitting both predictors and outcomes
        hs_lm_mcmc_samples : int
            the number of mcmc samples for estimation in the horseshoe-prior linear regression fitting the basis coefficients; shared by fitting both predictors and outcomes
        hs_lm_thinning : int
            the thining parameters in the horseshoe-prior linear regression fitting the basis coefficients; shared by fitting both predictors and outcomes
        hs_lm_a_sigma : double
            to do
        hs_lm_b_sigma : double
            to do
        hs_lm_A_tau : double
            to do
        hs_lm_A_lambda : double
            to do
        svgd_a_gamma : double
            the shape parameter in the gamma prior for the i.i.d. errors precision (gamma) in the Bayeian neural network
        svgd_b_gamma : double
            the scale parameter in the gamma prior for the i.i.d. errors precision (gamma) in the Bayeian neural network
        svgd_a_lambda : double
            the shape paramter in the gamma prior for the weights and biases precision (lambda) in the Bayeian neural network
        svgd_b_lambda : double
            the scale paramter in the gamma prior for the weights and biases precision (lambda) in the Bayeian neural network
        svgd_batch_size : int
            the batch size when training the Bayeian neural network by SVGD
        svgd_epochs : int
            the number of epochs when training the Bayeian neural network by SVGD
        device : string
            the device on which basis coefficients are fitted
        '''

        self.grids_in = predictor_grids
        self.grids_out = outcome_grids

        self.L_in = predictor_L
        self.L_out = outcome_L
        self.d_in = self.grids_in.shape[1]
        self.d_out = self.grids_out.shape[1]
        self.V_in = self.grids_in.shape[0]
        self.V_out = self.grids_out.shape[0]
        self.bf_predictor_lr = bf_predictor_lr
        self.bf_predictor_steps = bf_predictor_steps
        self.bf_outcome_lr = bf_outcome_lr
        self.bf_outcome_steps = bf_outcome_steps

        self.hs_lm_mcmc_burnin = hs_lm_mcmc_burnin
        self.hs_lm_mcmc_samples = hs_lm_mcmc_samples
        self.hs_lm_thinning = hs_lm_thinning
        self.hs_lm_a_sigma = hs_lm_a_sigma
        self.hs_lm_b_sigma = hs_lm_b_sigma 
        self.hs_lm_A_tau = hs_lm_A_tau
        self.hs_lm_A_lambda = hs_lm_A_lambda
        
        self.svgd_num_particles = svgd_num_particles
        self.svgd_a_gamma = svgd_a_gamma
        self.svgd_b_gamma = svgd_b_gamma
        self.svgd_a_lambda = svgd_a_lambda
        self.svgd_b_lambda = svgd_b_lambda
        self.svgd_batch_size = svgd_batch_size
        self.svgd_epochs = svgd_epochs
        
        self.disable_message = disable_message
        self.device = device
    

    def fit(self, predictors, outcomes):
        '''
        fit the BIRD-GP model

        Parameters
        ----------
        predictors : 2d array
            predictor images; of size n_train * V_in
        outcomes : 2d array
            outcome images; of size n_train * V_out
        '''
        self.train_outcomes = outcomes
        
        print("fit basis for predictors ...")
        Psi_predictors, _, _ = self.fit_basis(self.grids_in, 
                                              predictors, 
                                              self.bf_predictor_lr, 
                                              self.bf_predictor_steps,
                                              self.L_in
                                              )
        self.Psi_predictors, _, _ = self.orthogonalization(Psi_predictors)
        
        
        print("fit basis for outcomes ...")
        Psi_outcomes, _, _ = self.fit_basis(self.grids_out, 
                                            outcomes, 
                                            self.bf_outcome_lr, 
                                            self.bf_outcome_steps,
                                            self.L_out
                                            )
        self.Psi_outcomes, singular_values_outcomes, _ = self.orthogonalization(Psi_outcomes)
        
        
        self.Psi_predictors = np.hstack((np.ones((self.Psi_predictors.shape[0], 1)), self.Psi_predictors))
        self.Psi_outcomes = np.hstack((np.ones((self.Psi_outcomes.shape[0], 1)), self.Psi_outcomes))
        print("fit basis coefficients for predictors ...")
        self.theta_train_predictors, _ = self.fit_coefficients(predictors, 
                                                               self.Psi_predictors, 
                                                               self.hs_lm_mcmc_burnin,
                                                               self.hs_lm_mcmc_samples,
                                                               self.hs_lm_thinning,
                                                               self.hs_lm_a_sigma,
                                                               self.hs_lm_b_sigma,
                                                               self.hs_lm_A_tau,
                                                               self.hs_lm_A_lambda
                                                               )
        print("fit basis coefficients for outcomes ...")
        self.theta_train_outcomes, self.sigma2_ytrain = self.fit_coefficients(outcomes, 
                                                                              self.Psi_outcomes, 
                                                                              self.hs_lm_mcmc_burnin,
                                                                              self.hs_lm_mcmc_samples,
                                                                              self.hs_lm_thinning,
                                                                              self.hs_lm_a_sigma,
                                                                              self.hs_lm_b_sigma,
                                                                              self.hs_lm_A_tau,
                                                                              self.hs_lm_A_lambda
                                                                              )
        
        #self.eigen_values_outcomes = singular_values_outcomes ** 2
        self.empirical_eigen_values_outcomes = np.var(self.theta_train_outcomes, 0)
        print("stein variation gradient descent ...")
        self.svgd_nn = svgd.svgd_bnn(X_train = self.theta_train_predictors, 
                                     y_train = self.theta_train_outcomes,
                                     #var_vec_y = self.empirical_eigen_values_outcomes,
                                     var_vec_y = np.ones(self.empirical_eigen_values_outcomes.shape[0]),
                                     a_gamma = self.svgd_a_gamma, 
                                     b_gamma = self.svgd_b_gamma, 
                                     a_lambda = self.svgd_a_lambda,
                                     b_lambda = self.svgd_b_lambda, 
                                     batch_size = self.svgd_batch_size, 
                                     epochs = self.svgd_epochs,
                                     M = self.svgd_num_particles,
                                    )
        self.svgd_nn.train()
    
    
    def predict_train(self, sample = False, size = 1000, CI = 0.95):
        '''
        make prediction of outcomes on training data after fitting the model

        Return
        ------
        an 2d array of size n_train * V_out containing the predicted outcome images in the training set
        '''

        self.theta_train_outcomes_pred, self.theta_train_outcomes_samples = self.svgd_nn.predict(self.theta_train_predictors)
        self.train_outcomes_pred = self.theta_train_outcomes_pred @ self.Psi_outcomes.T
        
        '''
        M = self.theta_train_outcomes_samples.shape[0]
        n = self.theta_train_outcomes_pred.shape[0]
        V = self.Psi_outcomes.shape[0]
        self.train_outcomes_samples = np.zeros((M, n, V))
        for m in range(M):
            self.train_outcomes_samples[m, :, :] = self.theta_train_outcomes_samples[m, :, :] @ self.Psi_outcomes.T
        
        one_side_prob = (1 - CI) / 2
        self.lower_boundary = np.quantile(self.train_outcomes_samples, one_side_prob, axis = 0)
        self.upper_boundary = np.quantile(self.train_outcomes_samples, 1 - one_side_prob, axis = 0)
        ''' 
        
        if sample:
            n_train = self.theta_train_outcomes_pred.shape[0]
            coverage_rates = np.zeros(n_train)
            print("sampling training images")
            for i in tqdm(range(n_train), disable = self.disable_message):
                eigen_values_outcomes = self.empirical_eigen_values_outcomes#np.square(np.std(self.theta_train_outcomes, 0))
                samples = self.sample_images_for_one_input_image(size, 
                                                                 self.theta_train_predictors[i, :], 
                                                                 eigen_values_outcomes, 
                                                                 self.sigma2_ytrain[i])
                coverage_rates[i] = self.coverage_rate_for_one_input_image(CI, samples, self.train_outcomes[i, :])
            self.train_coverage_rates = coverage_rates
            return(self.train_outcomes_pred, self.train_coverage_rates)
        else:
            return(self.train_outcomes_pred)
        
        
    def predict_test_lr(self, test_predictors, test_confounders = None):    
        from sklearn.linear_model import LinearRegression
        self.test_predictors = test_predictors
        n_test = self.test_predictors.shape[0]
        self.theta_test_predictors = np.zeros((n_test, self.Psi_predictors.shape[1]))
        for i in range(n_test):
            lm = LinearRegression(fit_intercept = False).fit(self.Psi_predictors, self.test_predictors[i, :])
            self.theta_test_predictors[i, :] = lm.coef_
        if test_confounders is not None:
            self.theta_test_predictors = np.hstack((self.theta_test_predictors, test_confounders))
        self.theta_test_outcomes_pred, self.theta_test_outcomes_samples = self.svgd_nn.predict(self.theta_test_predictors)
        self.test_outcomes_pred = self.theta_test_outcomes_pred @ self.Psi_outcomes.T
        return self.test_outcomes_pred
    
    
    def predict_test(self, test_predictors, sample = False, test_outcomes = None, size = 1000, CI = 0.95):
        '''
        make prediction of outcomes on testing data after fitting the model

        Return
        ------
        an 2d array of size n_test * V_out containing the predicted outcome images in the testing set
        '''
        self.test_predictors = test_predictors
        self.theta_test_predictors, _ = self.fit_coefficients(self.test_predictors, 
                                                              self.Psi_predictors, 
                                                              self.hs_lm_mcmc_burnin,
                                                              self.hs_lm_mcmc_samples,
                                                              self.hs_lm_thinning,
                                                              self.hs_lm_a_sigma,
                                                              self.hs_lm_b_sigma,
                                                              self.hs_lm_A_tau,
                                                              self.hs_lm_A_lambda
                                                              )
        self.theta_test_outcomes_pred, self.theta_test_outcomes_samples = self.svgd_nn.predict(self.theta_test_predictors)
        self.test_outcomes_pred = self.theta_test_outcomes_pred @ self.Psi_outcomes.T
        
        '''
        M = theta_test_outcomes_samples.shape[0]
        n = theta_test_outcomes_pred.shape[0]
        V = self.Psi_outcomes.shape[0]
        test_outcomes_samples = np.zeros((M, n, V))
        for m in range(M):
            test_outcomes_samples[m, :, :] = theta_test_outcomes_samples[m, :, :] @ self.Psi_outcomes.T
        
        one_side_prob = (1 - CI) / 2
        lb = np.quantile(test_outcomes_samples, one_side_prob, axis = 0)
        ub = np.quantile(test_outcomes_samples, 1 - one_side_prob, axis = 0)
        '''
        if sample:
            n_test = test_predictors.shape[0]
            coverage_rates = np.zeros(n_test)
            print("sampling testing images")
            for i in tqdm(range(n_test), disable = self.disable_message):
                eigen_values_outcomes = self.empirical_eigen_values_outcomes #np.square(np.std(self.theta_train_outcomes, 0))
                samples = self.sample_images_for_one_input_image(size, 
                                                                 self.theta_test_predictors[i, :], 
                                                                 eigen_values_outcomes, 
                                                                 np.mean(self.sigma2_ytrain))
                coverage_rates[i] = self.coverage_rate_for_one_input_image(CI, samples, test_outcomes[i, :])
            self.test_coverage_rates = coverage_rates
            return(self.test_outcomes_pred, self.test_coverage_rates)
        else:
            return(self.test_outcomes_pred)
        
    
    def fit_basis(self, grids, images, lr, steps, L):
        '''
        fit basis functions by neural network

        Parameters
        ----------
        grids : 2d array
            the grid point coordinates of all voxels on the image input into this function; of size n * d
        images : 2d array
            images on which basis functions are fitted; of size n * V
        lr : double
            the learning rate of the Adam optimizer for training the basis fitting neural network
        steps : int
            the number of training steps of the Adam optimizer for the basis fitting neural network
        L : int
            the number of basis functions used to fit the image

        Return
        ------
        a 2d array of size V * L containing the fitted basis functions
        '''
        total_variance = np.var(images, 0)
        defined = (total_variance != 0)
        total_variance = total_variance[defined]
        
        n = images.shape[0]
        V = grids.shape[0]
        d = grids.shape[1]
        grids = torch.tensor(grids, dtype = torch.float32).to(self.device)
        images = torch.tensor(images, dtype = torch.float32).to(self.device)
        bfnn_model = bfnn.BFNN(d = d, L = L, n = n, V = V).to(self.device)
        bf_optimizer = torch.optim.Adam(bfnn_model.parameters(), lr = lr)
        mse_criterion = torch.nn.MSELoss()
        iterator = tqdm(range(steps), disable = self.disable_message)
        mse_trace = np.empty(0)
        r2_trace = np.empty(0)
        for i in iterator:
            yhat = bfnn_model.forward(grids)
            loss = mse_criterion(yhat, images)
            bf_optimizer.zero_grad()
            loss.backward()
            bf_optimizer.step()
            if i % 50 == 0:
                mse_trace = np.concatenate((mse_trace, [loss.detach().to("cpu").numpy()]))
                
                mse = np.mean((yhat.detach().to("cpu").numpy() - images.detach().to("cpu").numpy()) ** 2, 0)
                mse = mse[defined]
                r2_trace = np.concatenate((r2_trace, [np.mean(1 - mse / total_variance)]))
        return(bfnn_model.Psi.detach().to("cpu").numpy(), mse_trace, r2_trace)
    
    
    def fit_coefficients(self, 
                         images, 
                         Psi,
                         hs_lm_mcmc_burnin, 
                         hs_lm_mcmc_samples,
                         hs_lm_thinning,
                         hs_lm_a_sigma,
                         hs_lm_b_sigma,
                         hs_lm_A_tau,
                         hs_lm_A_lambda
                        ):
        '''
        fit basis coefficients via horseshoe-prior linear regression
        
        Parameters
        ----------
        images : 2d array
            images on which basis functions are fitted; of size n * V
        Psi : 2d array
            the fitted basis functions; of size V * L
        hs_lm_mcmc_burnin : int
            the number of burin samples in the horseshoe-prior linear regression fitting the basis coefficients; shared by fitting both predictors and outcomes
        hs_lm_mcmc_samples : int
            the number of mcmc samples for estimation in the horseshoe-prior linear regression fitting the basis coefficients; shared by fitting both predictors and outcomes
        hs_lm_thinning : int
            the thining parameters in the horseshoe-prior linear regression fitting the basis coefficients; shared by fitting both predictors and outcomes
        hs_lm_a_sigma : double
            to do
        hs_lm_b_sigma : double
            to do
        hs_lm_A_tau : double
            to do
        hs_lm_A_lambda : double
            to do
        '''
        
        n = images.shape[0]
        L = Psi.shape[1]
        lm = FastHorseshoeLM(burnin = hs_lm_mcmc_burnin, 
                             mcmc_sample = hs_lm_mcmc_samples, 
                             thinning = hs_lm_thinning, 
                             a_sigma = hs_lm_a_sigma,
                             b_sigma = hs_lm_b_sigma, 
                             A_tau = hs_lm_A_tau, 
                             A_lambda = hs_lm_A_lambda,
                             progression_bar = False,
                             track_loglik = False)
        
        theta = np.zeros((n, L))
        sigma2 = np.zeros(n)
        iterator = tqdm(range(n), disable = self.disable_message)
        for i in iterator:
            lm.fit(Psi, images[i, :])
            theta[i, :] = lm.coef_
            sigma2[i] = lm.posterior_mean("sigma2")
            
        return(theta, sigma2)
    
    
    def orthogonalization(self, Psi):
        '''
        orthogonalize the input basis functions

        Parameters
        ----------
        Psi : 2d array
            basis functions; of size V * L
        
        Return
        ------
        a tuple of the results for the Singular Value Decomposition on the input basis functions
        '''

        return(np.linalg.svd(Psi, full_matrices = False))
        
        
    def sample_images_for_one_input_image(self, size, theta_predictor, eigen_values_outcomes, sigma2):
        mean_theta_outcome, _ = self.svgd_nn.predict(theta_predictor.reshape(1, -1))
        outcome_image_samples = np.zeros((size, self.V_out))
        for s in range(size):
            pos_gamma = np.exp(np.mean(self.svgd_nn.loggamma))
            theta_outcome_sample = np.random.normal(loc = mean_theta_outcome, scale = np.sqrt(eigen_values_outcomes / pos_gamma))
            mean_outcome_image = (theta_outcome_sample @ self.Psi_outcomes.T).reshape(-1)
            outcome_image_samples[s, :] = np.random.normal(loc = mean_outcome_image, scale = np.sqrt(sigma2))
        return(outcome_image_samples)
    
    
    def coverage_rate_for_one_input_image(self, CI, outcome_image_samples, outcome_image):
        one_side_prob = (1 - CI) / 2
        lb = np.quantile(outcome_image_samples, one_side_prob, axis = 0)
        ub = np.quantile(outcome_image_samples, 1 - one_side_prob, axis = 0)
        coverage_rate = np.mean((lb < outcome_image) * (ub > outcome_image))
        return(coverage_rate)
    
    
    def evaluate_basis_fit(self, predictors, outcomes):
        fitted_predictors = self.theta_train_predictors @ self.Psi_predictors.T
        fitted_outcomes = self.theta_test_predictors @ self.Psi_outcomes.T
        mse_predictors = np.mean((predictors - fitted_predictors) ** 2)
        mse_outcomes = np.mean((outcomes - fitted_outcomes) ** 2)
        return(mse_predictors, mse_predictors)
        
    '''
    def generate_2d_grids(self, dims):
        dy = dims[0]
        dx = dims[1]
        grid_x = np.tile(np.arange(dx), dy)
        grid_y = np.repeat(np.arange(dy), dx)
        grid_x = (grid_x - (dx - 1) / 2) / ((dx - 1) / 2)
        grid_y = (grid_y - (dy - 1) / 2) / ((dy - 1) / 2)
        grids = np.vstack((grid_x, grid_y))
        return(grids.T)
    '''


def generate_grids(dims):
    '''
    the helper function to generate evenly separated grid points given the specified dimensions

    Parameters
    ----------
    dims : list
        number of points (voxels) in each dimension

    Return
    ------
    a 2d array of grid point coordinations; of size V * d
    '''

    #scale = np.array(dims) / np.min(dims)
    base_lists = list()
    counter = 0
    for d in dims:
        base_list_d = np.arange(d)
        base_list_d = (base_list_d - (d - 1) / 2) / ((d - 1) / 2) #* scale[counter]
        base_lists.append(base_list_d)
        counter = counter + 1
    
    grids = np.zeros((np.prod(dims), len(dims)))
    counter = 0
    for point in itertools.product(*base_lists):
        for dim in range(len(dims)):
            grids[counter, dim] = point[dim]
        counter = counter + 1
    
    return(grids)         
        
        
    