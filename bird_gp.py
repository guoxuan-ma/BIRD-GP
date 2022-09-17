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
        the shape parameter of the inverse gamma prior of the noise variance in the horseshoe-prior linear regression; shared by fitting both predictors and outcomes
    hs_lm_b_sigma : double
        the rate parameter of the inverse gamma prior of the noise variance in the horseshoe-prior linear regression; shared by fitting both predictors and outcomes
    hs_lm_A_tau : double
        the scale parameter of the half Cauchy prior of the global shrinkage parameter in the horseshoe-prior linear regression; shared by fitting both predictors and outcomes
    hs_lm_A_lambda : double
        the scale parameter of the half Cauchy prior of the local shrinkage parameter in the horseshoe-prior linear regression; shared by fitting both predictors and outcomes
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
                 svgd_a_gamma = 1, 
                 svgd_b_gamma = 1, 
                 svgd_a_lambda = 1, 
                 svgd_b_lambda = 1, 
                 svgd_batch_size = 64, 
                 svgd_epochs = 30,
                 device = None
                 ):
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
            the shape parameter of the inverse gamma prior of the noise variance in the horseshoe-prior linear regression; shared by fitting both predictors and outcomes
        hs_lm_b_sigma : double
            the rate parameter of the inverse gamma prior of the noise variance in the horseshoe-prior linear regression; shared by fitting both predictors and outcomes
        hs_lm_A_tau : double
            the scale parameter of the half Cauchy prior of the global shrinkage parameter in the horseshoe-prior linear regression; shared by fitting both predictors and outcomes
        hs_lm_A_lambda : double
        the scale parameter of the half Cauchy prior of the local shrinkage parameter in the horseshoe-prior linear regression; shared by fitting both predictors and outcomes
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

        self.svgd_a_gamma = svgd_a_gamma
        self.svgd_b_gamma = svgd_b_gamma
        self.svgd_a_lambda = svgd_a_lambda
        self.svgd_b_lambda = svgd_b_lambda
        self.svgd_batch_size = svgd_batch_size
        self.svgd_epochs = svgd_epochs
        
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

        print("fit basis for predictors ...")
        Psi_predictors = self.fit_basis(self.grids_in, 
                                        predictors, 
                                        self.bf_predictor_lr, 
                                        self.bf_predictor_steps,
                                        self.L_in
                                       )
        print("fit basis for outcomes ...")
        Psi_outcomes = self.fit_basis(self.grids_out, 
                                      outcomes, 
                                      self.bf_outcome_lr, 
                                      self.bf_outcome_steps,
                                      self.L_out
                                     )
        print("basis orthogonalization ...")
        self.Psi_predictors, _, _ = self.orthogonalization(Psi_predictors)
        self.Psi_outcomes, _, _ = self.orthogonalization(Psi_outcomes)
        print("fit basis coefficients for predictors ...")
        self.theta_train_predictors = self.fit_coefficients(predictors, 
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
        self.theta_train_outcomes = self.fit_coefficients(outcomes, 
                                                          self.Psi_outcomes, 
                                                          self.hs_lm_mcmc_burnin,
                                                          self.hs_lm_mcmc_samples,
                                                          self.hs_lm_thinning,
                                                          self.hs_lm_a_sigma,
                                                          self.hs_lm_b_sigma,
                                                          self.hs_lm_A_tau,
                                                          self.hs_lm_A_lambda
                                                         )
        print("stein variation gradient descent ...")
        self.svgd_nn = svgd.svgd_bnn(X_train = self.theta_train_predictors, 
                                     y_train = self.theta_train_outcomes,
                                     a_gamma = self.svgd_a_gamma, 
                                     b_gamma = self.svgd_b_gamma, 
                                     a_lambda = self.svgd_a_lambda,
                                     b_lambda = self.svgd_b_lambda, 
                                     batch_size = self.svgd_batch_size, 
                                     epochs = self.svgd_epochs
                                    )
        self.svgd_nn.train()
    
    
    def predict_train(self):
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
        
        return(self.train_outcomes_pred)
    
    
    def predict_test(self, test_predictors):
        '''
        make prediction of outcomes on testing data after fitting the model

        Return
        ------
        an 2d array of size n_test * V_out containing the predicted outcome images in the testing set
        '''

        theta_test_predictors = self.fit_coefficients(test_predictors, 
                                                      self.Psi_predictors, 
                                                      self.hs_lm_mcmc_burnin,
                                                      self.hs_lm_mcmc_samples,
                                                      self.hs_lm_thinning,
                                                      self.hs_lm_a_sigma,
                                                      self.hs_lm_b_sigma,
                                                      self.hs_lm_A_tau,
                                                      self.hs_lm_A_lambda
                                                     )
        theta_test_outcomes_pred, theta_test_outcomes_samples = self.svgd_nn.predict(theta_test_predictors)
        test_outcomes_pred = theta_test_outcomes_pred @ self.Psi_outcomes.T
        
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
        
        return(test_outcomes_pred)
        
    
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

        n = images.shape[0]
        V = grids.shape[0]
        d = grids.shape[1]
        grids = torch.tensor(grids, dtype = torch.float32)
        images = torch.tensor(images, dtype = torch.float32)
        bfnn_model = bfnn.BFNN(d = d, L = L, n = n, V = V)
        bf_optimizer = torch.optim.Adam(bfnn_model.parameters(), lr = lr)
        mse_criterion = torch.nn.MSELoss()
        iterator = tqdm(range(steps))        
        for _ in iterator:
            yhat = bfnn_model.forward(grids)
            loss = mse_criterion(yhat, images)
            bf_optimizer.zero_grad()
            loss.backward()
            bf_optimizer.step()
        return(bfnn_model.Psi.detach().numpy())
    
    
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
            the number of burin samples in the horseshoe-prior linear regression fitting the basis coefficients
        hs_lm_mcmc_samples : int
            the number of mcmc samples for estimation in the horseshoe-prior linear regression fitting the basis coefficients
        hs_lm_thinning : int
            the thining parameters in the horseshoe-prior linear regression fitting the basis coefficients
        hs_lm_a_sigma : double
            the shape parameter of the inverse gamma prior of the noise variance in the horseshoe-prior linear regression
        hs_lm_b_sigma : double
            the rate parameter of the inverse gamma prior of the noise variance in the horseshoe-prior linear regression
        hs_lm_A_tau : double
            the scale parameter of the half Cauchy prior of the global shrinkage parameter in the horseshoe-prior linear regression
        hs_lm_A_lambda : double
            the scale parameter of the half Cauchy prior of the local shrinkage parameter in the horseshoe-prior linear regression
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
                             device = self.device, 
                             track_loglik = False)
        
        theta = np.zeros((n, L))
        iterator = tqdm(range(n))
        for i in iterator:
            lm.fit(Psi, images[i, :])
            theta[i, :] = lm.coef_
                
        return(theta)
    
    
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
        
        
    