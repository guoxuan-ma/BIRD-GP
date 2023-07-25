import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch
import pandas as pd
import sys
import pickle

sys.path.append("/home/gxma/Image-on-image regression/real_data/birdgp_scripts")
import bird_gp_1 as bird_gp
import utils

# experiment settings
mode_list = ["fALFF+confounder", "connectivity+confounder", "fALFF+connectivity+confounder"]

exp = int(sys.argv[1]) - 1
mode = mode_list[exp]
outcome_name = "lang"

torch.random.manual_seed(exp)
random.seed(exp)
np.random.seed(exp)

# read, split and process data
outcomes = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/{}_fALFF-{}.npy".format(outcome_name, outcome_name))
outcomes[np.isnan(outcomes)] = 0
confounders = pd.read_csv("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/confounders.csv")
confounders = np.array(confounders.iloc[:, 2:])
mask = np.load("/home/gxma/Image-on-image regression/real_data/region_img.npy")
Psi_outcomes = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/fit_basis/{}_fALFF-{}_Psi_all_for_importance.npy".format(outcome_name, outcome_name))
V = len(mask[mask > 0])
n = outcomes.shape[0]

if exp == 0:
    predictors = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/fALFF.npy")
    Psi_predictors = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/fit_basis/tmp/fALFF_Psi_all_for_importance.npy")
    predictors[np.isnan(predictors)] = 0
    predictors = predictors.astype(np.float32)
elif exp == 1:
    predictors = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/connectivity.npy")
    predictors = predictors.astype(np.float32)
    Psi_predictors = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/fit_basis/connectivity_Psi_all_for_importance.npy")
elif exp == 2:
    fALFFs = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/fALFF.npy")
    connectivity = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/connectivity.npy")
    Psi_fALFFs = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/fit_basis/fALFF_Psi_all_for_importance.npy")
    Psi_connectivity = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/fit_basis/connectivity_Psi_all_for_importance.npy")
    fALFFs[np.isnan(fALFFs)] = 0
    fALFFs = fALFFs.astype(np.float32)
    connectivity = connectivity.astype(np.float32)
    predictors = np.hstack((fALFFs, connectivity))
    Psi_predictors = np.vstack((Psi_fALFFs, Psi_connectivity))

outcomes = outcomes.astype(np.float32)
confounders = confounders.astype(np.float32)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(outcomes)
scaled_outcomes = scaler.transform(outcomes, copy = True)

# run model
mask_arr = (mask > 0).reshape(-1)
grids = bird_gp.generate_grids([91, 109, 91])
grids = grids[mask_arr, :]
model = bird_gp.BIRD_GP(
    predictor_grids = grids,
    outcome_grids = grids,
    predictor_L = 150,
    outcome_L = 150,
    bf_predictor_lr = 5e-4,
    bf_predictor_steps = 500000, 
    bf_outcome_lr = 5e-4,
    bf_outcome_steps = 500000, 
    hs_lm_mcmc_burnin = None,
    hs_lm_mcmc_samples = None, 
    hs_lm_thinning = None, 
    hs_lm_a_sigma = None, 
    hs_lm_b_sigma = None, 
    hs_lm_A_tau = None, 
    hs_lm_A_lambda = None,
    svgd_num_particles = None,
    svgd_a_gamma = None, 
    svgd_b_gamma = None, 
    svgd_a_lambda = None, 
    svgd_b_lambda = None, 
    svgd_batch_size = None, 
    svgd_epochs = None,
    device = "cuda",
    disable_message = False
)
model.train_outcomes = scaled_outcomes

from sklearn.linear_model import LinearRegression

Psi_predictors, _, _ = np.linalg.svd(Psi_predictors, full_matrices = False)
model.Psi_predictors = np.hstack((np.ones((Psi_predictors.shape[0], 1)), Psi_predictors))
Psi_outcomes, _, _ = np.linalg.svd(Psi_outcomes, full_matrices = False)
model.Psi_outcomes = np.hstack((np.ones((Psi_outcomes.shape[0], 1)), Psi_outcomes))

model.theta_train_predictors = np.zeros((n, model.Psi_predictors.shape[1]))
for i in range(n):
    lm = LinearRegression(fit_intercept = False).fit(model.Psi_predictors, predictors[i, :])
    model.theta_train_predictors[i, :] = lm.coef_

model.theta_train_outcomes = np.zeros((n, model.Psi_outcomes.shape[1]))
for i in range(n):
    lm = LinearRegression(fit_intercept = False).fit(model.Psi_outcomes, scaled_outcomes[i, :])
    model.theta_train_outcomes[i, :] = lm.coef_

model.theta_train_predictors = np.hstack((model.theta_train_predictors, confounders))
model.empirical_eigen_values_outcomes = np.var(model.theta_train_outcomes, 0)
model.svgd_nn = bird_gp.svgd.svgd_bnn(
    X_train = model.theta_train_predictors, 
    y_train = model.theta_train_outcomes,
    var_vec_y = model.empirical_eigen_values_outcomes,
    #var_vec_y = np.ones(model.empirical_eigen_values_outcomes.shape[0]),
    a_gamma = 1, 
    b_gamma = 1, 
    a_lambda = 0.01,
    b_lambda = 0.01, 
    batch_size = 64, 
    epochs = 300,
    M = 20,
    step_size = 5e-4
)

model.svgd_nn.train()

with open("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/{}_importance/model_{}.pickle".format(outcome_name, mode), "wb") as f:
    pickle.dump(model, f, -1)