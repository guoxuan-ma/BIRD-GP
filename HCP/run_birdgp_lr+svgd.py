import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch
import pandas as pd
import sys

sys.path.append("/home/gxma/Image-on-image regression/real_data/birdgp_module")
import bird_gp
import utils

# experiment settings
mode_list = ["fALFF+confounder", "connectivity+confounder", "fALFF+connectivity+confounder"]

slurm_id = int(sys.argv[1]) - 1

#exp = int(np.floor(slurm_id / 3))
#mode_id = slurm_id % 3
exp = slurm_id
mode_id = 0

mode = mode_list[mode_id]
outcome_name = "socirand"

torch.random.manual_seed(exp)
random.seed(exp)
np.random.seed(exp)

# read, split and process data
outcomes = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/{}_fALFF-{}.npy".format(outcome_name, outcome_name))
outcomes[np.isnan(outcomes)] = 0
confounders = pd.read_csv("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/confounders.csv")
confounders = np.array(confounders.iloc[:, 2:])
mask = np.load("/home/gxma/Image-on-image regression/real_data/region_img.npy")
Psi_outcomes = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/fit_basis/{}_fALFF-{}_Psi_{}.npy".format(outcome_name, outcome_name, exp))
V = len(mask[mask > 0])
n = outcomes.shape[0]
n_train = int(n * 0.8)

split = np.zeros((2, n), dtype = bool)
train_idx = np.random.choice(np.arange(n), size = n_train, replace = False)
split[0, :] = np.isin(np.arange(n), train_idx)
split[1, :] = np.isin(np.arange(n), train_idx, invert = True)

if mode_id == 0:
    predictors = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/fALFF.npy")
    Psi_predictors = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/fit_basis/scaled_fALFF/fALFF_Psi_{}.npy".format(exp))
    predictors[np.isnan(predictors)] = 0
    predictors = predictors.astype(np.float32)
elif mode_id == 1:
    predictors = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/connectivity.npy")
    predictors = predictors.astype(np.float32)
    Psi_predictors = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/fit_basis/connectivity_Psi_{}.npy".format(exp))
elif mode_id == 2:
    fALFFs = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/fALFF.npy")
    connectivity = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/connectivity.npy")
    Psi_fALFFs = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/fit_basis/fALFF_Psi_{}.npy".format(exp))
    Psi_connectivity = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/fit_basis/connectivity_Psi_{}.npy".format(exp))
    fALFFs[np.isnan(fALFFs)] = 0
    fALFFs = fALFFs.astype(np.float32)
    connectivity = connectivity.astype(np.float32)
    predictors = np.hstack((fALFFs, connectivity))
    Psi_predictors = np.vstack((Psi_fALFFs, Psi_connectivity))

outcomes = outcomes.astype(np.float32)
confounders = confounders.astype(np.float32)

train_predictors = predictors[split[0, :], :]
train_outcomes = outcomes[split[0, :], :]
train_confouders = confounders[split[0, :], :]
test_predictors = predictors[split[1, :], :]
test_outcomes = outcomes[split[1, :], :]
test_confounders = confounders[split[1, :], :]

from sklearn import preprocessing
scaler_train = preprocessing.StandardScaler().fit(train_outcomes)
scaler_test = preprocessing.StandardScaler().fit(test_outcomes)
scaled_train_outcomes = scaler_train.transform(train_outcomes, copy = True)
scaled_test_outcomes_1 = scaler_train.transform(test_outcomes, copy = True)
scaled_test_outcomes_2 = scaler_test.transform(test_outcomes, copy = True)

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
    disable_message = True
)
model.train_outcomes = scaled_train_outcomes

from sklearn.linear_model import LinearRegression

Psi_predictors, _, _ = np.linalg.svd(Psi_predictors, full_matrices = False)
model.Psi_predictors = np.hstack((np.ones((Psi_predictors.shape[0], 1)), Psi_predictors))
Psi_outcomes, _, _ = np.linalg.svd(Psi_outcomes, full_matrices = False)
model.Psi_outcomes = np.hstack((np.ones((Psi_outcomes.shape[0], 1)), Psi_outcomes))

model.theta_train_predictors = np.zeros((n_train, model.Psi_predictors.shape[1]))
for i in range(n_train):
    lm = LinearRegression(fit_intercept = False).fit(model.Psi_predictors, train_predictors[i, :])
    model.theta_train_predictors[i, :] = lm.coef_

model.theta_train_outcomes = np.zeros((n_train, model.Psi_outcomes.shape[1]))
for i in range(n_train):
    lm = LinearRegression(fit_intercept = False).fit(model.Psi_outcomes, scaled_train_outcomes[i, :])
    model.theta_train_outcomes[i, :] = lm.coef_

model.theta_train_predictors = np.hstack((model.theta_train_predictors, train_confouders))
model.empirical_eigen_values_outcomes = np.var(model.theta_train_outcomes, 0)
segment_epochs = 100
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
    epochs = segment_epochs,
    M = 20,
    step_size = 5e-4
)

for segment in range(5):
    model.svgd_nn.train()
    
    test_pred = model.predict_test_lr(test_predictors, test_confounders)
    test_pred_back_by_train = scaler_train.inverse_transform(test_pred, copy = True)
    test_pred_back_by_test = scaler_test.inverse_transform(test_pred, copy = True)
    test_r2_img_1 = utils.voxelwise_r2(test_outcomes, test_pred_back_by_train, mask)
    test_r2_img_2 = utils.voxelwise_r2(test_outcomes, test_pred_back_by_test, mask)

    #np.save("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/{}_{}/train_r2_exp{}_epoch{}.npy".format(outcome_name, mode, exp, 200 * (segment + 1)),
    #       arr = train_r2_img)
    if segment > -1:
        np.save("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/{}_{}/test_r2_exp{}_epoch{}_2.npy".format(outcome_name, mode, exp, segment_epochs * (segment + 1)),
               arr = test_r2_img_2)
    # mode_id, outcome_name, segment_epochs, range(3/5), reduced1