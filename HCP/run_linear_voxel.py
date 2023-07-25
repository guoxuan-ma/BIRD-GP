import numpy as np
import random, sys
import pandas as pd

sys.path.append("/home/gxma/Image-on-image regression/real_data/birdgp_scripts")
import utils

mode_list = ["fALFF", "fALFF+confounder", "fALFF+connectivity+confounder"]

slurm_id = int(sys.argv[1]) - 1
#exp = int(np.floor(slurm_id / 3))
#mode_id = slurm_id % 3
exp = slurm_id
mode_id = 2
mode = mode_list[mode_id]
outcome_name = "socirand"

random.seed(exp)
np.random.seed(exp)

predictors = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/fALFF.npy")
outcomes = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/{}_fALFF-{}.npy".format(outcome_name, outcome_name))
connectivity = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/connectivity.npy")
confounders = pd.read_csv("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/confounders.csv")
confounders = np.array(confounders.iloc[:, 2:])
confounders = np.hstack((confounders, connectivity))
predictors[np.isnan(predictors)] = 0
outcomes[np.isnan(outcomes)] = 0
mask = np.load("/home/gxma/Image-on-image regression/real_data/region_img.npy")
V = len(mask[mask > 0])
n = predictors.shape[0]
n_train = int(n * 0.8)

split = np.zeros((2, n), dtype = bool)
train_idx = np.random.choice(np.arange(n), size = n_train, replace = False)
split[0, :] = np.isin(np.arange(n), train_idx)
split[1, :] = np.isin(np.arange(n), train_idx, invert = True)

train_predictors = predictors[split[0, :], :]
train_outcomes = outcomes[split[0, :], :]
train_confouders = confounders[split[0, :], :]
test_predictors = predictors[split[1, :], :]
test_outcomes = outcomes[split[1, :], :]
test_confounders = confounders[split[1, :], :]

if mode_id == 0:
    vr = utils.VoxelImageRegression()
    vr.fit(train_predictors, train_outcomes)
    train_pred = vr.predict(train_predictors)
    test_pred = vr.predict(test_predictors)
elif mode_id > 0:
    vr = utils.VoxelImageRegression()
    vr.fit(train_predictors, train_outcomes, train_confouders)
    train_pred = vr.predict(train_predictors, train_confouders)
    test_pred = vr.predict(test_predictors, test_confounders)
    

train_r2_img = utils.voxelwise_r2(train_outcomes, train_pred, mask)
test_r2_img = utils.voxelwise_r2(test_outcomes, test_pred, mask)

#np.save("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/{}_{}/vr_train_r2_exp{}.npy".format(outcome_name, mode, exp), arr = train_r2_img)
#np.save("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/{}_{}/vr_test_r2_exp{}.npy".format(outcome_name, mode, exp), arr = test_r2_img)


if mode_id == 0:
    lr = utils.LinearImageRegression()
    lr.fit(train_predictors, train_outcomes)
    train_pred = lr.predict(train_predictors)
    test_pred = lr.predict(test_predictors)

    train_r2_img = utils.voxelwise_r2(train_outcomes, train_pred, mask)
    test_r2_img = utils.voxelwise_r2(test_outcomes, test_pred, mask)

    #np.save("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/{}_{}/lr_train_r2_exp{}.npy".format(outcome_name, mode, exp), arr = train_r2_img)
    #np.save("/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/{}_{}/lr_test_r2_exp{}.npy".format(outcome_name, mode, exp), arr = test_r2_img)