import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch
import pandas as pd
import sys

sys.path.append("/home/gxma/Image-on-image regression/real_data/birdgp_scripts")
import bird_gp
import utils

image_name_list = ["fALFF", 
                   "connectivity", 
                   "lang_fALFF-lang",
                   "socirand_fALFF-socirand"]

exp = int(sys.argv[1]) - 1
image_name = image_name_list[exp]

torch.random.manual_seed(exp)
random.seed(exp)
np.random.seed(exp)

images = np.load("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/{}.npy".format(image_name))
images[np.isnan(images)] = 0
images = images.astype(np.float32)
mask = np.load("/home/gxma/Image-on-image regression/real_data/region_img.npy")
V = len(mask[mask > 0])
mask_arr = (mask > 0).reshape(-1)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(images)
images = scaler.transform(images, copy = True)

if image_name == "connectivity":
    grids = np.loadtxt("/scratch/jiankang_root/jiankang0/gxma/IIR/data_connectivity+confouder/connectivity_grid.csv", delimiter = ",")
else:
    grids = bird_gp.generate_grids([91, 109, 91])
    grids = grids[mask_arr, :]

birdgp = bird_gp.BIRD_GP(
    predictor_grids = grids,
    outcome_grids = grids,
    predictor_L = 150,
    outcome_L = 150,
    bf_predictor_lr = 5e-4,
    bf_outcome_lr = 5e-4,
    bf_predictor_steps = 500000, 
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
Psi_images, mse_trace_images, r2_trace_images = birdgp.fit_basis(
    birdgp.grids_in, 
    images, 
    birdgp.bf_predictor_lr, 
    birdgp.bf_predictor_steps,
    birdgp.L_in
)
np.save(file = "/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/fit_basis/tmp/{}_mse_trace_all_for_importance".format(image_name), arr = mse_trace_images)
np.save(file = "/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/fit_basis/tmp/{}_r2_trace_all_for_importance".format(image_name), arr = r2_trace_images)
np.save(file = "/scratch/jiankang_root/jiankang0/gxma/IIR/jasa_results_connectivity+confounder/fit_basis/tmp/{}_Psi_all_for_importance".format(image_name), arr = Psi_images)