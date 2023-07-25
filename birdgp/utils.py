import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from tqdm import tqdm

class VoxelImageRegression:
    def __init__(self):
        pass

    def fit(self, X, Y, confounders = None, lasso = False):
        self.p = X.shape[1]
        self.models = []
        for j in tqdm(range(self.p)):
            x, y = X[:, j].reshape((-1, 1)), Y[:, j]
            if confounders is not None:
                x = np.hstack((x, confounders))
            model = LinearRegression()
            if lasso:
                model = Lasso(alpha = 1e-3)
            else:
                model = LinearRegression()
            model.fit(x, y)
            self.models.append(model)
    
    def predict(self, newX, newConfounders = None):
        newY = []
        for j in range(self.p):
            newx = newX[:, j].reshape((-1, 1))
            if newConfounders is not None:
                newx = np.hstack((newx, newConfounders))
            newY.append(self.models[j].predict(newx))
        newY = np.stack(newY, axis = 1)
        return newY

    
class LinearImageRegression:
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        self.n = X.shape[0]
        self.p = X.shape[1]
        intercepts = np.zeros(self.n)
        betas = np.zeros(self.n)
        for i in range(self.n):
            x, y = X[i, :].reshape((-1, 1)), Y[i, :]
            model = LinearRegression()
            model.fit(x, y)
            intercepts[i], betas[i] = model.intercept_, model.coef_
        self.intercept = intercepts.mean()
        self.beta = betas.mean()
    
    def predict(self, newX):
        return newX * self.beta + self.intercept

    
def restore_images(images_arr, mask):
    n = images_arr.shape[0]
    restored_images = np.zeros((n, 91, 109, 91))
    for i in range(n):
        restored_images[i, :][mask > 0] = images_arr[i, :]
    return restored_images


def voxelwise_cor_by_region(true_arr, pred_arr, region_mask):
    n = true_arr.shape[0]
    V = true_arr.shape[1]
    cor_arr = np.zeros((1, V))
    for v in range(V):
        if (np.std(true_arr[:, v]) < 1e-6) or (np.std(pred_arr[:, v]) < 1e-6):
            continue
        else:
            cor_arr[0, v] = np.corrcoef(np.vstack((true_arr[:, v], pred_arr[:, v])))[0, 1]
    cor_img = restore_images(cor_arr, region_mask)
    cor_img = cor_img.reshape((91, 109, 91))
    
    region_numbers = np.unique(region_mask[region_mask > 0])
    region_cor = pd.Series(index = region_numbers, dtype = np.float64)
    for region_number in region_numbers:
        if region_number == 0:
            continue
        cor_subset = np.mean(cor_img[region_mask == region_number])
        cor_subset = cor_subset[cor_subset != 0]
        avg_cor = np.mean(cor_subset)
        region_cor[region_number] = avg_cor
    return cor_img, region_cor


def voxelwise_mse_by_region(true_arr, pred_arr, region_mask):
    n = true_arr.shape[0]
    V = true_arr.shape[1]
    mse_arr = np.mean((true_arr - pred_arr) ** 2, axis = 0).reshape((1, -1))
    mse_img = restore_images(mse_arr, region_mask)
    mse_img = mse_img.reshape((91, 109, 91))
    
    region_numbers = np.unique(region_mask[region_mask > 0])
    region_mse = pd.Series(index = region_numbers, dtype = np.float64)
    for region_number in region_numbers:
        if region_number == 0:
            continue
        mse_subset = np.mean(mse_img[region_mask == region_number])
        mse_subset = mse_subset[mse_subset != 0]
        avg_mse = np.mean(mse_subset)
        region_mse[region_number] = avg_mse
    return mse_img, region_mse


def mse_by_region(true_imgs, pred_imgs, region_mask):
    n = true_imgs.shape[0]
    region_numbers = np.unique(region_mask[region_mask > 0])
    region_mse = pd.Series(index = region_numbers, dtype = np.float64)
    for region_number in region_numbers:
        num_voxel_in_region = np.sum(region_mask == region_number)
        outcomes_region = np.zeros((n, num_voxel_in_region))
        pred_outcome_region = np.zeros((n, num_voxel_in_region))
        for i in range(n):
            outcomes_region[i, :] = true_imgs[i, region_mask == region_number]
            pred_outcome_region[i, :] = pred_imgs[i, region_mask == region_number]
        region_mse[region_number] = np.mean((outcomes_region - pred_outcome_region)**2)
    return region_mse


def voxelwise_r2(true_arr, pred_arr, region_mask):
    n = true_arr.shape[0]
    V = true_arr.shape[1]
    mse_arr = np.mean((true_arr - pred_arr) ** 2, axis = 0).reshape((1, -1))
    mse_img = restore_images(mse_arr, region_mask).reshape((91, 109, 91))
    var_img = restore_images(np.var(true_arr, 0).reshape((1, -1)), region_mask).reshape((91, 109, 91))
    r2_img = np.zeros((91, 109, 91))
    undefined = var_img == 0
    defined = np.invert(undefined)
    r2_img[undefined] = np.nan
    r2_img[defined] = 1 - mse_img[defined] / var_img[defined]
    return r2_img


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def r2_over_thre_by_region(voxelwise_r2, r2_thre, defined_voxel, mask):
    r2_on_defined_voxel = voxelwise_r2[defined_voxel]
    r2_larger_thre_on_defined_voxel = r2_on_defined_voxel >= r2_thre
    region_on_defined_voxel = mask[defined_voxel]
    region_r2_df = pd.DataFrame(np.hstack([region_on_defined_voxel.reshape((-1, 1)), r2_on_defined_voxel.reshape((-1, 1)), r2_larger_thre_on_defined_voxel.reshape((-1, 1))]), columns = ["region", "r2", "over_thre"])
    region_r2_over_thre_df = region_r2_df[region_r2_df["over_thre"] == 1].iloc[:, [0, 1]]
    results = pd.concat([region_r2_df[["region", "over_thre"]].groupby("region").mean(), region_r2_over_thre_df.groupby("region").agg([np.mean, np.median, percentile(25), percentile(75)])], axis = 1)
    results = results.sort_values(by = "over_thre", ascending = False)
    return results


def d_log_posterior_grad_d_basis(model, m):
    para_m = model.svgd_nn.theta[m, :]
    model.svgd_nn.unpack_weights(para_m)
    log_gamma = torch.autograd.Variable(
        torch.tensor(para_m[-2].copy(), dtype = torch.float32), 
        requires_grad = True)
    log_lambda = torch.autograd.Variable(
        torch.tensor(para_m[-1].copy(), dtype = torch.float32), 
        requires_grad = True)
    x = torch.tensor(model.theta_train_predictors, dtype = torch.float32)
    y = torch.tensor(model.theta_train_outcomes, dtype = torch.float32)
    x.requires_grad = True

    yhat = model.svgd_nn.nn(x)
    sum_of_squares = torch.zeros(1)
    for p in model.svgd_nn.nn.parameters():
        sum_of_squares = sum_of_squares + torch.sum(torch.square(p))

    diff = yhat - y
    log_lik_data = - 0.5 * model.svgd_nn.p_out * model.svgd_nn.n * (np.log(2 * np.pi) - log_gamma) \
                   - 0.5 * model.svgd_nn.n * np.log(model.svgd_nn.var_y_prod) \
                   - 0.5 * torch.exp(log_gamma) * torch.trace(diff @ model.svgd_nn.inv_var_mat_y @ diff.T)
    log_prior_data = (model.svgd_nn.a_gamma - 1) * log_gamma - model.svgd_nn.b_gamma * torch.exp(log_gamma) + log_gamma
    log_prior_w = - 0.5 * (model.svgd_nn.num_vars - 2) * (np.log(2 * np.pi) - log_lambda) - (torch.exp(log_lambda) / 2) * sum_of_squares  \
                    + (model.svgd_nn.a_lambda - 1) * log_lambda - model.svgd_nn.b_lambda * torch.exp(log_lambda) + log_lambda
    log_posterior = (log_lik_data + log_prior_data + log_prior_w)

    model.svgd_nn.nn_zero_grad()
    if log_gamma.grad is not None:
        log_gamma.grad.data.zero_()
    if log_lambda.grad is not None:
        log_lambda.grad.data.zero_()
    if x.grad is not None:
        x.grad.data.zero_()

    log_posterior.backward()

    return x.grad.data.numpy()


def basis_importance(model):
    grad_basis = np.zeros((model.svgd_nn.M, model.svgd_nn.n, model.theta_train_predictors.shape[1]))
    for i in range(model.svgd_nn.M):
        params = model.svgd_nn.theta[i, :]
        model.svgd_nn.unpack_weights(params)
        grad_basis[i, :] = d_log_posterior_grad_d_basis(model, i)
    return grad_basis