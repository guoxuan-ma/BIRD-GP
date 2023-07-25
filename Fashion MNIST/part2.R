library(BayesGPfit)
library(fastBayesReg)

L = commandArgs(trailingOnly = T)
L = as.numeric(L)

n = 2000
n_train = 1000
n_test = 1000

# load basis functions for predictors and outcomes (using only training predictors)
Psi_predictors = as.matrix(read.table("Psi_predictors.txt", sep = ","))
Psi_outcomes = as.matrix(read.table("Psi_outcomes.txt", sep = ","))


# load data
train_predictors = as.matrix(read.table("train_predictors.txt"))
train_outcomes = as.matrix(read.table("train_outcomes.txt"))
test_predictors = as.matrix(read.table("test_predictors.txt"))
test_outcomes = as.matrix(read.table("test_outcomes.txt"))
#grids = as.matrix(read.table("x.txt"))

# find coefficients for predictors
svd.decom = svd(Psi_predictors)
Psi_p = svd.decom$u
# eigen_p = (svd.decom$d)^2
# Psi_p = t(t(Psi_p) * eigen_p)
# qr.decom = qr(Psi_predictors)
# Psi_p = qr.Q(qr.decom)

# image(matrix(Psi_p[, 1], ncol = 28))
# i = 5
# image(matrix(train_predictors[i, ], ncol = 28))
# image(matrix((theta_train_predictors %*% t(Psi_p))[i, ], ncol = 28))

theta_train_predictors = matrix(0, nrow = n_train, ncol = L)
for (i in 1:n_train) {
  lm_fit = fast_horseshoe_lm(train_predictors[i, ], Psi_p)
  theta_train_predictors[i, ] = lm_fit$post_mean$betacoef
  cat(i)
}

theta_test_predictors = matrix(0, nrow = n_test, ncol = L)
for (i in 1:n_test) {
  lm_fit = fast_horseshoe_lm(test_predictors[i, ], Psi_p)
  theta_test_predictors[i, ] = lm_fit$post_mean$betacoef
  cat(i)
}


# find coefficients for outcomes
svd.decom = svd(Psi_outcomes)
Psi_o = svd.decom$u
# qr.decom = qr(Psi_outcomes)
# Psi_o = qr.Q(qr.decom)

theta_train_outcomes = matrix(0, nrow = n_train, ncol = L)
for (i in 1:n_train) {
  lm_fit = fast_horseshoe_lm(train_outcomes[i, ], Psi_o)
  theta_train_outcomes[i, ] = lm_fit$post_mean$betacoef
  cat(i)
}

theta_test_outcomes = matrix(0, nrow = n_test, ncol = L)
for (i in 1:n_test) {
  lm_fit = fast_horseshoe_lm(test_outcomes[i, ], Psi_o)
  theta_test_outcomes[i, ] = lm_fit$post_mean$betacoef
  cat(i)
}

write.table(theta_train_predictors, file = "theta_train_predictors.txt", col.names = F, row.names = F)
write.table(theta_test_predictors, file = "theta_test_predictors.txt", col.names = F, row.names = F)
write.table(theta_train_outcomes, file = "theta_train_outcomes.txt", col.names = F, row.names = F)
write.table(theta_test_outcomes, file = "theta_test_outcomes.txt", col.names = F, row.names = F)
write.table(Psi_p, file = "Psi_p.txt", col.names = F, row.names = F)
write.table(Psi_o, file = "Psi_o.txt", col.names = F, row.names = F)


# # run SVGD_bnn
# SVGD = SVGD_bayesian_nn(X_train = theta_train_predictors, 
#                         y_train = theta_train_outcomes, 
#                         X_test = theta_test_predictors, 
#                         y_test = theta_test_outcomes, 
#                         dev_split = 0.1,
#                         M = 20, 
#                         num_nodes = c(200, 200, L),
#                         a0 = c(1, 1),
#                         b0 = c(1, 1),
#                         initial_values = FALSE
# )
# SVGD = optimizer(SVGD,
#                  master_stepsize = 0.001,
#                  auto_corr = 0.9,
#                  max_iter = 500,
#                  batch_size = 50,
#                  method = 'adam',
#                  use_autodiff = F,
#                  tol = 1e-8,
#                  check_freq = 10)

# SVGD = development(SVGD)

# evaluation(SVGD, theta_test_predictors, theta_test_outcomes)

# y_train_pred = SVGD_bayesian_nn_predict(SVGD, theta_train_predictors)
# y_test_pred = SVGD_bayesian_nn_predict(SVGD, theta_test_predictors)

# train_image_pred = t(Psi_o %*% y_train_pred)
# test_image_pred = t(Psi_o %*% y_test_pred)

# # MSE
# mse = matrix(0, nrow = 1, ncol = 2)
# mse[1, 1] = mean((train_image_pred - train_outcomes)^2)
# mse[1, 2] = mean((test_image_pred - test_outcomes)^2)
# write.table(mse, file = "mse.txt", row.names = F, col.names = F)
# write.table(train_image_pred, file = "train_pred.txt", col.names = F, row.names = F)
# write.table(test_image_pred, file = "test_pred.txt", col.names = F, row.names = F)
