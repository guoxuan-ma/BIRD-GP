import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):
    def __init__(self, predictors, labels):
        self.labels = labels
        self.predictors = predictors

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        X = self.predictors[index, :]
        y = self.labels[index, :]
        return X, y
    
    
class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        '''
        self.body = nn.Sequential(
            nn.Linear(in_dim, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        '''
        self.body = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        '''
        self.body = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        '''

    def forward(self, x):
        y = self.body(x)
        return y

class svgd_bnn:
  def __init__(self, 
               X_train, 
               y_train, 
               var_vec_y,
               X_dev = None, 
               y_dev = None, 
               M = 20, 
               a_gamma = 1, 
               b_gamma = 0.1, 
               a_lambda = 1, 
               b_lambda = 0.1, 
               batch_size = 64, 
               epochs = 100, 
               step_size = 1e-3,
               ):
    self.n = X_train.shape[0]
    self.p_in = X_train.shape[1]
    self.p_out = y_train.shape[1]
    self.X_train = X_train
    self.y_train = y_train
    self.M = M

    self.a_gamma = a_gamma
    self.b_gamma = b_gamma
    self.a_lambda = a_lambda
    self.b_lambda = b_lambda
    self.batch_size = batch_size
    self.epochs = epochs
    self.step_size = step_size

    self.nn = NeuralNetwork(self.p_in, self.p_out)
    self.cumsum_vars = np.cumsum([p.numel() for p in self.nn.parameters() if p.requires_grad])
    self.cumsum_vars = np.concatenate(([0], self.cumsum_vars))
    self.num_vars = self.cumsum_vars[-1] + 2
    self.theta = np.zeros((self.M, self.num_vars))

    self.std_X_train = np.std(X_train, 0)
    self.std_X_train[self.std_X_train == 0] = 1
    self.mean_X_train = np.mean(X_train, 0)
    self.mean_y_train = np.mean(y_train, 0)
    self.std_y_train = np.std(y_train, 0)
    
    self.var_y_prod = np.prod(var_vec_y)
    self.inv_var_mat_y = np.diag(1 / var_vec_y)
    self.inv_var_mat_y = torch.tensor(self.inv_var_mat_y, dtype = torch.float32)

    self.dev = False
    if X_dev is not None:
      self.X_dev = X_dev
      self.dev = True
    if y_dev is not None:
      self.y_dev = y_dev
      
    # initilization
    for i in range(self.M):
      self.nn = NeuralNetwork(self.p_in, self.p_out)
      loggamma, loglambda = self.init_var()
      self.theta[i, :] = self.pack_weights(loggamma, loglambda)


  def train(self):
    X_train = self.normalization(self.X_train)
    y_train = self.y_train
    X_train = torch.tensor(X_train, dtype = torch.float32)
    y_train = torch.tensor(y_train, dtype = torch.float32)
    dataset = Dataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
    
    # training
    grad_theta = np.zeros((self.M, self.num_vars))

    t = 0
    m_t = 0
    v_t = 0
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    training_r2_list = []
    val_r2_list = []
    for epoch in range(self.epochs):
      for (idx, (X_batch, y_batch)) in enumerate(dataloader):
        X_batch = X_batch
        y_batch = y_batch
        for i in range(self.M):
          params = self.theta[i, :]
          self.unpack_weights(params) # load weigths into nn
          grad_theta[i, :] = self.batch_log_posterior_grad(X_batch, y_batch, params)
        
        kxy, dxkxy = self.svgd_kernel(h = -1)  
        grad_theta = (np.matmul(kxy, grad_theta) + dxkxy) / self.M

        # adam 
        t = t + 1
        g_t = grad_theta
        m_t = beta_1 * m_t + (1 - beta_1) * g_t
        v_t = beta_2 * v_t + (1 - beta_2) * (g_t ** 2)
        m_cap = m_t / (1 - (beta_1 ** t))
        v_cap = v_t / (1 - (beta_2 ** t))
        theta_prev = self.theta
        self.theta = self.theta + (self.step_size * m_cap) / (np.sqrt(v_cap) + epsilon)

      training_rmse, training_r2 = self.evaluation(self.X_train, self.y_train)
      training_r2_list.append(training_r2)
      if self.dev:
        val_rmse, val_r2 = self.evaluation(self.X_dev, self.y_dev)
        print("epoch: {} / {}, training rmse: {}, training r2: {}, validation rmse: {}, validation r2: {}".format(
            epoch + 1, self.epochs, round(training_rmse, 4), round(training_r2, 4), round(val_rmse, 4), round(val_r2, 4)
            ))
        val_r2_list.append(val_r2)
      else:
        print("epoch: {} / {}, training rmse: {}, training r2: {}".format(
            epoch + 1, self.epochs, round(training_rmse, 4), round(training_r2, 4)
            ))
    
    plt.plot(training_r2_list)
    if self.dev:
      plt.plot(val_r2_list)
    plt.show()
    
    self.loggamma = self.theta[:, -2]


  def init_var(self):
    loggamma = np.log(np.random.gamma(self.a_gamma, self.b_gamma))
    loglambda = np.log(np.random.gamma(self.a_lambda, self.b_lambda))
    return (loggamma, loglambda)


  def svgd_kernel(self, h = -1):
    sq_dist = pdist(self.theta)
    pairwise_dists = squareform(sq_dist)**2
    if h < 0:
      h = np.median(pairwise_dists)  
      h = np.sqrt(0.5 * h / np.log(self.theta.shape[0]+1))
    
    Kxy = np.exp( -pairwise_dists / h**2 / 2)

    dxkxy = -np.matmul(Kxy, self.theta)
    sumkxy = np.sum(Kxy, axis=1)
    for i in range(self.theta.shape[1]):
        dxkxy[:, i] = dxkxy[:,i] + np.multiply(self.theta[:,i],sumkxy)
    dxkxy = dxkxy / (h**2)
    return (Kxy, dxkxy)

  
  def pack_weights(self, loggamma, loglambda):
    param = np.empty(0)
    for p in self.nn.parameters():
      p_vec = p.detach().numpy().reshape(-1)
      param = np.concatenate((param, p_vec))
    param = np.concatenate((param, [loggamma], [loglambda]))
    return param


  def unpack_weights(self, param):
    for i, p in enumerate(self.nn.parameters()):
      value = torch.tensor(param[self.cumsum_vars[i]:self.cumsum_vars[i+1]], dtype = torch.float32).reshape(p.shape)
      p.data = value


  def normalization(self, X, y = None):
    X = (X - np.full(X.shape, self.mean_X_train)) / np.full(X.shape, self.std_X_train)
    if y is not None:
      y = (y - np.full(y.shape, self.mean_y_train)) / np.full(y.shape, self.std_y_train)
      return (X, y)  
    else:
      return X


  def batch_log_posterior_grad(self, X_batch, y_batch, para_m):
    log_gamma = torch.autograd.Variable(
        torch.tensor(para_m[-2].copy(), dtype = torch.float32), 
        requires_grad = True)
    log_lambda = torch.autograd.Variable(
        torch.tensor(para_m[-1].copy(), dtype = torch.float32), 
        requires_grad = True)
    
    yhat = self.nn(X_batch)

    sum_of_squares = torch.zeros(1)
    for p in self.nn.parameters():
      sum_of_squares = sum_of_squares + torch.sum(torch.square(p))
    
    diff = yhat - y_batch
    #log_lik_data = -0.5 * self.p_out * self.batch_size * (np.log(2 * np.pi) - log_gamma) - (torch.exp(log_gamma) / 2) * torch.sum(torch.square(yhat - y_batch))
    log_lik_data = - 0.5 * self.p_out * self.batch_size * (np.log(2 * np.pi) - log_gamma) \
                   - 0.5 * self.batch_size * np.log(self.var_y_prod) \
                   - 0.5 * torch.exp(log_gamma) * torch.trace(diff @ self.inv_var_mat_y @ diff.T)
    log_prior_data = (self.a_gamma - 1) * log_gamma - self.b_gamma * torch.exp(log_gamma) + log_gamma
    log_prior_w = - 0.5 * (self.num_vars - 2) * (np.log(2 * np.pi) - log_lambda) - (torch.exp(log_lambda) / 2) * sum_of_squares  \
                    + (self.a_lambda - 1) * log_lambda - self.b_lambda * torch.exp(log_lambda) + log_lambda
    log_posterior = (log_lik_data * self.n / self.batch_size + log_prior_data + log_prior_w)

    self.nn_zero_grad()
    if log_gamma.grad is not None:
      log_gamma.grad.data.zero_()
    if log_lambda.grad is not None:
      log_lambda.grad.data.zero_()

    log_posterior.backward()

    gradient = np.empty(0)
    for p in self.nn.parameters():
      g_vec = p.grad.data.reshape(-1)
      gradient = np.concatenate((gradient, g_vec))
    gradient = np.concatenate((gradient, [log_gamma.grad.data], [log_lambda.grad.data]))

    return gradient


  def nn_zero_grad(self):
    for p in self.nn.parameters():
      if p.grad is not None:
        p.grad.data.zero_()
  

  def predict(self, X_test):
    # normalization
    n_test = X_test.shape[0]
    X_test = self.normalization(X_test)
    X_test_tensor = torch.tensor(X_test, dtype = torch.float32)

    # average over the output
    pred_y_test = np.zeros([self.M, n_test, self.p_out])

    with torch.no_grad():
      for i in range(self.M):
        self.unpack_weights(self.theta[i, :])
        pred_y_test[i, :, :] = self.nn(X_test_tensor).numpy() #* np.full((n_test, self.p_out), self.std_y_train) + np.full((n_test, self.p_out), self.mean_y_train)
    pred = np.mean(pred_y_test, axis = 0)

    return (pred, pred_y_test)
    
  
  def evaluation(self, X_test, y_test):
    pred, _ = self.predict(X_test)

    SSR = np.sum((y_test - pred) ** 2)
    SST = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - SSR / SST

    rmse = np.sqrt(np.mean((pred - y_test)**2))
    
    return rmse, r2
  
  
  def development(self):
    X_dev = self.normalization(self.X_dev)
    X_dev = torch.tensor(X_dev, dtype = torch.float32)
    def f_log_lik(loggamma): 
          return np.sum(  np.log(np.sqrt(np.exp(loggamma)) /np.sqrt(2*np.pi) * np.exp( -1 * (np.power(pred_y_dev - self.y_dev, 2) / 2) * np.exp(loggamma) )) )
    with torch.no_grad():
      for i in range(self.M):
        loggamma = self.theta[i, -2]
        self.unpack_weights(self.theta[i, :])
        pred_y_dev = self.nn(X_dev).numpy() #* np.full(self.y_dev.shape, self.std_y_train) + np.full(self.y_dev.shape, self.mean_y_train)
        lik1 = f_log_lik(loggamma)
        loggamma = -np.log(np.mean(np.power(pred_y_dev - self.y_dev, 2)))
        lik2 = f_log_lik(loggamma)
        if lik2 > lik1:
          self.theta[i, -2] = loggamma  # update loggamma