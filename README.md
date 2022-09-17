# Bayesian Image-on-image Regression via Deep kernel learning based Gaussian Processes (BIRD-GP)

This is the python package for Bayesian Image-on-image Regression via Deep kernel learning based Gaussian Processes (BIRD-GP). The package can be installed by
```
$ pip install git+https://github.com/guoxuan-ma/2022_BIRD_GP
```
The package contains a ```BIRD_GP``` class for the BIRD-GP model. As part of BIRD-GP, the kernel learning neural network for basis fitting ```BFNN```, the horseshoe-prior linear regression ```FastHorseshoeLM``` and the Bayesian neural network with Stein variational gradient descent ```svgd_bnn``` are also implemented as python class and can be used independently. We also provide a helper function for generating grids over images voxels ```generate_grids```.

An example of applying BIRD-GP on a synthesized Fashion MNIST dataset is included in the ```example``` folder. 