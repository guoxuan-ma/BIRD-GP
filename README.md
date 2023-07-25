# Bayesian Image-on-image Regression via Deep kernel learning based Gaussian Processes (BIRD-GP)

This is the repository for Bayesian Image-on-image Regression via Deep kernel learning based Gaussian Processes (BIRD-GP). The repository contains
* birdgp: a python module for BIRD-GP
* birdgp_example: a Fashion MNIST based example of using the BIRD_GP class in birdgp module
* MNIST: code for the sythetic data analysis based on the MNIST dataset (Section 4.1 in the manuscript)
* Fashion MNIST: code for the sythetic data analysis based on the MNIST dataset (Section 4.2 in the manuscript)
* HCP: code for the HCP fMRI analysis (Section 5 in the manusript)

For synthetic data analysis, the notebook ```digits_birdgp``` and ```fashion_birdgp``` use an earlier version of BIRD-GP implementation that separates Stage 1 and Stage 2. In the ```birdgp_example``` folder, we provide a Fashion MNIST based example of using the BIRD_GP class in birdgp module. 

### The birdgp module
The python module can be loaded by 
```
import sys
sys.path.append("birdgp")
import bird_gp
```

The module depends on ```numpy```, ```torch```, ```tdqm```, ```scipy```, ```matplotlib```, ```pandas``` and ```sklearn```, please make sure the dependencies are installed. 

As part of BIRD-GP, the kernel learning neural network for basis fitting ```BFNN```, the horseshoe-prior linear regression ```FastHorseshoeLM``` and the Bayesian neural network with Stein variational gradient descent ```svgd_bnn``` are also implemented as python class and can be used independently. We also provide a helper function for generating grids over images voxels ```generate_grids```.

An example of applying BIRD-GP on a synthesized Fashion MNIST dataset is included in the ```birdgp_example``` folder. 