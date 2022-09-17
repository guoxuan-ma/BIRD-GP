from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'BIRD-GP'
LONG_DESCRIPTION = 'A package for the model Bayesian Image-on-image Regression via Deep kernel learning based Gaussian Processes (BIRD-GP) model'

setup(
        name = "BIRD-GP", 
        version = VERSION,
        author = "Guoxuan Ma",
        author_email = "<gxma@umich.com>",
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        packages = find_packages(),
        install_requires = ['numpy', 'torch', 'tdqm', 'scipy', 'matplotlib'],
        keywords = ['python', 'BIRD-GP'],
        classifiers = [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ],
        url = 
)