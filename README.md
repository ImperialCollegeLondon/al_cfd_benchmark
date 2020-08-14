# active_learning_cfd
> Active learning based regression for CFD cases

## Requirements

This requires the following packages and their dependencies:
- *numpy*
- *matplotlib*
- *modAL*
- *PyFoam*

To run the examples, it is also needed:
- *sklearn*

## Installation

We recommend that the package be installed in development mode:
```
pip3 install -e .
```

## Usage

Examples cases are provided in the *example* folder.

The test cases presented on the article are available on the *cases* folder:
1. static_mixer
2. orifice
3. mixer
4. mixer3D

Each case is composed by a folder with the OpenFOAM template and a *runner* 
python script for running the case and extracting outputs.
The *regression_batch* scripts run a set of different strategies, with the 
possibility of repeating  each one several times for statistics. 
The *reference* scripts generate reference results for estimation of 
interpolation error.

## About

G. F. N. Gonçalves, A. Batchvarov, Y. Liu, Y. Liu, L. Mason, I. Pan, 
O. K. Matar (2020). Data-driven surrogate modelling and benchmarking 
for process equipment. Data-Centric Engineering. DOI: 10.1017/dce.2020.8