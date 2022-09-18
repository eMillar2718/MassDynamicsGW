# Mass Dynamics Reconstruction from Gravitational Wave Bursts
## Project Aims
- To understand how to use python machine learning software (PyTorch, GlasFlows)
- To be able to implement and develop a conditional Normalising Flow for the analysis of time-series data  
- To simulate gravitational wave burst mass dynamics and derive gravitational wave strain data from it 
- To provide physical insight through visualisations of the 3+1 dimensional outputs of the algorithm 

## Tasks Completed
- Added Noise to the model
- Optimised flow parameters with the noise
- Plotted SNR curves and Latent Space
- Written window function along with its first and second derivatives
- Created alternate strain function with the new mathematical expression for the strain including the windowed mass dynamics
- Fix mathematical error where the mass dynamics are not tapering to zero
- Apply mass dynamic windowing to 2D case
- Compute strain for h_xx strain 

## To-Do
- Plot reconstructed dynamics for 2D case
- Plot latent space of flow (fixing the error of repeated data of length of the sample size)
- Animate the mass dynamics in 2D
- Find a more suitable, continuous, window function for the mass dynamics

## Overview
This project is the exploration of unmodelled gravitational wave signals, we will investigate whether machine learning techniques can be used to reconstruct the mass dynamics of a given burst signal. We use a conditional normalising flow, a type of neural network that maps a complex distrubution to a simple distribution, for this purpose. The flow is trained on simulated burst signals, initially modelled using point masses in 1D and noise free strain signals to demonstrate the flows ability to perform this operation. Upon the succession of doing so, the next step is to adapt the model to include noise, and scale to higher dimensions. 

## Data Generation

### Mass Dynamics

Due to the assumptions we made, the expression we use to describe our mass dynamics (quadroplole moment) reduces to the following form:

$$Q(t) = \frac{1}{2} \sum_{i=1}^n m_i x_{i}^{2}(t)$$ Which demonstrates our mass dynamics is described by the position of our masses, and the value of the mass. We have chosen to generate the motiton of the masses using chebyshev polynomials

### Strain

The gravitational wave strain can then be generated analystically by taking the second derivative of the quadrupole moment, giving the following expression:

$$h(t) \approx \frac{d^2Q}{dt^2} = \sum_{i=1}^{n} m_i(\dot x_i^2(t) + x_i(t)\ddot x_i(t))$$

For the sake of faster data generation with little compromise to accuracy, we chose a sampling rate of 256Hz for the project so far.

