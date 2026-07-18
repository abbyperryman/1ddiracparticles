This is code for simulating the trajectories of an electron in one space dimension using the ODE that describes its motion. The parameters to specify are the mass m_el, the mean frequency k_0, the standard deviation of the initial Gaussian wave packet sigma, the mixing angle theta, and the other initial Bloch sphere angle omega. It includes: 

- Functions that select N initial posititions and lets these trajectories involve, plotting the trajectories or plotting the evolution of the velocity, momentum, or energy of each individual trajectory over time.
- Functions that generate a histogram of the momentum values for N trajectories at each point of time and create a gif.
- Various functions for plotting trajectories on the Bloch sphere and plotting the evolution of each of the Bloch variables.

It was used to generate the figures for the research presented in the paper "Asymptotic Momentum of Dirac Particles in One Space Dimension" by Kabir Narayanan, Abigail Perryman, and A. Shadi Tahvildar-Zadeh. https://arxiv.org/abs/2512.21423
