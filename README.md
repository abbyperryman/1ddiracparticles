This is code for simulating the trajectories of an electron in one space dimension using the ODE that describes their motion, with the mass m_el, the mean frequency k_0, the standard deviation of the initial Gaussian wave packet sigma, the mixing angle theta, and the other initial Bloch sphere angle omega. 

The most important functions select N initial posititions and lets these trajectories involve, plotting the following for each individual trajectory (in different colors):
- plotTrajectores: Spacetime graphs
- plotTrajectoryVelocities: Velocity over time
- plotTrajectoryMomenta: Momentum over time
- plotTrajectoryEnergies: Energy over time

There's also the function momentum_animation, which generates a histogram of the momentum values for N trajectories at each point of time and creates a gif.

There are also various functions for plotting trajectories on the Bloch sphere and plotting the evolution of each of the Bloch variables.

It was used to generate the figures for the research presented in the paper "Asymptotic Momentum of Dirac Particles in One Space Dimension" by Kabir Narayanan, Abigail Perryman, and A. Shadi Tahvildar-Zadeh. 
