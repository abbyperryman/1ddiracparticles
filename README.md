# Bohmian Trajectories of a Dirac Electron in One Space Dimension

[![arXiv](https://img.shields.io/badge/arXiv-2512.21423-b31b1b.svg)](https://arxiv.org/abs/2512.21423)

Python code for simulating the trajectories of an electron in one space dimension, guided by a
spin-½ wave function that evolves under the free Dirac equation. The initial wave function is a
Gaussian wave packet, and the particle moves along the integral curves of the Bohmian velocity
field built from that wave function.

This code was used to generate the figures in the paper

> K. Narayanan, A. Perryman, and A. S. Tahvildar-Zadeh,
> **Asymptotic Momentum of Dirac Particles in One Space Dimension**,
> [arXiv:2512.21423](https://arxiv.org/abs/2512.21423).

<p align="center">
  <img src="docs/images/trajectories.png" width="32%" alt="Bohmian trajectories">
  <img src="docs/images/bloch_sphere.png" width="32%" alt="A trajectory on the Bloch sphere">
  <img src="docs/images/momentum_animation.gif" width="32%" alt="Momentum histogram animation">
</p>
<p align="center"><em>
Left: trajectories x(t). Middle: the spinor of one trajectory on the Bloch sphere.
Right: animated histogram of the momenta of 150 trajectories.
All generated with <code>k = 10, m = 1, σ = 1, θ = π/2, ω = 0</code>.
</em></p>

## Contents

- [Background](#background)
- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Parameters](#parameters)
- [Function reference](#function-reference)
- [Output files](#output-files)
- [Performance notes](#performance-notes)
- [Conventions and caveats](#conventions-and-caveats)
- [Citation](#citation)

## Background

Units are ħ = c = 1. The wave function is a two-component spinor ψ = (g, h) (called ψ₋ and ψ₊ in
the code) that solves the free Dirac equation in one space dimension,

$$(\partial_t + \partial_x)\, g = -i m\, h, \qquad (\partial_t - \partial_x)\, h = -i m\, g,$$

equivalently $i\partial_t \psi = (p\,\sigma_z + m\,\sigma_x)\,\psi$ with $p = -i\partial_x$, whose
energies are $\pm\sqrt{p^2+m^2}$.

**Initial data.** A Gaussian packet with mean momentum $k$ and width $\sigma$, with a spinor
described by two Bloch-sphere angles $(\theta, \omega)$:

$$\psi(x,0) = \begin{pmatrix}\cos(\theta/2)\,e^{i\omega/2}\\ \sin(\theta/2)\,e^{-i\omega/2}\end{pmatrix}
\frac{1}{(2\pi\sigma^2)^{1/4}}\, e^{-x^2/(4\sigma^2) + ikx}.$$

**Exact solution.** The solution is computed in closed form with Bessel kernels
(with $\rho = \sqrt{t^2-(x-r)^2}$ and both integrals over $r \in [x-t,\,x+t]$):

$$g(x,t) = g_0(x-t) - \frac{m}{2}\int g_0(r)\,\frac{J_1(m\rho)}{\rho}\,(t+x-r)\,dr
- \frac{im}{2}\int h_0(r)\,J_0(m\rho)\,dr,$$

and the same with $g_0 \leftrightarrow h_0$, $g_0(x-t)\to h_0(x+t)$ and $(t+x-r)\to(t-x+r)$ for $h$.

**Trajectories.** The particle's position solves $\dot{x} = j_1/j_0$, where
$j_0 = |g|^2+|h|^2$ is the probability density and $j_1 = |g|^2-|h|^2$ the probability current.
Initial positions are sampled from $j_0(x,0)$, which is a Gaussian with standard deviation $\sigma$.

**Bloch-sphere variables.** Writing $g = R\cos(\Theta/2)\,e^{i(\Phi-\Omega)/2}$ and
$h = R\sin(\Theta/2)\,e^{i(\Phi+\Omega)/2}$ gives the variables $(R, \Theta, \Omega, \Phi)$. The
velocity is $v = \cos\Theta$. The momentum and energy of a trajectory are read off from the
Bloch vector of a free plane-wave eigenspinor, $(m, 0, p)/E$:

$$p = m\cot\Theta\,\sec\Omega, \qquad E = \frac{m}{\sin\Theta\cos\Omega}.$$

**What the paper shows.** At large times the wave function is approximately a superposition of two
wave packets traveling in opposite directions. The trajectories then have approximately constant
asymptotic momentum $k$ and asymptotic energy $\pm\sqrt{m^2+k^2}$. The sign of the energy is
determined by the particle's initial position, and negative-energy particles move in the direction
opposite to their momentum. The simulations here let you watch this happen: for example, the
momentum histogram collapses onto $p \approx k$.

## Features

- **Trajectory simulation.** Draw N starting positions from the initial density (or supply your own)
  and evolve them under the exact Bohmian velocity field.
- **Velocity, momentum and energy plots.** Plot x(t), or the velocity, momentum, energy (and their
  signed-log versions) of each trajectory against time.
- **Momentum histograms and GIFs.** Histogram the momenta of N trajectories at every time step and
  save an animation. The canonical momentum distribution and its positive/negative-energy parts can
  be overlaid.
- **Bloch-sphere plots.** Trajectories on the Bloch sphere, final positions of an ensemble, and the
  time evolution of R, Θ, Ω and Φ.
- **Diagnostics.** Plots of the density, current and velocity field, a check that the total
  probability stays equal to 1, and an independent FFT solver to cross-check the closed-form
  solution.
- **Reproducible.** Every sampling function takes a `seed`, and every plot can be saved as a PNG
  with `save_image=True`.

## Installation

The code needs Python 3 and four packages: NumPy, SciPy, Matplotlib and Pillow (for GIFs). It was
developed with Python 3.12.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

## Quick start

1. Open `1d_dirac_particles_code.py` and set the physical parameters at the top of the file
   (see [Parameters](#parameters)):

   ```python
   k = 10             # central momentum of the packet
   m_el = 1           # electron mass
   theta = pi / 2     # initial Bloch-sphere polar angle
   omega = 0          # initial Bloch-sphere azimuthal angle
   sigma = 1          # width of the Gaussian
   ```

2. Call the functions you want, either at the bottom of the file under
   `if __name__ == "__main__":` or from your own script or Python session. Because the file name
   starts with a digit, `import 1d_dirac_particles_code` is a syntax error, so import it with
   `importlib`:

   ```python
   import importlib

   dirac = importlib.import_module("1d_dirac_particles_code")

   dirac.plotTrajectories(4, N=20, seed=0, save_image=True)  # 20 sampled trajectories up to t = 4
   dirac.plotTrajectories(4, x_init=[-1.0, 0.0, 1.0])        # or choose the starting points
   dirac.plotTrajectoryMomenta(4, N=10, seed=0)              # momentum of each trajectory vs. time
   dirac.momentum_animation(4, N=200, seed=0, frame_step=4)  # histogram GIF, saved to MomentumAnimations/
   dirac.plotOnBlochSphere(0.3, tmax=4)                      # one trajectory on the Bloch sphere
   ```

   Running `python 1d_dirac_particles_code.py` executes whatever is under the `__main__` guard.
   (Renaming the file to something that starts with a letter, such as `dirac_particles_1d.py`,
   would allow a plain `from dirac_particles_1d import ...`.)

The parameters, plot titles and file names are set when the module is imported, so after changing
a parameter, restart your Python session (or re-run the script) rather than reassigning it from
outside.

## Parameters

| Parameter | Meaning | Notes |
|---|---|---|
| `m_el` | Electron mass | Units with ħ = c = 1. |
| `k` | Mean momentum ("mean frequency" k₀) of the Gaussian packet | Either sign is supported. |
| `sigma` | Standard deviation σ of the initial Gaussian in position | The canonical momentum distribution has standard deviation 1/(2σ). |
| `theta` | Mixing angle θ, the polar Bloch angle of the initial spinor | Wrapped into [0, π]. |
| `omega` | Second initial Bloch angle ω | The initial Bloch azimuth is Ω₀ = −ω (see [caveats](#conventions-and-caveats)). |

The helper `eigenstate_angles(branch)` returns the `(theta, omega)` for which the initial spinor is
a free-particle energy eigenstate at momentum `k`: `branch=+1` for energy +√(k²+m²) and
`branch=-1` for −√(k²+m²). The titles label these cases as Θ₊ and Θ₋.

## Function reference

Starting positions can be given explicitly with `x_init`; if it is left out, `N` positions are
drawn from the initial density, optionally with a `seed`. Every plotting function takes
`save_image=False`.

### Trajectories

| Function | What it does |
|---|---|
| `plotTrajectories(t, x_init=None, N=20, seed=None)` | Plots x(t) for each trajectory and prints the mean final momentum and energy. |
| `plotSpecificTrajectories(t, xlist)` | Same plot for an explicit list of starting positions. |
| `plotTrajectoryVelocities(t, ...)` | Velocity (cos Θ) of each trajectory against time. |
| `plotTrajectoryMomenta(t, ...)` | Momentum of each trajectory against time. |
| `plotTrajectoryEnergies(t, ...)` | Energy of each trajectory against time. |
| `plotTrajectoryLogMomentum(t, ...)`, `plotTrajectoryLogEnergy(t, ...)` | Signed-log versions, sign(·)·log(\|·\|+1), for quantities that span orders of magnitude. |
| `rk4(x0=None, n=1000, t_max=5)` | One trajectory and its momentum, integrated with a hand-written RK4 loop. |

### Momentum histograms and animations

| Function | What it does |
|---|---|
| `momentum_animation(t, N=100, x_init=None, ...)` | Animated histogram of the momenta of N trajectories, saved as a GIF. |
| `animate_j0(t_max=3, n_frames=30, ..., save_gif=False)` | Animates the probability density j₀(x, t). |

`momentum_animation` has a fixed, wide x-axis (centered on `k` by default) so that you can follow
how the histogram moves. It overlays the canonical momentum distribution N·|φ(p)|², and with
`show_branches=True` its positive- and negative-energy parts. Useful options:

| Option | Default | Meaning |
|---|---|---|
| `method` | `"grid"` | `"grid"` integrates the trajectories on an FFT grid (fast); `"quad"` uses the Bessel-kernel solution with quadrature. |
| `p_range` | `None` | Fixed momentum axis `(lo, hi)`. By default the axis is computed from the trajectories. |
| `center_on_k` | `True` | Make the automatic axis symmetric about `k`. |
| `coverage`, `margin` | `0.95`, `0.1` | Fraction of the momenta the automatic axis must cover, and its padding. |
| `fixed_y` | `True` | Keep the y-axis fixed (`False` rescales each frame). |
| `frame_step` | `1` | Draw every n-th time step; larger values render faster and give shorter GIFs. |
| `fps`, `n_bins`, `seed` | `10`, `400`, `None` | Frame rate, number of histogram bins, random seed. |

### Bloch sphere

| Function | What it does |
|---|---|
| `plotOnBlochSphere(x0, tmax=4)` | Spinor of the trajectory starting at `x0`, drawn on the Bloch sphere and colored by time. |
| `plotFinalPosOnBlochSphere(t, N=100, seed=None)` | Final Bloch-sphere positions of N trajectories. |
| `plotTimeEvolutionofBlochVars(x0, t)` | R, Θ, Ω and Φ against time for the trajectory starting at `x0`. |

### Wave function and diagnostics

| Function | What it does |
|---|---|
| `plotProbabilityCurrentsAndVelocity(t)` | Plots j₀, j₁ and the velocity field v as functions of x at time t. |
| `plotWaveFunc(choice, t, N=200, seed=None)` | Scatter plot of Re ψ₋ (`choice=0`) or Re ψ₊ (`choice=1`) at time t. |
| `probability_check(t_max)` | Plots ∫ j₀ dx against time; it should stay equal to 1. |
| `verify_solution(t, x_target)` | Difference between the closed-form solution and the independent FFT solver. |

### Lower-level building blocks

`g1(x, t)`, `h1(x, t)` and `amplitudes(x, t)` give the exact spinor; `j0`, `j1`, `v`, `polar_decomp`,
`p` and `e` give the observables; `exact_spinor_grid(t)` is the FFT solver;
`grid_bohmian_trajectories(x_init, T)` is the fast trajectory integrator;
`canonical_momentum_density(q, branch=None)` and `positive_energy_weight(branch)` describe the
momentum-space distribution of the packet; `sample_initial_positions(N, seed)` draws starting
points.

## Output files

With `save_image=True`, each plot is saved as a PNG in a subdirectory next to the script. The
directories are created automatically, and file names include the parameters (`k`, `theta`, `omega`,
`m`, `sigma`) so runs with different parameters do not overwrite each other.

| Directory | Contents |
|---|---|
| `ElectronTrajectoriesImages/` | `plotTrajectories`, `plotSpecificTrajectories`, `rk4` |
| `VelocityImages/`, `MomentumImages/`, `EnergyImages/` | The velocity, momentum and energy plots, including the log versions |
| `BlochSphereImages/`, `BlochVariableImages/` | The Bloch-sphere plots |
| `WaveFunctionImages/`, `ProbabilityCurrentsVelocityImages/`, `ProbabilityCheckImages/` | The wave function and diagnostic plots |
| `MomentumAnimations/` | GIFs from `momentum_animation` (always saved) |
| `DensityAnimations/` | GIFs from `animate_j0(..., save_gif=True)` |

## Performance notes

- `momentum_animation` uses the fast grid integrator by default. For 100 trajectories up to t = 4
  the integration takes under a second, and most of the run time is spent rendering GIF frames
  (roughly 0.2 s per frame). Use `frame_step` to cut that.
- The other trajectory plots use adaptive quadrature of the closed-form solution, which takes on the
  order of seconds per trajectory at t = 4 with `k = 10`. Keep `N` modest for those.
- `method="quad"` in `momentum_animation` gives the same results as the default but is much slower.

## Conventions and caveats

- **Initial azimuth.** The first spinor component carries `exp(+iω/2)` and the second
  `exp(−iω/2)`, so the initial Bloch azimuth is Ω₀ = −ω. This only matters when ω is not 0 or π.
- **Initial momentum.** The momentum `p = m cot Θ sec Ω` is computed from the spinor, not from `k`.
  For a generic initial spinor it is *not* `k` at t = 0. With θ = π/2 and ω = 0, for example, every
  trajectory starts at p = 0 and approaches `k` only after a few time units. To start at `k`, set
  `theta, omega = eigenstate_angles(+1)`.
- **Singularities.** `p` and `E` diverge where cos Ω = 0, so individual trajectories can show large
  spikes at intermediate times. The histogram animation counts the trajectories that fall outside
  the plotted range in each frame.
- **Canonical vs. Bohmian momentum.** The dotted curve in the animation is the distribution of the
  canonical momentum, a Gaussian with mean `k` and standard deviation 1/(2σ) that does not change in
  time. It is a different quantity from the Bohmian momentum shown by the bars.
- **Domain size.** `exact_spinor_grid` and `grid_bohmian_trajectories` use a periodic grid that is
  sized automatically from `t` and `sigma`.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{narayanan2025asymptotic,
  title         = {Asymptotic Momentum of Dirac Particles in One Space Dimension},
  author        = {Narayanan, Kabir and Perryman, Abigail and Tahvildar-Zadeh, A. Shadi},
  year          = {2025},
  eprint        = {2512.21423},
  archivePrefix = {arXiv}
}
```

## Authors

Code by Abigail Perryman. The accompanying research is joint work with Kabir Narayanan and
A. Shadi Tahvildar-Zadeh.
