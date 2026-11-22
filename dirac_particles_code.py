#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bohmian trajectories for a 1+1D Dirac electron with a Gaussian initial packet.
Units: hbar = c = 1.

Created on Mon Jun 20 18:51:39 2022
@author: abby

Layout
------
1. Parameters
2. Initial data and exact time-dependent solution (g1, h1)
3. Observables: densities, velocity field, Bloch variables, momentum, energy
4. Plot labels / output paths
5. Plotting and animation functions
6. Entry point
"""

import cmath
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scipystats
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy import absolute, arctan2, cos, exp, pi, sin, sqrt, tan
from scipy.integrate import odeint, quad
from scipy.special import jv

# =============================================================================
# 1. Parameters
# =============================================================================
k = 50                  # central momentum of the packet
m_el = 1                # electron mass
theta = pi / 2          # initial Bloch-sphere polar angle
omega = 0               # initial Bloch-sphere azimuthal angle
sigma = 1               # width of the Gaussian

# Wrap theta into [0, pi]
if theta < 0:
    theta += pi
if theta > pi:
    theta -= pi


# =============================================================================
# 2. Initial data and exact solution
#    g = psi_-, h = psi_+  (the two spinor components)
# =============================================================================
def gauss(x):
    return (2.0 * pi * sigma**2) ** (-1 / 4) * exp(-x**2 / (4 * sigma**2) + 1j * k * x)


def g(x):
    return cos(theta / 2) * gauss(x) * exp(1j * omega / 2)


def h(x):
    return sin(theta / 2) * gauss(x) * exp(-1j * omega / 2)


def besselfuncfrac(X):
    """J_1(m X) / X, with the X -> 0 limit handled explicitly."""
    if absolute(X) < 1e-8:
        return m_el / 2
    return jv(1, m_el * X) / X


def _scalar(x):
    """odeint hands us length-1 arrays; quad and cmath want plain floats."""
    return float(np.squeeze(x))


def _complex_quad(f, a, b):
    """scipy.quad can't integrate complex functions, so do Re and Im separately."""
    re = quad(lambda r: f(r).real, a, b)[0]
    im = quad(lambda r: f(r).imag, a, b)[0]
    return re + 1j * im


def _rho(x, t, r):
    """sqrt(t^2 - (x - r)^2), clamped at 0 to survive round-off at the light cone."""
    return sqrt(max(t**2 - (x - r) ** 2, 0.0))


def _evolve(x, t, own, other, sign):
    """Integral part of the solution for one spinor component.

    sign = +1 for g (psi_-), -1 for h (psi_+).
    """
    lo, hi = x - t, x + t
    t1 = _complex_quad(
        lambda r: own(r) * besselfuncfrac(_rho(x, t, r)) * (t + sign * (x - r)), lo, hi
    )
    t2 = _complex_quad(lambda r: other(r) * jv(0, m_el * _rho(x, t, r)), lo, hi)
    return -0.5 * m_el * t1 - 0.5j * m_el * t2


def g1(x, t):
    """Time-dependent psi_-."""
    x, t = _scalar(x), _scalar(t)
    if t == 0:
        return g(x)
    return g(x - t) + _evolve(x, t, g, h, +1)


def h1(x, t):
    """Time-dependent psi_+."""
    x, t = _scalar(x), _scalar(t)
    if t == 0:
        return h(x)
    return h(x + t) + _evolve(x, t, h, g, -1)


def amplitudes(x, t):
    """Both components at once, so callers never recompute the integrals."""
    return g1(x, t), h1(x, t)


# =============================================================================
# 3. Observables
# =============================================================================
def j0(x, t):
    """Probability density."""
    gg, hh = amplitudes(x, t)
    return absolute(gg) ** 2 + absolute(hh) ** 2


def j1(x, t):
    """Probability current."""
    gg, hh = amplitudes(x, t)
    return absolute(gg) ** 2 - absolute(hh) ** 2


def v(x, t):
    """Bohmian velocity field j1 / j0."""
    gg, hh = amplitudes(x, t)
    a, b = absolute(gg) ** 2, absolute(hh) ** 2
    return float((a - b) / (a + b))


def polar_decomp(x, t):
    """Return (R, Theta, Omega, Phi), the Bloch-sphere variables of the spinor."""
    gg, hh = amplitudes(x, t)
    r1, f1 = cmath.polar(gg)
    r2, f2 = cmath.polar(hh)
    R = sqrt(r1**2 + r2**2)
    Theta = 2 * arctan2(r2, r1)   # equals 2*arctan(r2/r1), but safe when r1 == 0
    Omega = f2 - f1
    Phi = f1 + f2
    return R, Theta, Omega, Phi


def p(x, t):
    """Bohmian momentum."""
    _, Theta, Omega, _ = polar_decomp(x, t)
    return m_el / (tan(Theta) * cos(Omega))


def e(x, t):
    """Bohmian energy."""
    _, Theta, Omega, _ = polar_decomp(x, t)
    return m_el / (sin(Theta) * cos(Omega))


def limit(s, t):
    a, b = (absolute(amp) for amp in amplitudes(s, t))
    return (m_el / 2) * (a / b - b / a)


def sample_initial_positions(N, seed=None):
    """Draw N starting positions from |psi(x, 0)|^2 = Normal(0, sigma).

    Both spinor components share the same Gaussian envelope, so the density is
    Normal(0, sigma) for any theta and omega.  Pass `seed` for reproducibility.
    """
    return np.random.default_rng(seed).normal(0, sigma, N)


def _initial_positions(x_init, N, seed):
    """Explicit starting positions if given, otherwise N samples from |psi(x, 0)|^2."""
    if x_init is None:
        return sample_initial_positions(N, seed)
    return np.asarray(x_init, dtype=float)


def check_starting_omega(x_init=None, N=10, seed=None):
    return [polar_decomp(x, 0)[2] for x in _initial_positions(x_init, N, seed)]


# ---- Exact grid solver (fast, independent of the Bessel-kernel solution) -----
def exact_spinor_grid(t, half_width=None, dx=None):
    """Exact free evolution on an FFT grid (independent of the Bessel-kernel solution).

    In momentum space psi(p, t) = exp(-i H(p) t) psi(p, 0), with
    H(p) = p*sigma_z + m*sigma_x acting on (g, h).  Returns (x, g, h) arrays.
    The grid is periodic, so half_width must exceed |t| + ~8*sigma.
    """
    if half_width is None:
        half_width = absolute(t) + 10 * sigma + 5
    if dx is None:
        dx = pi / (32 * (absolute(k) + 10 / sigma))  # 8x oversampled: p(x) varies fast near cos(Omega) = 0
    n = 2 ** int(np.ceil(np.log2(2 * half_width / dx)))
    x = (np.arange(n) - n // 2) * (2 * half_width / n)
    p_axis = 2 * pi * np.fft.fftfreq(n, d=x[1] - x[0])
    gk, hk = np.fft.fft(g(x)), np.fft.fft(h(x))
    E = sqrt(p_axis**2 + m_el**2)
    c, s = cos(E * t), sin(E * t) / E
    g_t = np.fft.ifft(c * gk - 1j * s * (p_axis * gk + m_el * hk))
    h_t = np.fft.ifft(c * hk - 1j * s * (m_el * gk - p_axis * hk))
    return x, g_t, h_t


def verify_solution(t=1.0, x_target=0.3):
    """Max |difference| between the Bessel-kernel solution and the grid solver."""
    x, gg, hh = exact_spinor_grid(t)
    i = int(np.argmin(absolute(x - x_target)))
    g_q, h_q = amplitudes(x[i], t)
    return max(absolute(g_q - gg[i]), absolute(h_q - hh[i]))


def grid_bohmian_trajectories(x_init, T, dt_max=0.0125):
    """Fast Bohmian trajectories and momenta on the exact FFT grid (no quadrature).

    Integrates dx/dt = v(x, t) with RK4 and reads the spinor off the exact free-evolution
    solution by interpolation.  The fast carrier exp(i*k*x) is divided out before
    interpolating, so the interpolation error is tiny.

    T must start at 0.  Returns (X, P): arrays of shape (len(T), len(x_init)) holding the
    positions and the Bohmian momenta p = m*cot(Theta)*sec(Omega) at every time in T.
    """
    T = np.asarray(T, dtype=float)
    X = np.asarray(x_init, dtype=float).copy()

    half_width = T[-1] + 10 * sigma + 5
    dx = pi / (32 * (absolute(k) + 10 / sigma))
    n = 2 ** int(np.ceil(np.log2(2 * half_width / dx)))
    xg = (np.arange(n) - n // 2) * (2 * half_width / n)
    p_axis = 2 * pi * np.fft.fftfreq(n, d=xg[1] - xg[0])
    gk, hk = np.fft.fft(g(xg)), np.fft.fft(h(xg))
    E = sqrt(p_axis**2 + m_el**2)
    demod = exp(-1j * k * xg)

    def field(t):
        """Spinor on the grid at time t, with the carrier divided out."""
        c, s = cos(E * t), sin(E * t) / E
        g_t = np.fft.ifft(c * gk - 1j * s * (p_axis * gk + m_el * hk)) * demod
        h_t = np.fft.ifft(c * hk - 1j * s * (m_el * gk - p_axis * hk)) * demod
        return g_t, h_t

    def sample(fld, pos):
        """Spinor at the particle positions."""
        carrier = exp(1j * k * pos)
        out = []
        for comp in fld:
            out.append((np.interp(pos, xg, comp.real) + 1j * np.interp(pos, xg, comp.imag)) * carrier)
        return out

    def velocity(fld, pos):
        gi, hi = sample(fld, pos)
        a, b = absolute(gi) ** 2, absolute(hi) ** 2
        return (a - b) / (a + b)

    def momentum(fld, pos):
        gi, hi = sample(fld, pos)
        Theta = 2 * arctan2(absolute(hi), absolute(gi))
        Omega = np.angle(hi) - np.angle(gi)
        with np.errstate(divide="ignore", invalid="ignore"):
            return m_el / (tan(Theta) * cos(Omega))

    positions = np.empty((len(T), len(X)))
    momenta = np.empty((len(T), len(X)))
    fld = field(T[0])
    positions[0], momenta[0] = X, momentum(fld, X)
    for i in range(len(T) - 1):
        n_sub = max(1, int(np.ceil((T[i + 1] - T[i]) / dt_max - 1e-9)))
        dt = (T[i + 1] - T[i]) / n_sub
        t = T[i]
        for _ in range(n_sub):
            fld_mid, fld_end = field(t + dt / 2), field(t + dt)
            s1 = velocity(fld, X)
            s2 = velocity(fld_mid, X + dt * s1 / 2)
            s3 = velocity(fld_mid, X + dt * s2 / 2)
            s4 = velocity(fld_end, X + dt * s3)
            X = X + dt * (s1 + 2 * s2 + 2 * s3 + s4) / 6
            fld, t = fld_end, t + dt
        positions[i + 1], momenta[i + 1] = X, momentum(fld, X)
    return positions, momenta


def canonical_momentum_density(q, branch=None):
    """Canonical (momentum-space) probability density of the packet at momentum q.

    |phi(q)|^2 is a Gaussian with mean k and std 1/(2*sigma).  Free evolution
    conserves it, so it does not depend on time.  This is the distribution of
    the canonical momentum operator -i*d/dx, not of the Bohmian p(x, t).

    branch=+1 / -1 returns only the positive / negative energy part,
    |phi(q)|^2 * |<u_+-(q)|chi>|^2 = |phi(q)|^2 * (1 +- (m*b_x + q*b_z)/E(q)) / 2,
    where b is the Bloch vector of the initial spinor chi.  The two parts add up
    to the total.
    """
    q = np.asarray(q, dtype=float)
    density = scipystats.norm.pdf(q, k, 1 / (2 * sigma))
    if branch is None:
        return density
    b_x, b_z = sin(theta) * cos(Omega0), cos(theta)
    E = sqrt(q**2 + m_el**2)
    return density * (1 + branch * (m_el * b_x + q * b_z) / E) / 2


def positive_energy_weight(branch=+1):
    """Fraction of the packet in the positive (branch=+1) or negative (-1) energy
    subspace.  Conserved in time; equals the fraction of trajectories with E > 0."""
    width = 1 / (2 * sigma)
    return quad(lambda q: canonical_momentum_density(q, branch), k - 12 * width, k + 12 * width)[0]


# =============================================================================
# 4. Plot labels and output paths
# =============================================================================
def truncate_digits(value, num):
    """Cut str(value) after `num` non-'.' characters (keeps titles/filenames short)."""
    out, count = "", 0
    for ch in str(value):
        out += ch
        if ch != ".":
            count += 1
        if count >= num:
            break
    return out


roundedk = truncate_digits(k, 3)
roundedsigma = truncate_digits(sigma, 3)
roundedtheta = truncate_digits(theta, 5)
roundedomega = truncate_digits(omega, 5)
roundedm_el = truncate_digits(m_el, 3)


def eigenstate_angles(branch=+1):
    """(theta, omega) of the initial spinor that is the free-energy eigenspinor
    at momentum k.  branch=+1 -> E = +sqrt(k^2+m^2);  branch=-1 -> E = -sqrt(k^2+m^2).

    The eigenspinors of H(p) = p*sigma_z + m*sigma_x have Bloch vector
    +-(m, 0, p)/E, so theta_+ = atan2(m, k) (valid for either sign of k) with
    Omega = 0, and theta_- = pi - theta_+ with Omega = pi.
    """
    theta_plus = arctan2(m_el, k)
    if branch > 0:
        return theta_plus, 0.0
    return pi - theta_plus, pi


positive_theta_eigen, positive_omega_eigen = eigenstate_angles(+1)
negative_theta_eigen, negative_omega_eigen = eigenstate_angles(-1)

# Initial Bloch azimuth. g carries exp(+i*omega/2) and h carries exp(-i*omega/2),
# so Omega = arg(h) - arg(g) = -omega at t = 0 (same thing for omega = 0 or pi).
Omega0 = -omega


def _angles_close(a, b, tol=0.01):
    return absolute((a - b + pi) % (2 * pi) - pi) < tol


if _angles_close(Omega0, pi):
    omega_title = r"$\pi$"
elif _angles_close(Omega0, 0):
    omega_title = "0"
else:
    omega_title = truncate_digits(Omega0, 5)


def build_title():
    """Return (matplotlib title string, 'pos' | 'neg' | 'none')."""
    om = rf"$\Omega_0$ = {omega_title}"
    kk = rf"$k_0$ = {roundedk}"
    ss = rf"$\sigma$ = {roundedsigma}"
    mm = rf"$m_{{el}}$ = {roundedm_el}"

    if _angles_close(theta, positive_theta_eigen) and _angles_close(Omega0, positive_omega_eigen):
        return ", ".join([r"$\Theta_0$ = $\Theta_+$", om, kk, ss, mm]), "pos"
    if _angles_close(theta, negative_theta_eigen) and _angles_close(Omega0, negative_omega_eigen):
        return ", ".join([r"$\Theta_0$ = $\Theta_-$", om, kk, ss, mm]), "neg"
    for angle, label in ((0, "0"), (pi / 2, r"\pi/2"), (pi, r"\pi")):
        if absolute(theta - angle) < 0.01:
            return ", ".join([om, kk, ss, rf"$\Theta_0$ = ${label}$", mm]), "none"
    return ", ".join([kk, ss, mm]), "none"


title, eigenstate_label = build_title()

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # running in a notebook / Spyder cell
    BASE_DIR = os.getcwd()


def _output_path(subdir, filename):
    directory = os.path.join(BASE_DIR, subdir)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


def _param_tag(include_omega=True):
    omega_part = f"_omega={roundedomega}" if include_omega else ""
    return f"k={k}_theta={roundedtheta}{omega_part}_m={m_el}_sigma={sigma}"


def _time_grid(t):
    return np.linspace(0, t, int(100 * t))


def _integrate(x0, T):
    """Follow the Bohmian trajectory starting at x0; returns a flat array."""
    return odeint(v, x0, T)[:, 0]


def _signed_log(a):
    return np.sign(a) * np.log(np.abs(a) + 1)


def _save_figure(save_image, subdir, filename):
    """If save_image is True, save the current figure as a PNG in BASE_DIR/subdir.

    Call this before plt.show(), which may close the figure.
    """
    if not save_image:
        return None
    path = _output_path(subdir, filename)
    plt.savefig(path, bbox_inches="tight")
    print("Saved image to", path)
    return path


# =============================================================================
# 5. Plotting and animation
# =============================================================================
# ---- Bloch sphere -----------------------------------------------------------
def _bloch_axes(title_text):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    polar, azim = np.mgrid[0:pi:20j, 0:2 * pi:40j]
    ax.plot_surface(
        sin(polar) * cos(azim), sin(polar) * sin(azim), cos(polar),
        color="w", linewidth=0, antialiased=False, alpha=0.2,
    )
    plt.title(title_text)
    return ax


def _decompose_path(x_path, times):
    """Bloch variables (R, Theta, Omega, Phi) along a trajectory, as arrays."""
    return np.array([polar_decomp(x, t) for x, t in zip(x_path, times)]).T


def plotOnBlochSphere(x0, tmax=4, save_image=False):
    """Trajectory of the spinor on the Bloch sphere, coloured by time."""
    D = 120
    times = np.linspace(0, tmax, D)
    _, Theta, Omega, _ = _decompose_path(_integrate(x0, times), times)
    print("Final theta value: %f. Theta at p = hk: %f" % (Theta[-1], positive_theta_eigen))
    print("Final omega is:" + str(Omega[-1]))

    X, Y, Z = sin(Theta) * cos(Omega), sin(Theta) * sin(Omega), cos(Theta)
    ax = _bloch_axes("Trajectories on the Bloch Sphere\n" + title)
    cols = plt.cm.copper(np.linspace(0, 1, D))
    for i in range(D - 1):
        ax.plot(X[i:i + 2], Y[i:i + 2], Z[i:i + 2], color=cols[i])
    _save_figure(save_image, "BlochSphereImages",
                 f"bloch_trajectory_x0={x0:.3g}_tmax={tmax}_{_param_tag()}.png")
    plt.show()


def plotFinalPosOnBlochSphere(t, N=100, seed=None, save_image=False):
    """Final Bloch-sphere positions of N trajectories with Gaussian-sampled starts."""
    times = np.array([0, t])   # only the final point is used, so any t > 0 works
    initials = sample_initial_positions(N, seed)
    Theta_f, Omega_f = [], []
    for x0 in initials:
        xt = _integrate(x0, times)[-1]
        _, Th, Om, _ = polar_decomp(xt, times[-1])
        Theta_f.append(Th)
        Omega_f.append(Om)
    Theta_f, Omega_f = np.array(Theta_f), np.array(Omega_f)

    ax = _bloch_axes(f"Bloch sphere location after {t} seconds\n" + title)
    ax.scatter(
        sin(Theta_f) * cos(Omega_f), sin(Theta_f) * sin(Omega_f), cos(Theta_f),
        c=plt.cm.copper(np.linspace(0, 1, N)),
    )
    _save_figure(save_image, "BlochSphereImages",
                 f"bloch_final_positions_t={t}_N={N}_{_param_tag()}.png")
    plt.show()


def plotTimeEvolutionofBlochVars(x0, t, save_image=False):
    D = 120
    times = np.linspace(0, t, D)
    R, Theta, Omega, Phi = _decompose_path(_integrate(x0, times), times)

    for name, values, color in (
        ("R", R, "red"), ("Theta", Theta, "blue"),
        ("Omega", Omega, "green"), ("Phi", Phi, "orange"),
    ):
        if name in ("Omega", "Phi"):
            values = np.unwrap(values)   # the phases come back wrapped; remove the 2*pi jumps
        plt.figure()
        plt.title(f"{name} variable\n{title}")
        plt.xlabel("Time")
        plt.ylabel(name)
        plt.plot(times, values, color=color)
        _save_figure(save_image, "BlochVariableImages",
                     f"bloch_{name.lower()}_x0={x0:.3g}_t={t}_{_param_tag()}.png")
        plt.show()

    print("Final R value:", R[-1])
    print("Final Theta value:", Theta[-1])
    print("Final Omega value:", Omega[-1])
    print("Final Phi value:", Phi[-1])
    print("Final momentum:", m_el / (tan(Theta[-1]) * cos(Omega[-1])))


# ---- Snapshots in x ---------------------------------------------------------
def plotWaveFunc(choice, t, N=200, seed=None, save_image=False):
    """Scatter plot (real part) of psi_- (choice=0) or psi_+ (choice=1) at time t."""
    S = sample_initial_positions(N, seed)
    component = g1 if choice == 0 else h1
    y = [component(s, t) for s in S]
    plt.figure()
    plt.scatter(S, np.real(y))
    plt.xlabel("x")
    plt.ylabel("Re psi")
    component_name = "psi_minus" if choice == 0 else "psi_plus"
    _save_figure(save_image, "WaveFunctionImages",
                 f"wavefunction_{component_name}_t={t}_N={N}_{_param_tag()}.png")
    plt.show()


def plotProbabilityCurrentsAndVelocity(t, save_image=False):
    """Plot j0, j1 and v as functions of x at time t."""
    print("k = %f; sigma = %f; theta = %f, electron mass = %f" % (k, sigma, theta, m_el))
    xvals = np.linspace(-5, 5, 500)
    amps = [amplitudes(x, t) for x in xvals]
    a = np.array([absolute(gg) ** 2 for gg, _ in amps])
    b = np.array([absolute(hh) ** 2 for _, hh in amps])

    for name, values in (("j0", a + b), ("j1", a - b), ("v", (a - b) / (a + b))):
        plt.figure()
        plt.plot(xvals, values)
        if name == "v":
            plt.ylim([-1, 1])
        plt.title(f"{name} at time {t}")
        plt.xlabel("x")
        plt.ylabel(name)
        _save_figure(save_image, "ProbabilityCurrentsVelocityImages", f"{name}_t={t}_{_param_tag()}.png")
        plt.show()


def probability_check(t_max, save_image=False):
    """Sanity check: the integral of j0 over x should be 1 for all t."""
    T = np.linspace(0, t_max, int(t_max * 20))
    P = [quad(lambda x: j0(x, t), -5 * sigma - t, 5 * sigma + t)[0] for t in T]
    plt.figure()
    plt.xlabel("Time [t]")
    plt.ylabel("Integral of wave func **2")
    plt.axis([0, t_max, 0, 1.2])
    plt.plot(T, P)
    _save_figure(save_image, "ProbabilityCheckImages",
                 f"probability_check_tmax={t_max}_{_param_tag()}.png")
    plt.show()


# ---- Trajectories -----------------------------------------------------------
def rk4(x0=None, n=1000, t_max=5, save_image=False):
    """Plot one RK4 trajectory and its momentum. Prompts for x0 if not given."""
    if x0 is None:
        x0 = float(input("x0 = "))
    dt = t_max / n
    T = np.linspace(0, t_max, n + 1)
    X = [x0]
    P = [p(x0, 0)]
    for tn in T[:-1]:
        xn = X[-1]
        s1 = v(xn, tn)
        s2 = v(xn + dt * s1 / 2, tn + dt / 2)
        s3 = v(xn + dt * s2 / 2, tn + dt / 2)
        s4 = v(xn + dt * s3, tn + dt)
        x_new = xn + dt * (s1 + 2 * s2 + 2 * s3 + s4) / 6
        X.append(x_new)
        P.append(p(x_new, tn + dt))
    with plt.rc_context({"font.size": 16}):   # scoped, so later plots keep the default size
        plt.figure()
        plt.plot(X, T, "b-", label="x(t)")
        plt.plot(P, T, "m:", label="p(t)")
        plt.ylabel("Time (t)")
        plt.title("Position and Momentum of an Electron Trajectory", fontsize=18)
        plt.legend(fontsize=10)
        print("k = %f; sigma = %f; theta = %f, electron mass = %f" % (k, sigma, theta, m_el))
        _save_figure(save_image, "ElectronTrajectoriesImages",
                     f"rk4_x0={x0:.3g}_tmax={t_max}_{_param_tag()}.png")
        plt.show()


def plotTrajectories(t, x_init=None, N=20, seed=None, save_image=False):
    """Plot x(t) for each starting point; print summary stats of final p and E.

    If x_init is None, N starting points are sampled from |psi(x, 0)|^2.
    """
    x_init = sorted(_initial_positions(x_init, N, seed))
    T = _time_grid(t)
    plt.figure()
    plt.title(title)
    plt.xlabel("Position [x(t)]")
    plt.ylabel("Time [t]")

    P5, E5 = [], []
    for x0 in x_init:
        X = _integrate(x0, T)
        P5.append(float(p(X[-1], T[-1])))
        E5.append(float(e(X[-1], T[-1])))
        plt.plot(X, T)

    filename = f"electrontrajectoriesfortime{t}_{_param_tag(include_omega=False)}.png"
    _save_figure(save_image, "ElectronTrajectoriesImages", filename)

    pos = [m for m in P5 if m >= 0]
    neg = [m for m in P5 if m < 0]
    print(title)
    print("Momentum list:", P5)
    print("Energy list:", E5)
    print(f"The mean momentum at t={t} is: {np.mean(P5):f}")
    print(f"The mean positive momentum at t={t} is: {np.mean(pos) if pos else 'n/a'}")
    print(f"The mean negative momentum at t={t} is: {np.mean(neg) if neg else 'n/a'}")
    print(f"The mean energy at t={t} is: {np.mean(E5):f}")
    plt.show()
    return P5


def plotSpecificTrajectories(t, xlist, save_image=False):
    T = _time_grid(t)
    plt.figure()
    plt.title(title)
    plt.xlabel("Position [x(t)]")
    plt.ylabel("Time [t]")
    for x0 in xlist:
        plt.plot(_integrate(x0, T), T)
    _save_figure(save_image, "ElectronTrajectoriesImages",
                 f"specific_trajectories_t={t}_{_param_tag()}.png")
    plt.show()


def _plot_over_time(t, x_init, quantity, ylabel, ylim, subdir, prefix, transform=None,
                    save_image=False):
    """Shared body for the velocity / momentum / energy vs. time plots."""
    T = _time_grid(t)
    plt.figure(figsize=(12, 9))
    plt.title(title)
    plt.xlabel("Time [t]")
    plt.ylabel(ylabel)
    plt.axis([0, t, *ylim])

    finals = []
    for x0 in x_init:
        X = _integrate(x0, T)
        values = np.array([quantity(X[n], T[n]) for n in range(len(T))], dtype=float)
        if transform is not None:
            values = transform(values)
        finals.append(float(values[-1]))
        plt.plot(T, values)

    print(finals)
    _save_figure(save_image, subdir, f"{prefix}_{_param_tag()}.png")
    plt.show()
    return finals


def plotTrajectoryVelocities(t, x_init=None, N=20, seed=None, save_image=False):
    return _plot_over_time(t, _initial_positions(x_init, N, seed), v,
                           r"Velocity [$\cos\Theta$]", (-1, 1),
                           "VelocityImages", "velocity", save_image=save_image)


def plotTrajectoryMomenta(t, x_init=None, N=20, seed=None, save_image=False):
    return _plot_over_time(t, _initial_positions(x_init, N, seed), p,
                           r"Momentum [$m_{el}\cot\Theta\sec\Omega$]", (-40, 40),
                           "MomentumImages", "momentum", save_image=save_image)


def plotTrajectoryEnergies(t, x_init=None, N=20, seed=None, save_image=False):
    return _plot_over_time(t, _initial_positions(x_init, N, seed), e,
                           r"Energy [$m_{el}\csc\Theta\sec\Omega$]", (-40, 40),
                           "EnergyImages", "energy", save_image=save_image)


def plotTrajectoryLogMomentum(t, x_init=None, N=20, seed=None, save_image=False):
    return _plot_over_time(t, _initial_positions(x_init, N, seed), p,
                           "sign(p) * log(|p| + 1)", (-4, 4),
                           "MomentumImages", "log_momentum", transform=_signed_log,
                           save_image=save_image)


def plotTrajectoryLogEnergy(t, x_init=None, N=20, seed=None, save_image=False):
    return _plot_over_time(t, _initial_positions(x_init, N, seed), e,
                           "sign(E) * log(|E| + 1)", (-4, 4),
                           "EnergyImages", "log_energy", transform=_signed_log,
                           save_image=save_image)


# ---- Animations -------------------------------------------------------------
def momentum_animation(t, N=100, x_init=None, fps=10, p_range=None, n_std=3.0,
                       coverage=0.95, margin=0.1, center_on_k=True, n_bins=400, fixed_y=True,
                       frame_step=1,
                       method="grid", show_canonical=True, show_branches=False, seed=None):
    """Animated histogram of the Bohmian momenta p = m*cot(Theta)*sec(Omega) of N trajectories.

    Starting positions are sampled from |psi(x, 0)|^2 unless x_init is given, in
    which case N is taken to be len(x_init).

    method="grid" (default): trajectories are integrated on the exact FFT grid solution
        (see grid_bohmian_trajectories); seconds even for N in the hundreds.
    method="quad": integrates with the Bessel-kernel solution and scipy quadrature.  Same
        answers, but slower: roughly 2 s per trajectory at t = 4.

    frame_step: draw every frame_step-th time step.  Rendering the GIF is the slow part of
        the grid method (about 0.2 s per frame), so frame_step=4 makes it about 4x faster
        and the animation 4x shorter.

    x axis (fixed for the whole animation): by default it spans the central `coverage`
        fraction of ALL the trajectories' momenta pooled over every time step, padded by
        `margin` of that width on each side, and always widened to include the canonical
        window k +- n_std/(2*sigma).  That is wide enough to follow the histogram from its
        early spread (p can be far from k, e.g. p = 0 at t = 0 for theta = pi/2) to where
        it settles.  With center_on_k=True (default) the axis is then made symmetric about k
        by using the larger of its two distances from k on both sides.  Pass p_range=(lo, hi)
        to choose the axis yourself (used exactly as given).

    y axis: fixed too by default (scaled to a typical frame, so a one-frame spike such as
        t = 0 is clipped).  fixed_y=False rescales every frame, which makes the early
        frames easier to read but hides how the peak grows as the trajectories converge.

    Trajectories outside the view (e.g. where cos(Omega) ~ 0 makes p blow up) are not
    drawn; each frame says how many there are.

    grey dotted  : distribution of the canonical momentum, N * |phi(p)|^2 (Gaussian, mean k,
                   std 1/(2*sigma), constant in time).  A different observable from the
                   Bohmian p shown by the bars.
    show_branches: also draw the positive- and negative-energy parts of the canonical
                   distribution (they add up to the grey curve).
    """
    x_init = _initial_positions(x_init, N, seed)
    N = len(x_init)

    T = _time_grid(t)
    frames = len(T) - 1
    if method == "grid":
        _, momenta = grid_bohmian_trajectories(x_init, T)
    elif method == "quad":
        momenta = np.empty((len(T), len(x_init)))
        for col, x0 in enumerate(x_init):
            X = _integrate(x0, T)
            for row in range(len(T)):
                momenta[row, col] = p(X[row], T[row])
    else:
        raise ValueError(f"method must be 'grid' or 'quad', not {method!r}")

    # ---- one fixed x axis for the whole animation -----------------------------------
    half = n_std / (2 * sigma)
    if p_range is not None:
        lower, upper = p_range
    else:
        pooled = momenta[np.isfinite(momenta)]
        tail = (1 - coverage) / 2
        q_lo, q_hi = np.quantile(pooled, [tail, 1 - tail])
        pad = margin * (q_hi - q_lo)
        lower, upper = min(q_lo - pad, k - half), max(q_hi + pad, k + half)
        if center_on_k:
            reach = max(k - lower, upper - k)
            lower, upper = k - reach, k + reach

    bins = np.linspace(lower, upper, n_bins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    width = bins[1] - bins[0]
    counts = [np.histogram(momenta[f], bins)[0] for f in range(frames)]
    outside = [N - c.sum() for c in counts]
    canonical = N * width * canonical_momentum_density(centers)
    canonical_pm = {br: N * width * canonical_momentum_density(centers, br) for br in (+1, -1)}

    # Fixed y axis too.  A few frames can be a tall spike (e.g. every trajectory shares one p
    # at t = 0), so scale to a typical frame; the spike is simply clipped.
    y_max = np.percentile([c.max() for c in counts], 90)
    if show_canonical:
        y_max = max(y_max, canonical.max())
    y_max = 1.15 * max(y_max, 1)

    fig, ax = plt.subplots(1, 1)

    def draw(frame):
        ax.clear()
        ax.bar(centers, counts[frame], width=width, ec="black", fc="blue", lw=0.3,
               label="trajectories")
        if show_canonical:
            ax.plot(centers, canonical, color="dimgrey", ls=":", lw=2,
                    label=r"canonical momentum $|\phi(p)|^2$")
            if show_branches:
                ax.plot(centers, canonical_pm[+1], color="darkorange", lw=1, label=r"  $E>0$ part")
                ax.plot(centers, canonical_pm[-1], color="purple", lw=1, label=r"  $E<0$ part")
        if outside[frame]:
            ax.text(0.02, 0.96, f"{outside[frame]} of {N} outside view", transform=ax.transAxes,
                    fontsize=8, va="top")
        ax.set_xlabel("p")
        ax.set_ylabel("N")
        ax.set_xlim([lower, upper])
        if fixed_y:
            ax.set_ylim([0, y_max])
        else:
            top = counts[frame].max()
            ax.set_ylim([0, 1.15 * max(top, canonical.max() if show_canonical else 0, 1)])
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Momenta of {N} trajectories, t = {round(T[frame], 2)}\n{title}")

    anim = FuncAnimation(fig, draw, frames=range(0, frames, frame_step), interval=100, repeat=False)

    filename = f"momentum_animation_t_{t}_N_{N}_{_param_tag()}.gif"
    anim.save(_output_path("MomentumAnimations", filename), writer=PillowWriter(fps=fps))
    plt.show()
    return anim


def animate_j0(t_max=3, n_frames=30, x_max=5.0, y_max=1.0, fps=10, save_gif=False):
    """Animate the probability density j0(x, t)."""
    xs = np.linspace(-x_max, x_max, 50)
    with plt.rc_context({"font.size": 16}):   # scoped, so later plots keep the default size
        fig = plt.figure(figsize=(8, 5))
        (curve,) = plt.plot([], [])
        label = plt.text(x_max - 3, 0.9 * y_max, "")
        plt.xlim(-x_max, x_max)
        plt.ylim(0, y_max)
        plt.title("wave function evolution")
        plt.xlabel("x")
        plt.ylabel("probability amplitude")
        fig.tight_layout()

        def init():
            curve.set_data([], [])
            label.set_text("")
            return curve, label

        def frame(i):
            t = t_max * i / n_frames
            curve.set_data(xs, [j0(x, t) for x in xs])
            label.set_text(f"t = {t:.3f}")
            return curve, label

        anim = FuncAnimation(fig, frame, init_func=init, frames=n_frames, interval=30, blit=True)

        if save_gif:
            path = _output_path("DensityAnimations",
                                f"j0_animation_tmax={t_max}_frames={n_frames}_{_param_tag()}.gif")
            anim.save(path, writer=PillowWriter(fps=fps))
            print("Saved animation to", path)
        plt.show()
    return anim


# =============================================================================
# 6. Entry point
# =============================================================================
if __name__ == "__main__":
    start_time = time.time()

    # Put whatever functions you want to run here, e.g.:
    # plotTrajectories(5)                              # 20 starts sampled from |psi(x,0)|^2
    # plotTrajectories(5, x_init=[-1, 0, 1])           # or choose them yourself
    # momentum_animation(5, N=100, seed=0)

    print(f"--- {time.time() - start_time:.2f} seconds ---")
    print(title)
