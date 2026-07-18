from typing import KeysView
from operator import pos
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 18:51:39 2022

@author: abby
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sin, cos, pi, inf, tan, sqrt, arctan, log, arange, absolute
from scipy.integrate import odeint, quad
from scipy.special import jv
from matplotlib.animation import FuncAnimation
import pandas as pd
import statistics as stats
import scipy.stats as scipystats
import cmath
import math
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pylab import meshgrid, cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time
from datetime import date
import os

start_time = time.time()
#Some trig funcion shorthands
def coss(y):
    return cos(y)**2
def sins(y):
    return sin(y)**2
def tans(y):
    return tan(y)**2

# the parameters

k = -10
m_el = 1
# theta = arctan(m_el/k) 
theta = pi/2
omega = 0
if theta < 0: 
   theta += pi
if theta > pi:
   theta -= pi
# theta = eval(input("theta: "))
sigma = 1/sqrt(2)
i = complex(0,1)


#theta = np.random.uniform(0, pi)
#k = np.random.uniform(-10, 10)
#sigma = np.random.uniform(0.5, 1.5)
#m_el = np.random.uniform(1, 8)

# We have set h_bar = c = 1.



kappa = k/m_el

if theta <= pi/2:
    V = absolute(k)/sqrt(m_el**2 + k**2)
else:
    V = -1*absolute(k)/sqrt(m_el**2 + k**2)
    


# The initial data. 
#g and gv_init represent psi- and its derivative at t=0. h is psi+, and hv_init it's time derivative.

def gauss(x):
  return (2.0*pi*(sigma)**2)**(-1/4)*exp(-x**2/(4*sigma**2)+i*k*x)
def g(x):
    return cos(theta/2)*gauss(x)*exp(i*omega/2)

def h(x):
    return sin(theta/2)*gauss(x)*exp(-1*i*omega/2)

def gv_init(x):
   return (x/(2*sigma**2) - i*k)*g(x) - i*m_el*h(x)

def hv_init(x):
   return (i*k - x/(2*sigma**2))*h(x)-i*m_el*g(x)

""" annoyingly, scipy.quad cannot evaluate complex integrands, 
so we split everything into real and imaginary parts before proceeding"""

def rg(x):
  return np.real(g(x))
def ig(x):
  return np.imag(g(x))
def rh(x):
  return np.real(h(x))
def ih(x):
  return np.imag(h(x))
def rgv(x):
  return np.real(gv_init(x))
def igv(x):
  return np.imag(gv_init(x))
def rhv(x):
  return np.real(hv_init(x))
def ihv(x):
  return np.imag(hv_init(x))



def besselfuncfrac(X):
    if absolute(X)<0.00000001:
        return m_el/2
    else: 
        return jv(1, m_el*X)/X

# g1 and h1 are the time-dependent sol'ns for each component

def g1(x,t):
  if t == 0:
    return g(x)
  else: 
    initg = g(x-t)
    grossg1 = -.5*m_el*(quad(lambda r, x, t : rg(r)*besselfuncfrac(sqrt(t**2-(x-r)**2))*(t + (x-r)), x-t, x+t,args=(x,t))[0]) + i*(-.5)*m_el*(quad(lambda r, x, t : ig(r)*besselfuncfrac(sqrt(t**2-(x-r)**2))*(t + (x-r)), x-t, x+t,args=(x,t))[0])
    grossg2 = .5*m_el*(quad(lambda r, x, t: ih(r)*jv(0,m_el*sqrt(t**2-(x-r)**2)),x-t,x+t,args=(x,t))[0]) + i*(-.5)*m_el*(quad(lambda r, x, t: rh(r)*jv(0,m_el*sqrt(t**2-(x-r)**2)),x-t,x+t,args=(x,t))[0])
    #rpart = -.5*m_el*(quad(lambda r, x, t : rg(r)*besselfuncfrac(sqrt(t**2-(x-r)**2))*(t + (x-r)), x-t, x+t,args=(x,t))[0])+.5*m_el*(quad(lambda r, x, t: ih(r)*jv(0,m_el*sqrt(t**2-(x-r)**2)),x-t,x+t,args=(x,t))[0])
    #ipart = -.5*m_el*(quad(lambda r, x, t : ig(r)*besselfuncfrac(sqrt(t**2-(x-r)**2))*(t + (x-r)), x-t, x+t,args=(x,t))[0])-.5*m_el*(quad(lambda r, x, t: rh(r)*jv(0,m_el*sqrt(t**2-(x-r)**2)),x-t,x+t,args=(x,t))[0])
    return initg + grossg1 + grossg2


def h1(x,t):
  if t == 0:
    return h(x) 
  else:
      inith = h(x+t)
      grossh1 = -.5*m_el*(quad(lambda r, x, t : rh(r)*besselfuncfrac(sqrt(t**2-(x-r)**2))*(t - (x-r)), x-t, x+t,args=(x,t))[0]) + i*(-.5)*m_el*(quad(lambda r, x, t : ih(r)*besselfuncfrac(sqrt(t**2-(x-r)**2))*(t - (x-r)), x-t, x+t,args=(x,t))[0])
      grossh2 = .5*m_el*(quad(lambda r, x, t: ig(r)*jv(0,m_el*sqrt(t**2-(x-r)**2)),x-t,x+t,args=(x,t))[0]) + i*(-.5)*m_el*(quad(lambda r, x, t: rg(r)*jv(0,m_el*sqrt(t**2-(x-r)**2)),x-t,x+t,args=(x,t))[0])
      #rpart = -.5*m_el*(quad(lambda r, x, t : rh(r)*besselfuncfrac(sqrt(t**2-(x-r)**2))*(t - (x-r)), x-t, x+t,args=(x,t))[0])+.5*m_el*(quad(lambda r, x, t: ig(r)*jv(0,m_el*sqrt(t**2-(x-r)**2)),x-t,x+t,args=(x,t))[0])
      #ipart = -.5*m_el*(quad(lambda r, x, t : ih(r)*besselfuncfrac(sqrt(t**2-(x-r)**2))*(t - (x-r)), x-t, x+t,args=(x,t))[0])-.5*m_el*(quad(lambda r, x, t: rg(r)*jv(0,m_el*sqrt(t**2-(x-r)**2)),x-t,x+t,args=(x,t))[0])
      return inith + grossh1 + grossh2


  

# j0 is the density fxn, j1 is prob. current, and v is velocity field

def j0(x,t):
  return absolute(g1(x, t))**2 + absolute(h1(x, t))**2

def j1(x,t):
  return absolute(g1(x, t))**2 - absolute(h1(x, t))**2

def v(x, t):
  return float(absolute(g1(x, t))**2 - absolute(h1(x, t))**2)/(absolute(g1(x, t))**2 + absolute(h1(x, t))**2)


def polar_decomp(x,t):
  (r1, f1) = cmath.polar(g1(x,t))
  (r2, f2) = cmath.polar(h1(x,t))
  R = sqrt(r1**2+r2**2)
  if r1 == 0:
      if r2 < 0:
          T = -1*pi/2
      if r2 > 0:
          T = pi/2
  else:
      T = 2*arctan(r2/r1)
  O = (f2-f1)
  P = (f1+f2)
  return (R, T, O, P)






def plotWaveFunc(choice, t):
  S = np.random.normal(0, sigma, 200)
  y_coords = []
  if choice == 0:
    for s in S:
      y = g1(s, t)
      y_coords.append(y)
  if choice ==1:
    for s in S:
      y = h1(s, t)
      y_coords.append(y)
  plt.scatter(S, y_coords)
  plt.show()
 



#plotWaveFunc(0, 0)

# momentum!
def p(x,t):
  (R, T, O, P) = polar_decomp(x, t)
  return m_el / (tan(T)*cos(O))

def e(x, t):
    (R, T, O, P) = polar_decomp(x, t)
    return m_el / (sin(T)*cos(O))


# randomly sampled initial values


# for a given time, this plots j0, j1 and v as fxns of x.
def plotStuff(t):
  print("k = %f; sigma = %f; theta = %f, electron mass = %f" % (k, sigma, theta, m_el))

  j0vals = []
  xvals = np.linspace(-5,5,500)
  for x in xvals:
    j0vals.append(j0(x,t))
  plt.plot(xvals,j0vals)
  plt.title("j0 at time " + str(t))
  plt.xlabel("x")
  plt.ylabel("j0")
  plt.show()


  j1vals = []
  for x in xvals:
    j1vals.append(j1(x,t))
  plt.plot(xvals,j1vals)
  plt.title("j1 at time " + str(t))
  plt.xlabel("x")
  plt.ylabel("j1")
  plt.show()

  v_vals = []
  for x in xvals:
    v_vals.append(v(x,t))
  plt.plot(xvals,v_vals)
  plt.ylim([-1,1])
  plt.title("v at time " + str(t))
  plt.xlabel("x")
  plt.ylabel("v")
  plt.show()
  



# call rk4 to plot a trajectory. it will ask for an initial value.

def rk4():
#RK4
  x0 = float(input("x0 = "))
  n = 1000
  d = .005
  t = 5
  X = [x0]
  T = np.linspace(0,t,n+1)
  P = [p(x0, 0)]
  for tn in T[:-1]:
    xn = X[-1]
    k1 = v(xn,tn)
    k2 = v(xn+d*k1/2, tn+d/2)
    k3 = v(xn+d*k2/2, tn+d/2)
    k4 = v(xn+d*k3,tn+d)
    x_new = xn + d*(k1+2*k2+2*k3+k4)/6
    X.append(x_new)
    P.append(p(x_new, tn + d))
  plt.rc('font', size=16) 
  plt.plot(X,T, 'b-', label='x(t)')
  plt.plot(P,T, 'm:',label = 'p(t)')
  plt.ylabel('Time (t)')
  plt.title('Position and Momentum of an Electron Trajectory', fontsize=18)
  plt.legend(fontsize = 10)
  print("k = %f; sigma = %f; theta = %f, electron mass = %f" % (k, sigma, theta, m_el))

def mean(list):
  s = 0
  for y in list:
    s += y
  return s/len(list)

def limit(s, t):
    return (m_el/2)*(absolute(g1(s, t))/absolute(h1(s, t)) - absolute(h1(s, t))/absolute(g1(s, t)))





#ignore
"""

# Vector Fields
x, y = np.meshgrid(np.linspace(-5, 5, 10), 
                   np.linspace(-5, 5, 10))
  
# Directional vectors
a = v(x,y)
b = 1
  
# Plotting Vector Field with QUIVER
plt.quiver(x, y, a, b, color='g')
plt.title('Velocity Field')
  
# Setting x, y boundary limits
plt.xlim(-5, 5)
plt.ylim(0, 5)
  
# Show plot with grid
plt.grid()
plt.show()
"""
#---------------THE CODE BELOW IS TO ANIMATE THE PDF J_0--------------------------
# remove triple quotes and call animate() to run.
"""
xmax = 5.0                   # x range of plot will be 0 to xmax
ymax = 1.0                    # y range of plot will be -ymax to +ymax
xp = np.linspace(-xmax, xmax, 50) # array of x values for plotting

# Make the plot, but with no plotting data or label text:
plt.rc('font', size=16)
fig = plt.figure(figsize=(8,5))
curve = plt.plot([],[])[0]
label = plt.text(xmax-3,-ymax+0.5,'')
plt.xlim(-xmax,xmax)
plt.ylim(0,ymax)
plt.title('wave function evolution')
plt.xlabel('x')
plt.ylabel('probability amplitude')
"""
tmax = 3   # pick so wave packet will reach x=xmax when t=tmax
nf = 30         # pick number of frames in animation

'''
def frame0():
    """Initialization function,
    used as background for all frames.
    """
    curve.set_data([], [])
    label.set_text('')
    return curve,label

def frame(i):
    """Animation function,
    called for each frame with i = frame number.
    """
    t = tmax*i/nf
    y=[]
    for val in xp:
      y.append(j0(val,t))
    curve.set_data(xp, y)
    label.set_text('t = {:.3f}'.format(t))
    return curve,label
'''
"""
# Call the animator: 
anim = FuncAnimation(fig, frame, init_func=frame0, frames=nf, interval=30, blit=True)
#anim.save('wavepacket.mp4')       # optionally save the animation to a video file
plt.close()                     # supress automatic display of non-animated plot
plt.rc('animation', html='html5')  # needed for Jupyter Notebook to display animation
anim
"""
#--------------------------------------------------------------


def formatstring(string, num):
    newstr = ''
    trial = 0
    for i in string:
        if i != '.':
            newstr+= i
            trial+=1
        if i == '.':
            newstr+=i
        if trial >= num:
            break
    return newstr

roundedk = formatstring(str(k), 3)
roundedsigma = formatstring(str(sigma), 3)
roundedtheta = formatstring(str(theta), 5)
roundedomega = formatstring(str(omega), 5)
roundedm_el = formatstring(str(m_el), 3)
if k != 0:
    positive_theta_eigen = arctan(m_el/k) 
    negative_theta_eigen = pi - arctan(m_el/k) 
else:
   positive_theta_eigen = pi/2
   negative_theta_eigen = pi/2
is_neg_or_pos = "none"
title = 'k = ' + roundedk + ', sigma = ' + roundedsigma + ', theta = ' + roundedtheta + ', m_el = ' + roundedm_el
omega_title = omega
if absolute(omega - pi) < 0.01:
   omega_title = r'$\pi$'
if absolute(omega) < 0.01:
   omega_title = "0"
if absolute(theta - positive_theta_eigen) < 0.01:
    print("pos")
    is_neg_or_pos = "pos"
    title = r'$\Theta_0$ = $\Theta_+$, $\Omega_0$ = ' + omega_title + r', $k_0$ = ' + roundedk + r'$, \sigma$ = ' + roundedsigma +  r', $m_{el}$ = ' + roundedm_el
elif absolute(theta - negative_theta_eigen) < 0.01:
    print("neg")
    is_neg_or_pos = "neg"
    title = r'$\Theta_0$ = $\Theta_-$, $\Omega_0$ = ' + omega_title + r', $k_0$ = ' + roundedk + r'$, \sigma$ = ' + roundedsigma +  r', $m_{el}$ = ' + roundedm_el
elif absolute(theta) < 0.01:
    title =  r'$\Omega_0$ = ' +  omega_title + r', $k_0$ = ' + roundedk + r'$, \sigma$ = ' + roundedsigma +  r', $\Theta_0$ = $0$, $m_{el}$ = ' + roundedm_el
elif absolute(theta-(pi/2)) < 0.01:
    title = r'$\Omega_0$ = ' +  omega_title + r', $k_0$ = ' + roundedk + r'$, \sigma$ = ' + roundedsigma +  r', $\Theta_0$ = $\pi/2$, $m_{el}$ = ' + roundedm_el
elif absolute(theta - pi) < 0.01:
    title = r'$\Omega_0$ = ' +  omega_title + r', $k_0$ = ' + roundedk + r'$, \sigma$ = ' + roundedsigma +  r', $\Theta_0$ = $\pi$, $m_{el}$ = ' + roundedm_el
else: 
    titlenotheta = r'$k_0$ = ' + roundedk + r'$, \sigma$ = ' + roundedsigma +  r', $m_{el}$ = ' + roundedm_el
    title = titlenotheta





def plotTrajectories(t, x_init):
  x_init.sort()
  T = np.linspace(0,t,int(t*100))
  P5 = []
  E5 = []
  ultimatedict = {"time":T}
  roundedk = formatstring(str(k), 3)
  roundedsigma = formatstring(str(sigma), 3)
  roundedtheta = formatstring(str(theta), 5)
  roundedm_el = formatstring(str(m_el), 3)
  #title = 'k = ' + roundedk + ', sigma = ' + roundedsigma + ', theta = ' + roundedtheta + ', m_el = ' + roundedm_el
  plt.title(title)
  plt.xlabel('Position [x(t)]')
  plt.ylabel('Time [t]')
  i = 1

  imgname =  "electrontrajectoriesfortime" + str(t) + "_k=" + str(k) + "_theta=" + str(roundedtheta) + "_m=" + str(m_el) + "_sigma=" + str(sigma) + '.png'
  #filepath1 = "/Users/abbyperryman/Documents/Math Phys Research/ElectronTrajectoriesSpreadsheets/" + filename
  #filepath2 = "/Users/abbyperryman/Documents/Math Phys Research/ElectronTrajectoriesImages/" + imgname
  script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
  image_dir = os.path.join(script_dir, "ElectronTrajectoriesImages")
  filepath = os.path.join(image_dir, imgname)
  for x in x_init:
    X = odeint(v,x,T)
    p_end = p(X[-1],T[-1])
    e_end = e(X[-1], T[-1])
    E5.append(float(e_end))
    P5.append(float(p_end))
    plt.plot(X,T)
    X = list(X)
    ultimatedict[str(i)] = X
    i+=1
  data = pd.DataFrame(ultimatedict)
  plt.savefig(filepath)
  negplist = []
  posplist = []
  for mom in P5:
      if mom < 0: 
          negplist.append(mom)
      else:
          posplist.append(mom)
  if len(posplist)>0:
      pospmean = mean(posplist)
  else:
      pospmean = "n/a"
  if len(negplist)>0:
      negpmean = mean(negplist)
  else:
      negpmean = "n/a"
  print(title)
  print("Momentum list:", P5)
  print("Energy list:", E5)
  print("The mean momentum at t=" + str(t) + " is: %f" % (mean(P5)))
  print("The mean positive momentum at t=" + str(t) + " is: " + str(pospmean))
  print("The mean negative momentum at t=" + str(t) + " is: " + str(negpmean))
  print("The mean energy at t=" + str(t) + " is: %f" % (mean(E5)))
  plt.show()
  return P5

def plotSpecificTrajectories(t, xlist):
    T = np.linspace(0,t,int(t*100))
    plt.title(title)
    plt.xlabel('Position [x(t)]')
    plt.ylabel('Time [t]')
    for x in xlist:
        X = odeint(v, x, T)
        plt.plot(X, T)
    plt.show()


def plotTrajectoryVelocities(t, x_init):
  T = np.linspace(0,t,100*t)
  plt.figure(figsize=(12,9))
  plt.title(title)
  plt.xlabel('Time [t]')
  plt.ylabel('Velocity ' + r'[$\cos\Theta$]')
  plt.ylim=[-1, 1]
  plt.axis([0, t, -1, 1])
  finalV = []
  imgname =  "velocity_k=" + str(k) + "_theta=" + str(roundedtheta) + "_omega=" + str(roundedomega) + "_m=" + str(m_el) + "_sigma=" + str(sigma) + '.png'
  #filepath1 = "/Users/abbyperryman/Documents/Math Phys Research/ElectronTrajectoriesSpreadsheets/" + filename
  #filepath2 = "/Users/abbyperryman/Documents/Math Phys Research/ElectronTrajectoriesImages/" + imgname
  script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
  image_dir = os.path.join(script_dir, "VelocityImages")
  filepath = os.path.join(image_dir, imgname)
  for x in x_init:
    V = []
    X = odeint(v,x,T) 
    for i in range(len(X)):
        V.append(v(X[i], T[i]))
    finalV.append(float(V[-1]))
    plt.plot(T,V)
  print(finalV)
  plt.savefig(filepath)
  plt.show()
  return



def plotTrajectoryMomenta(t, x_init):
  T = np.linspace(0,t,100*t)
  plt.figure(figsize=(12,9))
  plt.title(title)
  plt.xlabel('Time [t]')
  plt.ylabel('Momentum ' + r'[$m_{el}\cot\Theta\sec\Omega$]')
  plt.ylim=[-40, 40]
  plt.axis([0, t, -40, 40])
  imgname =  "momentum_k=" + str(k) + "_theta=" + str(roundedtheta) + "_omega=" + str(roundedomega) +  "_m=" + str(m_el) + "_sigma=" + str(sigma) + '.png'
  #filepath1 = "/Users/abbyperryman/Documents/Math Phys Research/ElectronTrajectoriesSpreadsheets/" + filename
  #filepath2 = "/Users/abbyperryman/Documents/Math Phys Research/ElectronTrajectoriesImages/" + imgname
  script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
  image_dir = os.path.join(script_dir, "MomentumImages")
  filepath = os.path.join(image_dir, imgname)
  finalP = []
  for x in x_init:
    P = []
    X = odeint(v,x,T) 
    for i in range(len(X)):
        P.append(p(X[i], T[i]))
    finalP.append(float(P[-1]))
    plt.plot(T,P)
  print(finalP)
  plt.savefig(filepath)
  plt.show()
  return
 

def momentum_animation(t, N, x_init):
  import functools

  import matplotlib.pyplot as plt
  import numpy as np

  import matplotlib.animation as animation

  # Setting up a random number generator with a fixed state for reproducibility.
  rng = np.random.default_rng(seed=19680801)
  # Fixing bin edges.
  HIST_BINS = np.linspace(-1.2*k, 1.2*k, 50)


  T = np.linspace(0,t,100*t)
  all_data = {i: [] for i in range(len(T))}

  for x in x_init:
    X = odeint(v,x,T) 
    for i in range(len(T)):
      all_data[i].append(p(X[i], T[i]))

  # Histogram our data with numpy.
  def animate(frame_number, bar_container):
    # Simulate new data coming in.
    data = all_data[frame_number]
    n, _ = np.histogram(data, HIST_BINS)
    for count, rect in zip(n, bar_container.patches):
        rect.set_height(count)
    return bar_container.patches
  
  initial_momentum = all_data[0]
  

  fig, ax = plt.subplots()
  ax.set_ylim(0, 1.05*N)
  plt.xlabel('p')
  plt.ylabel('N')

  plt.show()
  _, _, bar_container = ax.hist(initial_momentum, HIST_BINS, lw=1,
                                ec="black", fc="blue", alpha=0.5)
  
  anim = functools.partial(animate, bar_container=bar_container)
  ani = animation.FuncAnimation(fig, anim, 10, repeat=False, blit=True)
  plt.title("Bohmian momenta of " + str(N) + " trajectories\n" + title)
  script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
  animation_dir = os.path.join(script_dir, "MomentumAnimations")
  animation_name = 'momentum_animation_t_' + str(t) + '_N_' + str(N) + '.gif'
  filepath = os.path.join(animation_dir, animation_name)
  plt.show()
  ani.save(filepath, writer="pillow") 
  plt.close()


def momentum_animation_with_theoretical(t, N, x_init, fps):
  # Note that the theoretical graph is only correct if sigma = sqrt(2)
  import functools
  import matplotlib.animation as animation

  # Setting up a random number generator with a fixed state for reproducibility.
  rng = np.random.default_rng(seed=19680801)
  # Fixing bin edges.
  if k > 0:
     lower_val = -1.5*k
     upper_val = 3.5*k
  else:
     lower_val = 3.5*k
     upper_val = -1.5*k
  HIST_BINS = np.linspace(lower_val, upper_val, 200)
  x_for_normal = np.linspace(lower_val, upper_val, 500)

  T = np.linspace(0,t,100*t)
  all_data = {i: [] for i in range(len(T))}

  for x in x_init:
    X = odeint(v,x,T) 
    for i in range(len(T)):
      all_data[i].append(p(X[i], T[i]))

  fig, ax = plt.subplots(1, 1)

  # Histogram our data with numpy.
  def animate(frame_number):
    ax.clear()
    ax.plot(x_for_normal, N*scipystats.norm.pdf(x_for_normal, k, sigma), 'g--')
    # Simulate new data coming in.
    data = all_data[frame_number]
    t_val = round(T[frame_number], 2)
    ax.hist(data, HIST_BINS, lw=1,
                                ec="black", fc="blue")
    ax.set_xlabel('p')
    ax.set_ylabel('N')
    ax.set_xlim([lower_val, upper_val])
    ax.set_ylim([0, N])
    ax.set_title("Bohmian momenta of " + str(N) + " trajectories, t = " + str(t_val) + "\n" + title)

  ani = animation.FuncAnimation(fig, animate, frames=len(T)-1, interval=100, repeat=False)
  
  plt.close()

  script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
  animation_dir = os.path.join(script_dir, "MomentumAnimations")
  imgname =  "log_energy_k=" + str(k) + "_theta=" + str(roundedtheta) + "_omega=" + str(roundedomega) +  "_m=" + str(m_el) + "_sigma=" + str(sigma) 
  animation_name = 'momentum_animation_t_' + str(t) + '_N_' + str(N) + "_k=" + str(k) + "_theta=" + str(roundedtheta) + "_omega=" + str(roundedomega) +  "_m=" + str(m_el) + "_sigma=" + str(sigma) + '.gif'
  filepath = os.path.join(animation_dir, animation_name)
  from matplotlib.animation import PillowWriter

  ani.save(filepath, writer=PillowWriter(fps=fps)) 



def probability_check(t_max):
  # Function to check the code by confirming that the integral of j0 over all x as a function of t is the constant function 1
  T = np.linspace(0,t_max,int(t_max*20))
  def prob_check(t):
    prob = quad(lambda x, t : j0(x, t), -5*sigma - t, 5*sigma + t, args=(t))[0]
    return prob
  
  P = [prob_check(t) for t in T]
  plt.xlabel('Time [t]')
  plt.ylabel('Integral of wave func **2')
  plt.axis([0, t_max, 0, 1.2])
  plt.plot(T, P)
  plt.show()


def plotTrajectoryEnergies(t, x_init):
    T = np.linspace(0, t, 100*t)
    plt.figure(figsize=(12,9))
    plt.title(title)
    plt.xlabel('Time [t]')
    plt.ylabel('Energy ' + r'[$m_{el}\csc\Theta\sec\Omega$]')
    plt.ylim=[-40, 40]
    plt.axis([0, t, -40, 40])
    finalE = []
    imgname =  "energy_k=" + str(k) + "_theta=" + str(roundedtheta) + "_omega=" + str(roundedomega) + "_m=" + str(m_el) + "_sigma=" + str(sigma) + '.png'
    #filepath1 = "/Users/abbyperryman/Documents/Math Phys Research/ElectronTrajectoriesSpreadsheets/" + filename
    #filepath2 = "/Users/abbyperryman/Documents/Math Phys Research/ElectronTrajectoriesImages/" + imgname
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    image_dir = os.path.join(script_dir, "EnergyImages")
    filepath = os.path.join(image_dir, imgname)
    for x in x_init:
        E = []
        X = odeint(v, x, T)
        for i in range(len(X)):
            E.append(e(X[i], T[i]))
        finalE.append(float(E[-1]))
        plt.plot(T, E)
    print(finalE)
    plt.savefig(filepath)
    plt.show()
    return

def plotTrajectoryLogMomentum(t, x_init):
  T = np.linspace(0,t,100*t)
  plt.figure(figsize=(12,9))
  plt.title(title)
  plt.xlabel('Time [t]')
  plt.ylabel('sign(p) * log(|p| + 1)')
  plt.ylim=[-4, 4]
  plt.axis([0, t, -4, 4])
  imgname =  "log_momentum_k=" + str(k) + "_theta=" + str(roundedtheta) + "_omega=" + str(roundedomega) +  "_m=" + str(m_el) + "_sigma=" + str(sigma) + '.png'
  #filepath1 = "/Users/abbyperryman/Documents/Math Phys Research/ElectronTrajectoriesSpreadsheets/" + filename
  #filepath2 = "/Users/abbyperryman/Documents/Math Phys Research/ElectronTrajectoriesImages/" + imgname
  script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
  image_dir = os.path.join(script_dir, "MomentumImages")
  filepath = os.path.join(image_dir, imgname)
  finalP = []
  for x in x_init:
    P = []
    X = odeint(v,x,T) 
    for i in range(len(X)):
        p_val = p(X[i], T[i])
        altered_p = np.sign(p_val) * log(absolute(p_val) + 1)
        P.append(altered_p)
    finalP.append(float(P[-1]))
    plt.plot(T,P)
  print(finalP)
  plt.savefig(filepath)
  plt.show()
  return

def plotTrajectoryLogEnergy(t, x_init):
    T = np.linspace(0, t, 100*t)
    plt.figure(figsize=(12,9))
    plt.title(title)
    plt.xlabel('Time [t]')
    plt.ylabel('sign(E) * log(|E| + 1)')
    plt.ylim=[-4, 4]
    plt.axis([0, t, -4, 4])
    finalE = []
    imgname =  "log_energy_k=" + str(k) + "_theta=" + str(roundedtheta) + "_omega=" + str(roundedomega) +  "_m=" + str(m_el) + "_sigma=" + str(sigma) + '.png'
    #filepath1 = "/Users/abbyperryman/Documents/Math Phys Research/ElectronTrajectoriesSpreadsheets/" + filename
    #filepath2 = "/Users/abbyperryman/Documents/Math Phys Research/ElectronTrajectoriesImages/" + imgname
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    image_dir = os.path.join(script_dir, "EnergyImages")
    filepath = os.path.join(image_dir, imgname)
    for x in x_init:
        E = []
        X = odeint(v, x, T)
        for i in range(len(X)):
            e_val = e(X[i], T[i])
            altered_e = np.sign(e_val) * log(absolute(e_val) + 1)
            E.append(altered_e)
        finalE.append(float(E[-1]))
        plt.plot(T, E)
    print(finalE)
    plt.savefig(filepath)
    plt.show()
    return

def check_starting_omega(x_init):
   O_list = []
   for x in x_init:
      (R, T, O, P) = polar_decomp(x, 0)
      O_list.append(O)
   return O_list
    

# Put whatever functions you want to run here



print("--- %s seconds ---" % (time.time() - start_time))


     


print(title)



