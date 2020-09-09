from numpy import sin, cos
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Pendulum Set up
class pendulum:
   def __init__(self,l,m,c,g):
       self.l = l
       self.m = m
       self.c = c
       self.g = g

# l: initial length of pendulum 1 in m
# m: mass of pendulum 1 in kg
# c: Damping of the joint
# Environmental Constant: acceleration due to gravity, in m/s^2

def fitfunc(x, a, b, c):
    return a * np.sin(b * x) + c

pen1 = pendulum(1,1,0,9.8)

def derivs(state, t):

    dthdt = np.zeros_like(state)

    dthdt[0] = - pen1.g/pen1.l * np.sin(state[1])  - pen1.c/pen1.m * state[0]

    dthdt[1] = state[0]

    return dthdt

#time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0, 20, dt)

# initial conditions
# th is initial angle,  w is initial angular velocitie
# l0 is the initial length of the rod, v0 is the initial longitudial velocity of the pendulum
w0 = 0
th0 = 120

# initial value for state vectors
state = [np.radians(w0),np.radians(th0)]

# integrate ODE to obtain the angle values
th = integrate.odeint(derivs, state, t)

x = pen1.l*sin(th[:, 1])
y = -pen1.l*cos(th[:, 1])


popt, pcov = curve_fit(fitfunc, t, y)  # popt- fitting parameters, pcov- covariance
plt.plot(t, y, 'b-', label='data')
plt.plot(t, fitfunc(t, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc='upper right')

plt.show()

fig, axs = plt.subplots(1, 2)
plot1, = axs[0].plot(t, x)
axs[0].set_title('x over time')
axs[0].set_xlabel('time [s] ')
axs[0].set_ylabel('x [m]')
plot2, = axs[1].plot(t, y)
axs[1].set_title('y over time')
axs[1].set_xlabel('time [s] ')
axs[1].set_ylabel('y [m]')



plt.tight_layout()
plt.show()