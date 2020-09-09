from numpy import sin, cos
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Pendulum Set up
class pendulum:
   def __init__(self,l,m,c,k):
       self.l = l
       self.m = m
       self.c = c
       self.k = k

# l: initial length of pendulum 1 in m
# m: mass of pendulum 1 in kg
# c: Damping of the joint
# k: Spring constant of the pendulum rod

pen1 = pendulum(1,1,0,1000000)

# Environmental Constant
g = 9.8  # acceleration due to gravity, in m/s^2

def derivs(state, t):

    dthdt = np.zeros_like(state)

    dthdt[0] = - g/(state[3]+pen1.l) * np.sin(state[1]) - 2 * state[2] * state[0] / (state[3]+pen1.l) - pen1.c/pen1.m * state[0]

    dthdt[1] = state[0]

    dthdt[2] = (pen1.l+state[3])*state[0]**2 - pen1.k / pen1.m * state[3] + g * np.cos(state[1])

    dthdt[3] = state[2]

    return dthdt

#time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0, 20, dt)

# initial conditions
# th is initial angle,  w is initial angular velocitie
# l0 is the initial length of the rod, v0 is the initial longitudial velocity of the pendulum
w0 = 0
th0 = 120
v0 = 0
l0 = 0

# initial value for state vectors
state = [np.radians(w0),np.radians(th0),v0, l0]

# integrate ODE to obtain the angle values
th = integrate.odeint(derivs, state, t)

x = (pen1.l+th[:,3])*sin(th[:, 1])
y = -(pen1.l+th[:,3])*cos(th[:, 1])


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