from numpy import sin, cos
import numpy as np
import scipy.integrate as integrate
import matplotlib.animation as animation
import matplotlib.pyplot as plt

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

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [0, x[i]]
    thisy = [0, y[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text


ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
                              interval=dt*1000, blit=True, init_func=init)
plt.show()