# file for working on the ensemble run task G

# import the necessary modules
import numpy as np
import matplotlib.pyplot as plt

# define the global variables

# non-dimensionalised parameters

T_nd = 7.5 # SST anomaly-scale (kelvin)
h_nd = 150 # thermocline depth scale (m)
t_nd = 2 # time-scale - 2 months in seconds

# define the variables for Task G - non-linear, stochastic forcing

mu_c = 2/3 # critical value for the coupling parameter
b_0 = 2.5 # high end value for the coupling parameter
gamma = 0.75 # feedback of the thermocline gradient on the SST gradient
c = 1 # damping rate of SST anomalies
r = 0.25 # damping of upper ocean heat content
alpha = 0.125 # relates enhanced easterly wind stress to the recharge of ocean heat content
xi_2 = 0.0 # random heating added to the system

# define the enso Task F scheme which includes the coupling parameter
def enso_taskF(y, t):
    """RK4 scheme for task E where the coupling parameter varies with time"""

    # define the global variables for Task F
    e_n = 0.1  # include non-linearity for task F
    mu_0 = 0.75  # set to 0.75 for task F
    mu_ann = 0.2
    tau = 12 / 2  # months non-dimensionalised

    # define the coupling parameter
    mu = mu_0 * (1 + mu_ann * np.cos(2 * np.pi * t / tau - 5 * np.pi / 6))

    # define the thermocline slope
    b = b_0 * mu

    # define the Bjerknes positive feedback process
    R = gamma * b - c

    # define the ODEs
    dTdt = R * y[0] + gamma * y[1] - e_n * (
                y[1] + b * y[0]) ** 3 + gamma * xi_1 + xi_2
    dSdt = -r * y[1] - alpha * b * y[0] - alpha * xi_1

    return np.array([dTdt, dSdt])

# define the function for the fourth-order Runge-Kutta scheme
def rk4_taskF(f, y0, t):
    """
    Solve a system of ordinary differential equations using the RK4 method.
    f: function that defines the system of ODEs.
    y0: initial condition.
    t: array of equally spaced time points for which to solve the ODE.
    """
    # define the variables for Task F
    f_ann = 0.02
    f_ran = 0.2
    tau_cor = 1 / 60  # 1 day non-dimensionalised
    tau = 12 / 2  # months non-dimensionalised

    # set up the random numbers
    W = np.random.uniform(-1, 1, nt)

    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(n - 1):
        # specify the random wind stress forcing
        xi_1 = f_ann * np.cos(2 * np.pi * t[i] / tau) + f_ran * W[i] * (
                    tau_cor / dt)

        # the rest of the RK4 scheme
        h = t[i + 1] - t[i]
        k1 = h * f(y[i], t[i])
        k2 = h * f(y[i] + 0.5 * k1, t[i] + 0.5 * h)
        k3 = h * f(y[i] + 0.5 * k2, t[i] + 0.5 * h)
        k4 = h * f(y[i] + k3, t[i + 1])
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y

# set up the time step for the RK4 scheme
dt = 1/60 # 1 day non-dimensionalised 2 months = 60 days
nt = int(5*41/dt) # 5 periods of 41 months each
t = np.linspace(0, 5*41, nt)/2 # non-dimensionalised time

# set up the initial conditions
# for the ensemble runs we will vary the values of the initial conditions
# for the SST and the thermocline slope
T_0 = np.linspace(-0.5, 0.5, 11)/7.5 # SST anomaly non-dimensionalised

# set up the initial conditions for the thermocline slope
h_0 = np.linspace(-0.5, 0.5, 11)/150 # thermocline slope non-dimensionalised

# set up the initial conditions in an array of y0
# set up the array of zeros for y0
y0 = np.zeros((len(T_0), len(h_0), 2))

# run the for loop to set up the initial conditions
for i in range(len(T_0)):
    for j in range(len(h_0)):
        y0[i, j] = np.array([T_0[i], h_0[j]])

# initialize each ensemble member for the SST and thermocline slope
T = np.zeros((len(T_0), len(h_0), nt))
h = np.zeros((len(T_0), len(h_0), nt))

# run the for loop to solve the ODEs for each ensemble member
for i in range(len(T_0)):
    for j in range(len(h_0)):
        y = rk4_taskF(enso_taskF, y0[i, j], t)
        # redimensionalise these results
        T[i, j] = y[:, 0] * T_nd
        h[i, j] = y[:, 1] * h_nd

# plot the results
# set up the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# we want to plot the time series of SST anomalies for each ensemble member
# on the same plot so we will use a for loop
# we want these plots to be in the form of plume diagrams

# set up the colour map
cmap = plt.get_cmap('viridis')
# set up the colour range
c_range = np.linspace(0, 1, len(T_0))

# set up the for loop to plot each ensemble member
for i in range(len(T_0)):
    for j in range(len(h_0)):
        ax.plot(t, T[i, j], color=cmap(c_range[i]), alpha=0.5)

# set up the axes
ax.set_xlabel('Time (months)')
ax.set_ylabel('SST anomaly (K)')
ax.set_title('SST anomaly time series for each ensemble member')

plt.show()

# plot the results
# set up the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# we want to plot the time series of SST anomalies for each ensemble member
# on the same plot so we will use a for loop
# we want these plots to be in the form of plume diagrams

# set up the colour map
cmap = plt.get_cmap('viridis')
# set up the colour range
c_range = np.linspace(0, 1, len(T_0))

# set up the for loop to plot each ensemble member
for i in range(len(T_0)):
    for j in range(len(h_0)):
        ax.plot(T[i, j], h[i, j], color=cmap(c_range[i]), alpha=0.5)

# set up the axes
ax.set_xlabel('SST anomaly (K)')
