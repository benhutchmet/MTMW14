# file for testing the RK4 scheme in isolation

# import the required libraries
import numpy as np
import matplotlib.pyplot as plt

# define the global variables

# non-dimensionalised parameters

T_nd = 7.5 # SST anomaly-scale (kelvin)
h_nd = 150 # thermocline depth scale (m)
t_nd = 2 # time-scale - 2 months

# define the variables for Task A - neutral linear (deterministic) ROM

mu_c = 2/3 # critical value for the coupling parameter
mu = mu_c # set the coupling value to its critical parameter
b_0 = 2.5 # high end value for the coupling parameter
b = b_0*mu # measure of the thermocline slope
gamma = 0.75 # feedback of the thermocline gradient on the SST gradient
c = 1 # damping rate of SST anomalies
R = gamma*b - c # describes the Bjerknes positive feedback process
r = 0.25 # damping of upper ocean heat content
alpha = 0.125 # relates enhanced easterly wind stress to the recharge of ocean heat content
e_n = 0 # degree of nonlinearity of ROM
xi_1 = 0 # random wind stress forcing added to the system
xi_2 = 0 # random heating added to the system

# define the initial conditions
T_e_init = 1.125/T_nd
h_w_init = 0/h_nd

# specify the timestep
dt = 0.02

# specify the number of time steps
nt = int(41/dt)

# initialize arrays to store the results
time = np.zeros(nt)
T_e = np.zeros(nt)
h_w = np.zeros(nt)

# define the function for f1 of the RK4 scheme
def f1(T_e, h_w):
    """Function for f1 of the RK4 scheme which models the ODE for dh_wdt. Takes
    T_e, the SST anomaly and h_w, the thermocline. Returns the value of f1."""

    f1 = -r*h_w - alpha*b*T_e - alpha*xi_1
    return f1

# define the function for f2 of the RK4 scheme

def f2(T_e, h_w):
    """Function for f2 of the RK4 scheme which models the ODE for dT_edt. Takes
    T_e, the SST anomaly and h_w, the thermocline as input . Returns the
    value of f2."""

    f2 = R*T_e + gamma*h_w - e_n*(h_w + b*T_e)**2 + gamma*xi_1 + xi_2
    return f2

# define the function for the RK4 scheme

def RK4(T_e, h_w, dt):
    """Function for the RK4 scheme which models the ODE for dh_wdt and dT_edt.
    Takes T_e, the SST anomaly, h_w, the thermocline, and dt, the time step size
    as input. Returns the value of T_e and h_w at the next time step,
    T_e_new and h_w_new."""

    k1 = f1(T_e, h_w)
    l1 = f2(T_e, h_w)

    k2 = f1(T_e + k1 * dt/2, h_w + l1 * dt/2)
    l2 = f2(T_e + k1 * dt/2, h_w + l1 * dt/2)

    k3 = f1(T_e + k2 * dt/2, h_w + l2 * dt/2)
    l3 = f2(T_e + k2 * dt/2, h_w + l2 * dt/2)

    k4 = f1(T_e + k3 * dt, h_w + l3 * dt)
    l4 = f2(T_e + k3 * dt, h_w + l3 * dt)

    T_e_new = T_e + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    h_w_new = h_w + dt/6 * (l1 + 2*l2 + 2*l3 + l4)

    return T_e_new, h_w_new



# set the initial conditions
T_e[0] = T_e_init
h_w[0] = h_w_init

# now run the RK4 scheme in a for loop
for i in range(1,nt-1):
    T_e[i], h_w[i] = RK4(T_e[i-1], h_w[i-1], dt)

    time[i] = (time[i-1] + dt)

# redimensionalise the results
T_e_rd = T_e*T_nd
h_w_rd = h_w*h_nd


# plot the results on two seperate y-axes
fig, ax1 = plt.subplots()
ax1.plot(time, T_e, 'b-')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('T_e', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')
ax2 = ax1.twinx()
ax2.plot(time, h_w, 'r-')
ax2.set_ylabel('h_w', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.show()

