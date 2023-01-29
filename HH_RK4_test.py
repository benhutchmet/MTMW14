# test scheme from mr houtman

# import the relevant modules
import numpy as np
import matplotlib.pyplot as plt

# define the RK4 scheme
def rk4(f, y0, t):
    """
    Solve a system of ordinary differential equations using the RK4 method.
    f: function that defines the system of ODEs.
    y0: initial condition.
    t: array of equally spaced time points for which to solve the ODE.
    """
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = h * f(y[i], t[i])
        k2 = h * f(y[i] + 0.5 * k1, t[i] + 0.5 * h)
        k3 = h * f(y[i] + 0.5 * k2, t[i] + 0.5 * h)
        k4 = h * f(y[i] + k3, t[i + 1])
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


# define the function of ODEs
def enso(y, t):
    """
    Define the system of ODEs for the recharge oscillator model of ENSO.
    y: array of state variables [T, S].
    t: time.
    """

    # define the constants
    mu_c = 2 / 3  # critical value for the coupling parameter
    mu = mu_c  # set the coupling value to its critical parameter
    b_0 = 2.5  # high end value for the coupling parameter
    b = b_0 * mu  # measure of the thermocline slope
    gamma = 0.75  # feedback of the thermocline gradient on the SST gradient
    c = 1  # damping rate of SST anomalies
    R = gamma * b - c  # describes the Bjerknes positive feedback process
    r = 0.25  # damping of upper ocean heat content
    alpha = 0.125  # relates enhanced easterly wind stress to the recharge of ocean heat content
    e_n = 0  # degree of nonlinearity of ROM
    xi_1 = 0  # random wind stress forcing added to the system
    xi_2 = 0  # random heating added to the system

    # define the simple ODEs for Task A
    dTdt = R * y[0] + gamma * y[1]
    dSdt = -r * y[1] - alpha * b * y[0]

    return np.array([dTdt, dSdt])


# Example usage:

# define t
dt = 0.02
nt = int(41 / 0.02)
t = np.linspace(0, 41, nt)
# non-dimensionaliseed initial conditions
y0 = [1.125 / 7.5, 0 / 150]  # initial condition: T = 0.5, S = 0.5
y = rk4(enso, y0, t)
# redimensionalise the output
Te_rd = y[:, 0] * 7.5  # for the SST temperature
hw_rd = y[:, 1] * 150  # for the thermocline depth

# plot the time series
plt.plot(t, Te_rd, label='$T_E$')
plt.plot(t, hw_rd, label='$h_w$')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.show()