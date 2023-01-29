# bringing all the schemes together and plotting the output

# import the relevant libraries
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
alpha = 0.125 # relates enhanced easterly wind stress to the recharge of
# ocean heat content
e_n = 0 # degree of nonlinearity of ROM
xi_1 = 0 # random wind stress forcing added to the system
xi_2 = 0 # random heating added to the system

# define the schemes

# define the function for the explicit euler scheme
def euler(T_init, h_init, mu, R, gamma, e_n, xi_1, xi_2, dt, nt):
    """Explicit euler scheme for solving the ocean recharge oscillator model
    using finite differences. Takes T, the SST anomaly, h, the thermocline depth
    scale, t, the time scale, mu, the coupling value, R, the bjerkness positive
    feedback, gamma, the feedback of the thermocline gradient on the SST
    gradient, e_n, the degree of nonlinearity xi_1 and xi_2, the random wind
    stress and random heating added to the system, dt, the time step size and
    nt, the number of time steps."""

    # initialize arrays to store the results
    time = np.zeros(nt)
    T_e = np.zeros(nt)
    h_w = np.zeros(nt)

    # set the initial conditions
    T_e[0] = T_init
    h_w[0] = h_init
    time[0] = 0

    # step forward in time with the explicit euler scheme
    for i in range(1, nt):
        time[i] = i * dt

        # for T - SST anomaly scale
        T_e[i] = T_e[i - 1] + dt * (R * T_e[i - 1] + gamma * h_w[i - 1])

        # for h - thermocline depth scale
        h_w[i] = h_w[i - 1] + dt * (r * h_w[i - 1] - alpha * b * T_e[i - 1])

    return T_e, h_w, time

# define the adams-bashforth (modified euler) scheme
def adams_bashforth(T_init, h_init, mu, R, gamma, e_n, xi_1, xi_2, dt, nt):
    """Modified euler scheme to give Adams-Bashforth time scheme for modelling the ocean recharge oscillator model using finite difference. Should give second order accuracy. Takes T, the SST anomaly, h, the thermocline depth scale, t, the time scale, mu, the coupling value, R, the bjerknes positive feedback, gamma, the feedback of the thermocline gradient on  the SST gradient, e_n, the degree of nonlinearity xi_1 and xi_2, the random wind stress and random heating added to the system, dt, the size of the timestep and nt, the number of timesteps."""

    # initialize arrays to store the results
    time = np.zeros(nt)
    T_e = np.zeros(nt)
    h_w = np.zeros(nt)

    # set the initial conditions
    T_e[0] = T_init
    h_w[0] = h_init

    # calculate the 1th value using the explicit euler
    T_e[1] = T_e[0] + dt * (R * T_e[0] + gamma * h_w[0])
    h_w[1] = h_w[0] + dt * (r * h_w[0] + alpha * b * T_e[0])

    # step forward in time with the Adams-Bashforth scheme
    for i in range(2, nt):
        # time variable
        time[i] = i * dt

        # for T - SST anomaly scale
        T_e[i] = T_e[i - 1] + dt * (
                    3 / 2 * (R * T_e[i - 1] + gamma * h_w[i - 1]) - 1 / 2 * (
                        R * T_e[i - 2] + gamma * h_w[i - 2]))

        # for h - thermocline depth scale
        h_w[i] = h_w[i - 1] + dt * (-3 / 2 * (
                    r * h_w[i - 1] + alpha * b * T_e[i - 1]) + 1 / 2 * (
                                                r * h_w[i - 2] + alpha * b *
                                                T_e[i - 2]))

    return T_e, h_w, time

# define the implicit trapezoidal scheme
def trapezoidal(T_init, h_init, mu, R, gamma, e_n, xi_1, xi_2, dt, nt):
    """Implicit trapezoidal finite difference scheme for modelling the ocean
    recharge oscillator model. Takes T_init, the initial SST anomaly, h_init,
    the initial thermocline depth scale, mu, the coupling value, R, the bjerknes
     positive feedback, gamma, the feedback of the thermocline gradient on the
     SST gradient, e_n, the degree of nonlinearity xi_1 and xi_2, the random
     wind stress and random heating added to the system, dt, the size of the
     time steps and nt the number of time steps."""

    # initialize arrays to store the results
    time = np.zeros(nt)
    T_e = np.zeros(nt)
    h_w = np.zeros(nt)

    # set the initial conditions
    T_e[0] = T_init
    h_w[0] = h_init

    # step forward in time with the general implementation of the implicit
    # trapezoidal scheme
    for i in range(1, nt):
        # time variable
        time[i] = i * dt

        # for T - SST anomaly scale
        T_e[i] = T_e[i - 1] + dt / 2 * ((R * T_e[i] + gamma * h_w[i]) + (
                    R * T_e[i - 1] + gamma * h_w[i - 1]))

        # for h - thermocline depth scale
        h_w[i] = h_w[i - 1] - dt / 2 * ((r * h_w[i] + alpha * b * T_e[i]) + (
                    r * h_w[i - 1] + alpha * b * T_e[i - 1]))

    return T_e, h_w, time

# define the fourth order Runge-Kutta scheme
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
    # define the simple ODEs for Task A
    dTdt = R * y[0] + gamma * y[1]
    dSdt = -r * y[1] - alpha * b * y[0]

    return np.array([dTdt, dSdt])

# define the time-step parameters
dt = 0.02
nt = int(41 / 0.02)
t = np.linspace(0, 41, nt)

# define the non-dimensionalised initial conditions for the three schemes
# euler, AB2 and trapezoidal
T_init = 1.125/7.5
h_init = 0/150

# define the non-dimensionalised initial conditions for the RK4 scheme
y0 = [1.125 / 7.5, 0 / 150]
# initialize the RK4 scheme
y = rk4(enso, y0, t)

# run the other implementations: euler, AB2 and trapezoidal
sst_anomaly, thermocline_depth, time = euler(T_init, h_init, mu, R, gamma, e_n,
                                             xi_1, xi_2, dt, nt)
sst_anomaly_ab, thermocline_depth_ab, time_ab = adams_bashforth(T_init, h_init,
                                                                mu, R, gamma, e_n, xi_1, xi_2, dt, nt)
sst_anomaly_tz, thermocline_depth_tz, time_tz = trapezoidal(T_init, h_init,
                                                            mu, R, gamma, e_n, xi_1, xi_2, dt, nt)

