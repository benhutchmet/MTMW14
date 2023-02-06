# (will be) a python notebook containing all the functions needed for ocean ROM coursework

# import the relevant modules
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

# set up the global variables - these don't change
b_0 = 2.5
gamma = 0.75
c = 1
r = 0.25
alpha = 0.125
xi_2 = 0
mu_c = 2/3 # critical value of coupling parameter

# set up the non-dimensionalised constants
Tnd = 7.5 # (K) SST anomalies
hnd = 150 # (m) thermocline depth
tnd = 2   # (months) time-scale

# define the functions needed for Task_A

# write a function for the simple explicit euler function
def euler(T_init, h_init, mu, e_n, xi_1, dt, nt):
    """
    Explicit euler scheme for solving the ocean recharge oscillator model using finite differences.
    -----------------
    Inputs:
    T_init: initial temperature (K).
    h_init: initial thermocline depth (m).
    mu: coupling coefficient.
    e_n: degree of nonlinearity of the ROM.
    xi_1: random wind stress forcing term.
    dt: time step size.
    nt: number of time steps.
    -----------------
    Returns:
    T_e: array of redimensionalised SST anomaly values (K).
    h_w: array of redimensionalised thermocline depth values (m).
    time: array of redimensionalised time steps.
    """
    
    # define the variables which vary with mu
    b = b_0*mu
    R = gamma*b - c
    
    # initialize arrays to store the results
    time = np.zeros(nt)
    T_e = np.zeros(nt)
    h_w = np.zeros(nt)

    # set the initial conditions (non-dimensionalised)
    T_e[0] = T_init/Tnd
    h_w[0] = h_init/hnd
    time[0] = 0/tnd

    # step forward in time with the explicit euler scheme
    for i in range(1,nt):

        time[i] = (i * dt)/tnd # non-dimensionalised time
        
        # for T - SST anomaly scale
        T_e[i] = T_e[i-1] + dt*(R*T_e[i-1] + gamma*h_w[i-1] - e_n*(h_w[i-1] + b*T_e[i-1])**3 + gamma*xi_1 + xi_2)

        # for h - thermocline depth scale
        h_w[i] = h_w[i-1] + dt*(-r*h_w[i-1] - alpha*b*T_e[i-1] - alpha*xi_1)
        
    # redimensionalise the results
    T_e = Tnd*T_e
    h_w = hnd/10*h_w # in units of 10m
    time = tnd*time
    
    return T_e, h_w, time
    
    
# write a function for the modified euler (adams-bashforth) scheme
def adams_bashforth(T_init, h_init, mu, e_n, xi_1, dt, nt):
    """
    Modified euler (Adams-Bashforth) scheme for modelling the ocean recharge oscillator model using finite differences.
    -----------------
    Inputs:
    T_init: initial temperature (K).
    h_init: initial thermocline depth (m).
    mu: coupling coefficient.
    e_n: degree of nonlinearity of the ROM.
    xi_1: random wind stress forcing term.
    dt: time step size.
    nt: number of time steps.
    -----------------
    Returns:
    T_e: array of redimensionalised SST anomaly values (K).
    h_w: array of redimensionalised thermocline depth values (m).
    time: array of redimensionalised time steps.
    """

    # define the variables which vary with mu
    b = b_0*mu
    R = gamma*b - c
    
    # initialize arrays to store the results
    time = np.zeros(nt)
    T_e = np.zeros(nt)
    h_w = np.zeros(nt)

    # set the initial conditions (non-dimensionalised)
    T_e[0] = T_init/Tnd
    h_w[0] = h_init/hnd
    time[0] = 0/tnd
    
    # calculate the 1th value using the explicit euler
    T_e[1] = T_e[0] + dt*(R*T_e[0] + gamma*h_w[0] - e_n*(h_w[0] + b*T_e[0])**3 + gamma*xi_1 + xi_2)
    h_w[1] = h_w[0] + dt*(-r*h_w[0] - alpha*b*T_e[0] - alpha*xi_1)
    time[1] = dt
    
    # step forward in time with the Adams-Bashforth scheme
    for i in range(2,nt):
        
        # time variable
        time[i] = (i * dt)/tnd # non-dimensionalised time
        
        # for T - SST anomaly scale
        T_e[i] = T_e[i-1] + dt*(3/2*(R*T_e[i-1] + gamma*h_w[i-1] - e_n*(h_w[i-1] + b*T_e[i-1])**3 + gamma*xi_1 + xi_2) - 1/2*(R*T_e[i-2] + gamma*h_w[i-2] - e_n*(h_w[i-2] + b*T_e[i-2])**3 + gamma*xi_1 + xi_2))

        # for h - thermocline depth scale
        h_w[i] = h_w[i-1] + dt*(-3/2*(r*h_w[i-1] + alpha*b*T_e[i-1] + alpha*xi_1) + 1/2*(r*h_w[i-2] + alpha*b*T_e[i-2] - alpha*xi_1))
        
    # redimensionalise the results
    T_e = Tnd*T_e
    h_w = hnd/10*h_w # in units of 10m
    time = tnd*time
    
    return T_e, h_w, time
    
    
# define a function for the implicit trapezoidal scheme
def trapezoidal(T_init, h_init, mu, e_n, xi_1, dt, nt):
    """
    Implicit trapezoidal finite difference scheme for modelling the ocean recharge oscillator model.
    -----------------
    Inputs:
    T_init: initial temperature (K).
    h_init: initial thermocline depth (m).
    mu: coupling coefficient.
    e_n: degree of nonlinearity of the ROM.
    xi_1: random wind stress forcing term.
    dt: time step size.
    nt: number of time steps.
    -----------------
    Returns:
    T_e: array of redimensionalised SST anomaly values (K).
    h_w: array of redimensionalised thermocline depth values (m).
    time: array of redimensionalised time steps.
    """
    
    # define the variables which vary with mu
    b = b_0*mu
    R = gamma*b - c
    
    # initialize arrays to store the results
    time = np.zeros(nt)
    T_e = np.zeros(nt)
    h_w = np.zeros(nt)

    # set the initial conditions (non-dimensionalised)
    T_e[0] = T_init/Tnd
    h_w[0] = h_init/hnd
    time[0] = 0/tnd

    # step forward in time with the general implementation of the implicit trapezoidal scheme
    for i in range(1,nt):
        
        # time variable
        time[i] = (i * dt)/tnd # non-dimensionalised time
        
        # for T - SST anomaly scale
        T_e[i] = T_e[i-1] + dt/2*((R*T_e[i] + gamma*h_w[i] - e_n*(h_w[i] + b*T_e[i])**3 + gamma*xi_1 + xi_2) + (R*T_e[i-1] + gamma*h_w[i-1] - e_n*(h_w[i-1] + b*T_e[i-1])**3 + gamma*xi_1 + xi_2))

        # for h - thermocline depth scale
        h_w[i] = h_w[i-1] - dt/2*((r*h_w[i] + alpha*b*T_e[i] + alpha*xi_1) + (r*h_w[i-1] + alpha*b*T_e[i-1] + alpha*xi_1))
        
    # redimensionalise the results
    T_e = Tnd*T_e
    h_w = hnd/10*h_w  # in units of 10m
    time = tnd*time
    
    return T_e, h_w, time

# define the two functions used for the RK4 scheme of the ROM in task A

# define the function of ODEs
def enso(y, t, mu, e_n, xi_1, annual_mu=False):
    """
    Define the system of ODEs for the recharge oscillator model of ENSO.
    -----------------
    Inputs:
    y: array of state variables [T, h] (K, m).
    t: time.
    mu: coupling coefficient.
    e_n: degree of nonlinearity.
    annual_mu: if/else to determine whether or not to vary mu on annual cycle.
    -----------------
    Returns:
    Array[dTdt, dSdt]: an array of temperature and thermocline depth.
    """
    
    # define the constants for all of the tasks
    mu_0 = 0.75
    mu_ann = 0.2
    tau = 12/tnd # months non-dimensionalised
    
    # define an if statement for determining which mu to use
    if annual_mu:
        # specify the coupling paramater to vary on an annual cycle
        mu = mu_0 * (1 + mu_ann * np.cos(2 * np.pi * t / tau - 5 * np.pi / 6))
    else:
        mu = mu

    # define the parameters that vary with mu
    b = b_0*mu
    R = gamma*b - c

    # define the ODEs
    dTdt = R*y[0] + gamma*y[1] - e_n*(y[1] + b*y[0])**3 + gamma*xi_1 + xi_2
    dSdt = -r*y[1] - alpha*b*y[0] - alpha*xi_1

    return np.array([dTdt, dSdt])
    
    
# define the function for the fourth-order Runge-Kutta scheme
def rk4(f, y0, t, nt, xi_1_forcing=False):
    """
    Solve a system of ordinary differential equations using the RK4 method.
    -----------------
    Inputs:
    f: function that defines the system of ODEs.
    y0: initial condition.
    t: array of equally spaced time points for which to solve the ODE.
    nt: number of time steps.
    xi_1_forcing: if/else to determine whether or not to apply random wind stress forcing.
    -----------------
    Returns:
    y: array containing dimensionalised values of the SST anomalies (K) and thermocline depth (m).
    """
    
    # define the variables 
    f_ann = 0.02
    f_ran = 0.2
    tau_cor = 1/60 # 1 day non-dimensionalised
    tau = 12/2 # months non-dimensionalised
    
    # set up the random numbers
    W = np.random.uniform(-1, 1, nt)

    # initialize the parameters for the loop
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    # run the for loop
    for i in range(n - 1):
    
        # specify whether or not to apply random wind stress forcing
        if xi_1_forcing:
            xi_1 = f_ann * np.cos(2 * np.pi * t[i] / tau) + f_ran * W[i] * (
                tau_cor / dt)
        else:
            xi_1 = 0
        
        h = t[i + 1] - t[i]
        k1 = h * f(y[i], t[i])
        k2 = h * f(y[i] + 0.5 * k1, t[i] + 0.5 * h)
        k3 = h * f(y[i] + 0.5 * k2, t[i] + 0.5 * h)
        k4 = h * f(y[i] + k3, t[i + 1])
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
    # dimensionalise the results
    y[:, 0] = y[:, 0] * Tnd # for the SST anomalies (K)
    y[:, 1] = y[:, 1] * hnd/10 # for the thermocline depth (m)
    
    return y
    


def Task_A_plotting(dt=0.02, no_periods=1, len_period=41):
    """Function which returns the plots for Task A...
    --------------
    Inputs:
    dt: the size of the time step.
    no_periods: the number of periods at mu = critical value.
    len_period: the length of the period (41 months for condition above.
    
    Returns:
    Plots of the SST anomalies and thermocline depth for the simple schemes and the RK4 scheme.
    """

    # define the initial conditions required for plotting in task A
    
    # time step parameters
    nt = int(no_periods*len_period/dt)
    t = np.linspace(0,no_periods*len_period,nt)/tnd # months non-dimensionalised
    
    # parameters
    e_n = 0
    xi_1 = 0
    xi_2 = 0
    mu = mu_c = 2/3

    # initial conditions non-dimensionalised
    T_init = 1.125/Tnd
    h_init = 0/hnd
    # for the RK4 scheme
    y0 = [1.125 / 7.5, 0 / 150]

    # run the simple schemes
    sst_anomaly_euler, thermocline_depth_euler, time_euler = euler(T_init, h_init, mu, e_n, xi_1, dt, nt)
    sst_anomaly_AB, thermocline_depth_AB, time_AB = adams_bashforth(T_init, h_init, mu, e_n, xi_1, dt, nt)
    sst_anomaly_trapezoid, thermocline_depth_trapezoid, time_trapezoid = trapezoidal(T_init, h_init, mu, e_n, xi_1, dt, nt)
    # run the RK4 scheme
    y = rk4(lambda y, t: enso(y, t, mu, e_n, xi_1, annual_mu=False), y0, t, nt, xi_1_forcing=False)
    sst_anomaly_rk4 = y[:, 0]
    thermocline_depth_rk4 = y[:, 1]
    time_rk4 = t * 2 # months re-dimensionalised

    # set up the figure as a 2x3 grid where the top row will share one y-axis and the bottom row will share another y-axis
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    # on the top rows, we will plot the time series for euler, trapezoidal and AB/RK4
    axs[0].plot(time_euler, sst_anomaly_euler, label='Euler T ($^{\circ}$C)')
    # plot the euler thermocline depths
    axs[0].plot(time_euler, thermocline_depth_euler, label='Euler h (10m)')
    # legend for the top row
    axs[0].legend(loc='upper right')
    # x-axis label for the top row
    axs[0].set_xlabel('Time (months)')

    # plot the time series for trapezoidal
    axs[2].plot(time_trapezoid, sst_anomaly_trapezoid, label='Trapezoidal T ($^{\circ}$C)')
    # plot the trapezoidal thermocline depths
    axs[2].plot(time_trapezoid, thermocline_depth_trapezoid, label='Trapezoidal h (10m)')
    # legend for the top row
    axs[2].legend(loc='upper right')
    # x-axis label for the top row
    axs[2].set_xlabel('Time (months)')

    # plot the time series for AB and rK4
    axs[1].plot(time_AB, sst_anomaly_AB, label='AB T ($^{\circ}$C)')
    # plot the AB thermocline depths
    axs[1].plot(time_AB, thermocline_depth_AB, label='AB h (10m)')
    # legend for the top row
    axs[1].legend(loc='upper right')
    # x-axis label for the top row
    axs[1].set_xlabel('Time (months)')

    # plot the time series for RK
    axs[3].plot(time_rk4, sst_anomaly_rk4, label='RK4 T ($^{\circ}$C)')
    # plot the RK4 thermocline depths
    axs[3].plot(time_rk4, thermocline_depth_rk4, label='RK4 h (10m)')
    # legend for the top row
    axs[3].legend(loc='upper right')
    # x-axis label for the top row
    axs[3].set_xlabel('Time (months)')

    # specify a suptitle for the plot
    plt.suptitle('ROM, $\mu = 2/3$, neutral, linear case')

    # show the first plot
    plt.show()

    # specify a tight layout
    fig.tight_layout()

    # save the figure
    fig.savefig('Task_A_time_series_plots.png', dpi=300)

    # on the bottom rows, we will plot the phase space for euler, trapezoidal and AB/RK4

    # set up a seperate figure for the phase space plots
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    # plot the phase space for euler with a colourbar to show time
    im = axs[0].scatter(sst_anomaly_euler, thermocline_depth_euler, c=time_euler, cmap='viridis',label='Euler')
    # legend for the bottom row
    axs[0].legend(loc='upper right')
    # label the x-axis
    axs[0].set_xlabel('Temperature ($^{\circ}$C)')
    # label the y-axis
    axs[0].set_ylabel('Thermocline depth (10m)')

    # plot the phase space for trapezoidal with a colourbar to show time
    im = axs[2].scatter(sst_anomaly_trapezoid, thermocline_depth_trapezoid, c=time_trapezoid, cmap='viridis', label='Trapezoidal')
    # legend for the bottom row
    axs[2].legend(loc='upper right')
    # label the x-axis
    axs[2].set_xlabel('Temperature ($^{\circ}$C)')
    # label the y-axis
    axs[2].set_ylabel('Thermocline depth (10m)')

    # plot the phase space for AB and RK4 with a colourbar to show time
    im = axs[1].scatter(sst_anomaly_AB, thermocline_depth_AB, c=time_AB, cmap='viridis', label='AB')
    # legend for the bottom row
    axs[1].legend(loc='upper right')
    # label the x-axis
    axs[1].set_xlabel('Temperature ($^{\circ}$C)')
    # label the y-axis
    axs[1].set_ylabel('Thermocline depth (10m)')

    # plot the phase space for RK4 with a colourbar to show time
    im = axs[3].scatter(sst_anomaly_rk4, thermocline_depth_rk4, c=time_rk4, cmap='viridis', label='RK4')
    # legend for the bottom row
    axs[3].legend(loc='upper right')
    # label the x-axis
    axs[3].set_xlabel('Temperature ($^{\circ}$C)')
    # label the y-axis
    axs[3].set_ylabel('Thermocline depth (10m)')

    # add a colourbar to the bottom row
    fig.colorbar(im, ax=axs[3], label='Time (months)')

    # specify a suptitle for the plot
    plt.suptitle('ROM, $\mu = 2/3$, neutral, linear case')

    # specify a tight layout
    plt.tight_layout()

    # show the plot
    plt.show()

    # save the figure
    fig.savefig('ROM_linear_neutral_final_test_phase space.png')

# test the taskA plotting
# Task_A_plotting()

# now for the second part of Task A, we will perform stability analysis
# calculate and plot the magnitude of the eigenvalues for the euler scheme
# use a function to calculate the eigenvalues for the simple euler scheme
def eigenvalue_analysis_euler(T_init, h_init, mu, e_n, xi_1, dt, nt):
    """
    Function to calculate the magnitude of the eigenvalues for the simple euler scheme.
    --------------------------------
    Input:
    T_init: initial temperature.
    h_init: initial thermocline depth.
    mu: parameter for the ROM.
    e_n: parameter for the ROM.
    xi_1: parameter for the ROM.
    dt: timestep.
    nt: number of timesteps.
    --------------------------------
    Returns:
    mag_eigenvalues: magnitude of the eigenvalues.
    """
    T_e, h_w, time = euler(T_init, h_init, mu, e_n, xi_1, dt, nt)

    # Initialize an array to store the magnitude of the eigenvalues
    mag_eigenvalues = np.zeros(nt)

    # define R
    b = b_0*mu
    R = gamma*b - c

    # Loop over each time step
    for i in range(1, nt):
        system_matrix = np.array([[1 + dt*R, dt*gamma], [dt*mu*h_w[i-1], 1 + dt*mu*R]])
        eigenvalues = np.linalg.eigvals(system_matrix)
        mag_eigenvalues[i] = np.abs(eigenvalues).max()

    return mag_eigenvalues

# define a function for the jacobian matrix of the simple euler scheme for the ROM
def jacobian(T, h, mu, e_n, xi_1):
    """
    Function to calculate the jacobian matrix for the simple euler scheme.
    --------------------------------
    Input:
    T: temperature.
    h: thermocline depth.
    mu: parameter for the ROM.
    e_n: parameter for the ROM.
    xi_1: parameter for the ROM.
    --------------------------------
    Returns:
    J: jacobian matrix.
    """
    # calculate the jacobian matrix
    J = np.array([[-mu * T, 0], [0, -mu * h]])

    return J

# write a function that will plot the magnitude of the eigenvalues for an array of timesteps (four different values of dt)
def plot_eigenvalues_euler(T_init, h_init, mu, e_n, xi_1, dt, nt):
    """
    Function to plot the magnitude of the eigenvalues for the simple euler scheme.
    --------------------------------
    Input:
    T_init: initial temperature.
    h_init: initial thermocline depth.
    mu: parameter for the ROM.
    e_n: parameter for the ROM.
    xi_1: parameter for the ROM.
    dt: timestep.
    nt: number of timesteps.
    --------------------------------
    Returns:
    None.
    """
    # create an array of values of dt
    dt_array = np.array([0.01, 0.1, 0.2, 1.0])

    # create an array to store the magnitude of the eigenvalues
    mag_eigenvalues = np.zeros((nt, len(dt_array)))

    # loop over each value of dt
    for i in range(len(dt_array)):
        # calculate the magnitude of the eigenvalues
        mag_eigenvalues[:, i] = eigenvalue_analysis_euler(T_init, h_init, mu, e_n, xi_1, dt_array[i], nt)

    # create an array of time
    time = np.linspace(0,41,nt)

    # create a figure
    fig = plt.figure(figsize=(10, 6))
    # create an axis
    ax = fig.add_subplot(111)
    # plot the magnitude of the eigenvalues for each value of dt
    ax.plot(time, mag_eigenvalues[:, 0], label='dt = 0.001')
    ax.plot(time, mag_eigenvalues[:, 1], label='dt = 0.01')
    ax.plot(time, mag_eigenvalues[:, 2], label='dt = 0.1')
    ax.plot(time, mag_eigenvalues[:, 3], label='dt = 1.0')
    # add a legend
    ax.legend(loc='upper right')
    # label the x-axis
    ax.set_xlabel('Time (months)')
    # label the y-axis
    ax.set_ylabel('Magnitude of the eigenvalues')
    # add a title
    ax.set_title('Magnitude of the eigenvalues for the simple euler scheme')
    # set y-axis limits
    ax.set_ylim(1.0, 1.3)
    # show the plot
    plt.show()
    # save the figure
    fig.savefig('ROM_linear_neutral_final_test_eigenvalues_euler.png')


# Pierre-Luigi-Vidale's code for stability analysis of the RK4 scheme
def rk4_stability():
    """
    Stability of rk4 for the linear recharge oscillator.
    """

    x = np.arange(-4,2,0.01)
    y = np.arange(-3,3,0.01)
    [X,Y] = np.meshgrid(x,y)
    z = X + 1j*Y

    A = 1 + z + 0.5*z**2 + 1/6*z**3 + 1/24*z**4
    Aabs = abs(A)

    plt.contourf(X, Y, Aabs, np.linspace(0,1,11))
    #    plt.xlabel('Re[$\lambda$]'); plt.ylabel('Im[$\lambda$]')
    plt.xlabel('$\Re(\lambda) \Delta t$'); plt.ylabel('$\Im(\lambda) \Delta t$')
    cbar = plt.colorbar()
    cbar.set_label("|A|", rotation=0)

# PLV's code for stability analysis of the Euler scheme (+ modification by BWH)
def Euler_stability():
    "Stability of rk4 for the linear recharge oscillator"

    x = np.arange(-4,2,0.01)
    y = np.arange(-3,3,0.01)
    [X,Y] = np.meshgrid(x,y)
    z = X + 1j*Y

    # CPL for the linear recharge oscillator euler scheme
    A = 1 + z
    Aabs = abs(A)

    plt.contourf(X, Y, Aabs, np.linspace(0,1,11))
    #    pl.xlabel('$\lambda \Delta t$'); plt.ylabel('$\omega \Delta t$ ')
    plt.xlabel('$\Re(\lambda) \Delta t$'); plt.ylabel('$\Im(\lambda) \Delta t$')
    cbar = plt.colorbar()
    cbar.set_label("|A|", rotation=0)

# define a function for the stability analysis in Task A
def Task_A_stability_analysis(dt=0.02, no_periods=1, len_period=41):
    """
    Function to perform stability analysis for the simple euler scheme.
    --------------------------------
    Input:
    None.
    --------------------------------
    Returns:
    None.
    """
    # define the initial conditions
    T_init = 1.125/Tnd
    h_init = 0.0/hnd
    # define the parameters
    mu = mu_c = 2/3
    e_n = 0.0
    xi_1 = 0.0
    # define the timestep
    dt = dt
    # define the number of timesteps
    nt = int(no_periods*len_period/dt)
    # call the function to plot the magnitude of the eigenvalues
    plot_eigenvalues_euler(T_init, h_init, mu, e_n, xi_1, dt, nt)

    # call the function to plot the stabiity of the RK4 scheme
    #rk4_stability()

    # call the function to plot the stabiity of the Euler scheme
    Euler_stability()

# test the stability analysis
Task_A_stability_analysis()






























# copy format from Hette for Task G ensemble plotting

#plot_ensemble(RK4, T_init, h_init, dt, nt=nt, mu=mu, params=params, n_members=n_members, mu_ann=mu_ann T_a=T_a, h_a=h_a...
#%%
