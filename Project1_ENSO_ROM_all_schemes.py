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
sst_anomaly_rk, thermocline_depth_rk = y[:, 0], y[:, 1]

# run the other implementations: euler, AB2 and trapezoidal
sst_anomaly, thermocline_depth, time = euler(T_init, h_init, mu, R, gamma, e_n,
                                             xi_1, xi_2, dt, nt)
sst_anomaly_ab, thermocline_depth_ab, time_ab = adams_bashforth(T_init, h_init,
                                                                mu, R, gamma, e_n, xi_1, xi_2, dt, nt)
sst_anomaly_tz, thermocline_depth_tz, time_tz = trapezoidal(T_init, h_init,
                                                            mu, R, gamma, e_n, xi_1, xi_2, dt, nt)

# redimensionalise the results
# for all of the sst_anomaly results, multiply by 7.5
sst_anomaly = sst_anomaly * 7.5
sst_anomaly_ab = sst_anomaly_ab * 7.5
sst_anomaly_tz = sst_anomaly_tz * 7.5
sst_anomaly_rk = sst_anomaly_rk * 7.5

# for all of the thermocline_depth results, multiply by 150 to redimensionalise
# then divide by 10 to convert to units of 10m
thermocline_depth = thermocline_depth * 150/10
thermocline_depth_ab = thermocline_depth_ab * 150/10
thermocline_depth_tz = thermocline_depth_tz * 150/10
thermocline_depth_rk = thermocline_depth_rk * 150/10

# now plot the results as a time series
# specify labels for each of the schemes used
# first for a time series plot, for both the sst anomaly and thermocline depth
# using one y-axis
# specify two subplots in a 2x1 grid
fig, (ax1, ax2) = plt.subplots(1, 2)
# plot all of the SST anomaly results
#ax1.plot(time, sst_anomaly, label='Euler T ($^{\circ}$C)')
ax1.plot(time, sst_anomaly_ab, label='AB2 T ($^{\circ}$C)')
# specify a dashed linestyle for the trapezoidal scheme
#ax1.plot(time, sst_anomaly_tz, label='Trapezoidal T ($^{\circ}$C)')
ax1.plot(time, sst_anomaly_rk, label='RK4 T ($^{\circ}$C)')
# plot all of the thermocline depth results
#ax1.plot(time, thermocline_depth, label='Euler h (10m)')
ax1.plot(time, thermocline_depth_ab, label='AB2 h (10m)')
#ax1.plot(time, thermocline_depth_tz, label='Trapezoidal h (10m)')
ax1.plot(time, thermocline_depth_rk, label='RK4 h (10m)')
# set the plot title as '$\mu$ = 2/3, neutral, linear case'
ax1.title.set_text('Time series for T ($^{\circ}$C) and h (10m)')
# specify the legend in the top right hand corner
ax1.legend(loc='upper right')

# now plot the phase space plot for all of the schemes using ax2
#ax2.plot(sst_anomaly, thermocline_depth, label='Euler')
ax2.plot(sst_anomaly_ab, thermocline_depth_ab, label='AB2')
#ax2.plot(sst_anomaly_tz, thermocline_depth_tz, label='Trapezoidal')
ax2.plot(sst_anomaly_rk, thermocline_depth_rk, label='RK4')
# set the plot title as '$\mu$ = 2/3, neutral, linear case'
# set a font size of 8
ax2.title.set_text('Phase space for T ($^{\circ}$C) and h (10m)')
# specify the legend in the top right hand corner
ax2.legend(loc='upper right')

# specify a tight layout
plt.tight_layout()

# show the plot
plt.show()

# save the plot as 'combined_plot.png', specify high resolution png
plt.savefig('combined_plot.png')


def eigenvalue_analysis(T_init, h_init, mu, R, gamma, e_n, xi_1, xi_2, dt, nt):
    T_e, h_w, time = euler(T_init, h_init, mu, R, gamma, e_n, xi_1, xi_2, dt,
                           nt)

    # Initialize an array to store the magnitude of the eigenvalues
    mag_eigenvalues = np.zeros(nt)

    # Loop over each time step
    for i in range(1, nt):
        system_matrix = np.array(
            [[1 + dt * R, dt * gamma], [dt * mu * h_w[i - 1], 1 + dt * mu * R]])
        eigenvalues = np.linalg.eigvals(system_matrix)
        mag_eigenvalues[i] = np.abs(eigenvalues).max()

    print(mag_eigenvalues)
    # Plot the magnitude of the eigenvalues against the time step size
    # make the size of the scatter points smaller and the type as crosses


    plt.plot(t, mag_eigenvalues, 'o-')
    plt.xlabel('Time step size (dt)')
    plt.ylabel('Magnitude of eigenvalues')
    plt.title('Eigenvalue analysis of the ocean recharge oscillator model')
    # add a horizontal line at 1
    plt.axhline(y=1, color='r', linestyle='--')
    # set y limits between 0.8 and 1.5
    plt.ylim(0.8, 1.5)
    plt.show()
    plt.savefig('stability_analysis_explicit_euler.png')


# define the sub-critical and super critical values of \mu
# redefine the global variables

# define the global variables

# non-dimensionalised parameters

T_nd = 7.5 # SST anomaly-scale (kelvin)
h_nd = 150 # thermocline depth scale (m)
t_nd = 2*30*24*60*60 # time-scale - 2 months in seconds

# define the variables for task B - neutral linear (deterministic) ROM

mu_c = 2/3 # critical value for the coupling parameter
mu_subc = 1/3
mu = mu_subc # set the coupling value to its subcritical parameter
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

# run the RK4 scheme
# define the time-step parameters
dt = 0.02
nt = int(41 / 0.02)
t = np.linspace(0, 41, nt)

# define the non-dimensionalised initial conditions for the three schemes
# euler, AB2 and trapezoidal
T_init = 1.125/7.5
h_init = 0/150

# run the AB scheme

sst_anomaly_ab_subc, thermocline_depth_ab_subc, time_ab_subc = adams_bashforth(T_init, h_init,
                                                                mu, R, gamma, e_n, xi_1, xi_2, dt, nt)

# define the non-dimensionalised initial conditions for the RK4 scheme
y0 = [1.125 / 7.5, 0 / 150]
# initialize the RK4 scheme
y_subc = rk4(enso, y0, t)
sst_anomaly_rk_subc, thermocline_depth_rk_subc = y_subc[:, 0], y_subc[:, 1]

# now for the supercritical

mu_supc = 1
mu = mu_supc # set the coupling value to its subcritical parameter
b_0 = 2.5 # high end value for the coupling parameter
b = b_0*mu # measure of the thermocline slope
gamma = 0.75 # feedback of the thermocline gradient on the SST gradient
c = 1 # damping rate of SST anomalies
R = gamma*b - c # describes the Bjerknes positive feedback process

# run the AB scheme

sst_anomaly_ab_supc, thermocline_depth_ab_supc, time_ab_supc = adams_bashforth(T_init, h_init,
                                                                mu, R, gamma, e_n, xi_1, xi_2, dt, nt)

# define the non-dimensionalised initial conditions for the RK4 scheme
y0 = [1.125 / 7.5, 0 / 150]
# initialize the RK4 scheme
y_supc = rk4(enso, y0, t)
sst_anomaly_rk_supc, thermocline_depth_rk_supc = y_supc[:, 0], y_supc[:, 1]

# now plot the subcrcitical and supercritical cases for both time series and
# phase space plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(time_ab_subc, sst_anomaly_ab_subc, label='AB2 subc')
ax1.plot(time_ab_supc, sst_anomaly_ab_supc, label='AB2 supc')
ax1.plot(t, sst_anomaly_rk_subc, label='RK4 subc')
ax1.plot(t, sst_anomaly_rk_supc, label='RK4 supc')
ax1.set_xlabel('Time')
# set the title of the plot
ax1.set_title('Time series for T and h')
# specify the legend in the upper right corner
ax1.legend(loc='upper right')

ax2.plot(thermocline_depth_ab_subc, sst_anomaly_ab_subc, label='AB2 subc')
ax2.plot(thermocline_depth_ab_supc, sst_anomaly_ab_supc, label='AB2 supc')
ax2.plot(thermocline_depth_rk_subc, sst_anomaly_rk_subc, label='RK4 subc')
ax2.plot(thermocline_depth_rk_supc, sst_anomaly_rk_supc, label='RK4 supc')
# set the title of the plot
ax2.set_title('Phase space plot for T and h')
# specify the legend in the upper right corner
ax2.legend(loc='upper right')

# specify a super title for the whole figure
fig.suptitle('Subcritical and supercritical cases for the ROM')

plt.show()

# save the figure

# first define the global variables for the Task C experiment where we introduce non-linearity

mu_c = 2/3 # critical value for the coupling parameter
mu = mu_c # set the coupling value to its subcritical parameter
b_0 = 2.5 # high end value for the coupling parameter
b = b_0*mu # measure of the thermocline slope
gamma = 0.75 # feedback of the thermocline gradient on the SST gradient
c = 1 # damping rate of SST anomalies
R = gamma*b - c # describes the Bjerknes positive feedback process
r = 0.25 # damping of upper ocean heat content
alpha = 0.125 # relates enhanced easterly wind stress to the recharge of ocean heat content
# modify the degree of nonlinearity for task C
e_n = 0.1 # degree of nonlinearity of ROM
xi_1 = 0 # random wind stress forcing added to the system
xi_2 = 0 # random heating added to the system

# set up the time steps

# set up four different values of dt (0.01,0.02,0.1,0.3)
dt = [0.01, 0.02, 0.1, 0.3]
# set up the time steps to loop over the four values of dt
nt = [int(5*41 / dt[0]), int(5*41 / dt[1]), int(5*41 / dt[2]), int(5*41 / dt[3])]
# set up the time array for all four values of dt
t = [np.linspace(0, 5*41, nt[0]), np.linspace(0, 5*41, nt[1]), np.linspace(0, 5*41, nt[2]), np.linspace(0, 5*41, nt[3])]

# modify the code below to run with four different values of dt
# plotting the rsults for each dt in a single figure

# define the non-dimensionalised initial conditions for the AB2 scheme
T_init = 1.125/7.5
h_init = 0/150

# define the non-dimensionalised initial conditions for the RK4 scheme
y0 = [1.125 / 7.5, 0 / 150]

# initialize the RK4 scheme
#y_nl = rk4(enso, y0, t)
#sst_anomaly_rk_nl, thermocline_depth_rk_nl = y_nl[:, 0], y_nl[:, 1]
# inititialize the function y with all t's
y_nl = [rk4(enso, y0, t[0]), rk4(enso, y0, t[1]), rk4(enso, y0, t[2]), rk4(enso, y0, t[3])]
# initialize the function sst_anomaly_rk_nl with all t's
sst_anomaly_rk_nl = [y_nl[0][:, 0], y_nl[1][:, 0], y_nl[2][:, 0], y_nl[3][:, 0]]
# initialize the function thermocline_depth_rk_nl with all t's
thermocline_depth_rk_nl = [y_nl[0][:, 1], y_nl[1][:, 1], y_nl[2][:, 1], y_nl[3][:, 1]]

# initialize the AB2 scheme for all four values of dt
sst_anomaly_ab_nl = [adams_bashforth(T_init, h_init,mu, R, gamma, e_n, xi_1, xi_2, dt[0], nt[0])[0], adams_bashforth(T_init, h_init,mu, R, gamma, e_n, xi_1, xi_2, dt[1], nt[1])[0], adams_bashforth(T_init, h_init,mu, R, gamma, e_n, xi_1, xi_2, dt[2], nt[2])[0], adams_bashforth(T_init, h_init,mu, R, gamma, e_n, xi_1, xi_2, dt[3], nt[3])[0]]
thermocline_depth_ab_nl = [adams_bashforth(T_init, h_init,mu, R, gamma, e_n, xi_1, xi_2, dt[0], nt[0])[1], adams_bashforth(T_init, h_init,mu, R, gamma, e_n, xi_1, xi_2, dt[1], nt[1])[1], adams_bashforth(T_init, h_init,mu, R, gamma, e_n, xi_1, xi_2, dt[2], nt[2])[1], adams_bashforth(T_init, h_init,mu, R, gamma, e_n, xi_1, xi_2, dt[3], nt[3])[1]]
# initialize the time for all four values of dt
time_ab_nl = [adams_bashforth(T_init, h_init,mu, R, gamma, e_n, xi_1, xi_2, dt[0], nt[0])[2], adams_bashforth(T_init, h_init,mu, R, gamma, e_n, xi_1, xi_2, dt[1], nt[1])[2], adams_bashforth(T_init, h_init,mu, R, gamma, e_n, xi_1, xi_2, dt[2], nt[2])[2], adams_bashforth(T_init, h_init,mu, R, gamma, e_n, xi_1, xi_2, dt[3], nt[3])[2]]

# redimensioalise the results
sst_anomaly_ab_nl = [sst_anomaly_ab_nl[0]*7.5, sst_anomaly_ab_nl[1]*7.5, sst_anomaly_ab_nl[2]*7.5, sst_anomaly_ab_nl[3]*7.5]
thermocline_depth_ab_nl = [thermocline_depth_ab_nl[0]*150, thermocline_depth_ab_nl[1]*150, thermocline_depth_ab_nl[2]*150, thermocline_depth_ab_nl[3]*150]
sst_anomaly_rk_nl = [sst_anomaly_rk_nl[0]*7.5, sst_anomaly_rk_nl[1]*7.5, sst_anomaly_rk_nl[2]*7.5, sst_anomaly_rk_nl[3]*7.5]
thermocline_depth_rk_nl = [thermocline_depth_rk_nl[0]*150, thermocline_depth_rk_nl[1]*150, thermocline_depth_rk_nl[2]*150, thermocline_depth_rk_nl[3]*150]

# set up the sublplots as a 2x4 grid with the time series plots on the left and the phase plots on the right
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

# plot the time series plots
for i in range(4):
    axs[0, i].plot(time_ab_nl[i], sst_anomaly_ab_nl[i], label='AB2 $T_e$')
    axs[0, i].plot(time_ab_nl[i], thermocline_depth_ab_nl[i], label='AB2 $h$')
    axs[0, i].plot(time_ab_nl[i], sst_anomaly_rk_nl[i], label='RK4 $T_e$')
    axs[0, i].plot(time_ab_nl[i], thermocline_depth_rk_nl[i], label='RK4 $h$')
    axs[0, i].set_title('dt = ' + str(dt[i]))
    #axs[0, i].set_xlabel('time (years)')
    #axs[0, i].set_ylabel('dimensionless quantity')
    axs[0, i].legend()

# plot the phase plots
for i in range(4):
    axs[1, i].plot(sst_anomaly_ab_nl[i], thermocline_depth_ab_nl[i], label='AB2')
    axs[1, i].plot(sst_anomaly_rk_nl[i], thermocline_depth_rk_nl[i], label='RK4')
    axs[1, i].set_title('dt = ' + str(dt[i]))
    axs[1, i].set_xlabel('$T_e$')
    axs[1, i].set_ylabel('$h$')
    axs[1, i].legend()

# set the spacing between the subplots
fig.tight_layout()

# specify a suptitle for the whole figure
fig.suptitle('Non-linear model for the ROM for a range of dt', fontsize=12)

# save the figure
fig.savefig('nonlinear_model_AB2_RK4.png')


# redimensioalise the results
sst_anomaly_ab_nl = sst_anomaly_ab_nl*7.5
thermocline_depth_ab_nl = thermocline_depth_ab_nl*150
sst_anomaly_rk_nl = sst_anomaly_rk_nl*7.5
thermocline_depth_rk_nl = thermocline_depth_rk_nl*150

# redimensioalise the time
time_ab_nl = time_ab_nl*41
t = t*41

# plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# time series plot
ax1.plot(time_ab_nl, sst_anomaly_ab_nl, label='AB2 $T_e$')
ax1.plot(time_ab_nl, thermocline_depth_ab_nl, label='AB2 $h_w$')
ax1.plot(time_ab_nl, sst_anomaly_rk_nl, label='RK4 $T_e$',linestyle='--')
ax1.plot(time_ab_nl, thermocline_depth_rk_nl, label='RK4 $h_w$',linestyle='--')
ax1.set_xlabel('Time')
# set the title of the plot
ax1.set_title('Time series for T and h ($e_n = 0.1)')
# specify the legend in the upper right corner
ax1.legend(loc='upper right')

# plot the phase space results for the nonlinear case
ax2.plot(thermocline_depth_ab_nl, sst_anomaly_ab_nl, label='AB2')
ax2.plot(thermocline_depth_rk_nl, sst_anomaly_rk_nl, label='RK4',linestyle='--')
# set the title of the plot
ax2.set_title('Phase space plot for T and h ($e_n = 0.1$)')
# specify the legend in the upper right corner
ax2.legend(loc='upper right')

# specify a super title for the whole figure including dt = 'value of dt'
fig.suptitle('Nonlinear cases for the ROM (dt = %s' % dt + ')')

plt.show()

# save the figure
fig.savefig('nonlinear_cases_RK4_AB2.png')
fig.savefig('subc_and_supc_cases.png')


# define the function of ODEs
def enso_RK4(y, t, mu, e_n, xi_1, xi_2):
    """
    Define the system of ODEs for the recharge oscillator model of ENSO.
    y: array of state variables [T, S].
    t: time.
    mu: coupling coefficient.
    e_n: degree of nonlinearity.
    xi_1: random wind stress forcing.
    xi_2: random heating.
    """

    # define the variables which vary with mu
    b = b_0 * mu
    R = gamma * b - c

    # define the ODEs
    dTdt = R * y[0] + gamma * y[1] - e_n * (
                y[1] + b * y[0]) ** 3 + gamma * xi_1 + xi_2
    dSdt = -r * y[1] - alpha * b * y[0] - alpha * xi_1

    return np.array([dTdt, dSdt])

# test the schemes for different values of mu between 2/3 and 0.75
mu = np.linspace(2/3, 0.75, 4)

# set up the time step

dt = 0.02 # may have to change the dt to acheive stability
nt = int(5*41 / dt) # for 5 periods
t = np.linspace(0, 5*41, nt)

# define the non-dimensionalised initial conditions for the AB2 scheme
T_init = 1.125/7.5
h_init = 0/150

# define the non-dimensionalised initial conditions for the RK4 scheme
y0 = [1.125 / 7.5, 0 / 150]

# run the updated RK4 scheme for the range of mu
sst_anomaly_rk_nl_mu = np.zeros((len(mu), nt))
thermocline_depth_rk_nl_mu = np.zeros((len(mu), nt))
for i in range(len(mu)):
    sst_anomaly_rk_nl_mu[i], thermocline_depth_rk_nl_mu[i] = rk4(enso_RK4, y0, t, mu[i], e_n, xi_1, xi_2)

# run the updated AB2 scheme for the range of mu
sst_anomaly_ab_nl_mu = np.zeros((len(mu), nt))
thermocline_depth_ab_nl_mu = np.zeros((len(mu), nt))
for i in range(len(mu)):
    sst_anomaly_ab_nl_mu[i], thermocline_depth_ab_nl_mu[i] = adams_bashforth(
        T_init, h_init,mu[i], R, gamma, e_n, xi_1, xi_2, dt, nt)

# redimensioalise the results
sst_anomaly_ab_nl_mu = sst_anomaly_ab_nl_mu*7.5
thermocline_depth_ab_nl_mu = thermocline_depth_ab_nl_mu*150/10
sst_anomaly_rk_nl_mu = sst_anomaly_rk_nl_mu*7.5
thermocline_depth_rk_nl_mu = thermocline_depth_rk_nl_mu*150/10

# set up the subplots as a 4x2 grid of subplots with the time series plots on
# the left and the phase space plots on the right
fig, axs = plt.subplots(4, 2, figsize=(10, 20))

# plot the time series for a range of mu
for i in range(len(mu)):
    axs[i, 0].plot(t, sst_anomaly_ab_nl_mu[i], label='AB2 $T_e$')
    axs[i, 0].plot(t, thermocline_depth_ab_nl_mu[i], label='AB2 $h_w$')
    axs[i, 0].plot(t, sst_anomaly_rk_nl_mu[i], label='RK4 $T_e$',linestyle='--')
    axs[i, 0].plot(t, thermocline_depth_rk_nl_mu[i], label='RK4 $h_w$',linestyle='--')
    axs[i, 0].set_xlabel('Time')
    # set the title of the plot
    axs[i, 0].set_title('Time series for T and h ($\mu = %s$)' % mu[i])
    # specify the legend in the upper right corner
    axs[i, 0].legend(loc='upper right')

# plot the phase space results for the nonlinear case
for i in range(len(mu)):
    axs[i, 1].plot(thermocline_depth_ab_nl_mu[i], sst_anomaly_ab_nl_mu[i], label='AB2')
    axs[i, 1].plot(thermocline_depth_rk_nl_mu[i], sst_anomaly_rk_nl_mu[i], label='RK4',linestyle='--')
    # set the title of the plot
    axs[i, 1].set_title('Phase space plot for T and h ($\mu = %s$)' % mu[i])
    # specify the legend in the upper right corner
    axs[i, 1].legend(loc='upper right')

# specify a tight layout
fig.tight_layout()

# specify a sup title for the plots
fig.suptitle('Nonlinear cases for the ROM for a range of $\mu$ values')

plt.show()

# save the figure
fig.savefig('nonlinear_cases_RK4_AB2_range_of_mu.png')


# first define the global variables for the task D experiment

# for Task D we want to redefine the RK4 scheme to allow the coupling
# parameter to vary with time

def enso_taskD(y,t):
    """RK4 scheme for task D where the coupling parameter varies with time"""

    # define the global variables
    e_n = 0.1
    mu_0 = 0.75
    mu_ann = 0.2
    tau = 12 # months

    # define the coupling parameter
    mu = mu_0*(1 + mu_ann*np.cos(2*np.pi*t/tau - 5*np.pi/6))

    # define the thermocline slope
    b = b_0*mu

    # define the Bjerknes positive feedback process
    R = gamma*b - c

    # define the ODEs
    dTdt = R*y[0] + gamma*y[1] - e_n*(y[1] + b*y[0])**3 + gamma*xi_1 + xi_2
    dSdt = -r*y[1] - alpha*b*y[0] - alpha*xi_1

    return np.array([dTdt, dSdt])




# first define the global variables for the Task C experiment where we introduce non-linearity

mu_0 = 0.75 # base value for the coupling parameter
mu_ann = 0.2 # annual cycle of the coupling parameter
tau = 12 # period of the forcing in months (for mu)

# allow the coupling parameter to vary with time in an annual cycle
mu = mu_0*(1 + mu_ann*np.cos(2*np.pi*t/tau))

b_0 = 2.5 # high end value for the coupling parameter
b = b_0*mu # measure of the thermocline slope
gamma = 0.75 # feedback of the thermocline gradient on the SST gradient
c = 1 # damping rate of SST anomalies
R = gamma*b - c # describes the Bjerknes positive feedback process
r = 0.25 # damping of upper ocean heat content
alpha = 0.125 # relates enhanced easterly wind stress to the recharge of ocean heat content
# modify the degree of nonlinearity for task C
e_n = 0.1 # degree of nonlinearity of ROM
xi_1 = 0 # random wind stress forcing added to the system
xi_2 = 0 # random heating added to the system

# set up the time step for the RK4 scheme
dt = 0.02
nt = int(5*41/dt) # 5 periods of 41 months each
t = np.linspace(0, 5*41, nt)

# set up the initial conditions non-dimensionalised
y0 = [1.125 / 7.5, 0 / 150]

# initialize the updated rk4 scheme
y_taskD = rk4(enso_taskD, y0, t)
sst_anomaly_SEH, thermocline_depth_SEH = y_taskD[:,0], y_taskD[:,1]

# redimensioalise the results
sst_anomaly_SEH = sst_anomaly_SEH*7.5
thermocline_depth_SEH = thermocline_depth_SEH*150/10

# plots two subplots, one on the left (time series) and the other on the
# right (phase space plot)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# plot the time series for the SEH case
axs[0].plot(t, sst_anomaly_SEH, label='SEH $T_e$')
axs[0].plot(t, thermocline_depth_SEH, label='SEH $h_w$')
axs[0].set_xlabel('Time')
# set the title of the plot
axs[0].set_title('Time series for T and h (SEH)')
# specify the legend
axs[0].legend(loc='upper right')

# plot the phase space results for the SEH case
axs[1].plot(thermocline_depth_SEH, sst_anomaly_SEH, label='SEH')
# set the title of the plot
axs[1].set_title('Phase space plot for T and h (SEH)')

# specify a tight layout
fig.tight_layout()

# specify a sup title for the plots
fig.suptitle('SEH case for the ROM')

# save the figure
fig.savefig('SEH_case_RK4.png')

# define W, an array of random numbers for the forcing with normal distribution
W = np.random.normal(-1, 1, nt)


