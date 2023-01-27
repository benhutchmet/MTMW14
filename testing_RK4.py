# file for testing the RK4 scheme in isolation

# define the global variables

# non-dimensionalised parameters

T_nd = 7.5 # SST anomaly-scale (kelvin)
h_nd = 150 # thermocline depth scale (m)
t_nd = 2*30*24*60*60 # time-scale - 2 months in seconds

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
T_e_init = 1.125
h_w_init = 0

# specify the timestep
dt = 10

# specify the number of time steps
nt = 1000

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

    k2 = f1(T_e + k1/2, h_w + l1/2)
    l2 = f2(T_e + k1/2, h_w + l1/2)

    k3 = f1(T_e + k2/2, h_w + l2/2)
    l3 = f2(T_e + k2/2, h_w + l2/2)

    k4 = f1(T_e + k3, h_w + l3)
    l4 = f2(T_e + k3, h_w + l3)

    T_e_new = T_e + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    h_w_new = h_w + dt/6 * (l1 + 2*l2 + 2*l3 + l4)

    return T_e_new, h_w_new