import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# solve the system dy/dt = f(y, t)
def f1(y, t, B, r):  # SIR model
    Si = y[0]  # susceptible
    Ii = y[1]  # infected
    Ri = y[2]  # recovered
    # the equations
    f0 = -B * Si * Ii
    f1 = B * Si * Ii - r * Ii
    f2 = r * Ii
    return [f0, f1, f2]


def f2(y, t, B, r):  # SI model
    Si = y[0]  # susceptible
    Ii = y[1]  # infected

    f0 = -B * Si * Ii
    f1 = B * Si * Ii - r * Ii
    return [f0, f1]


def plot_change_in_population(t, S0, S, I0, I, R0=None, R=None):
    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    if R is not None:
        plt.plot(t, R, label='Recovered')
    if R0 is None:
        title = 'Plague model for S(0)=' + str(S0) + ', I(0)=' + str(I0)
    else:
        title = 'Plague model for S(0)=' + str(S0) + ', I(0)=' + str(I0) + ' and R(0)=' + str(R0)

    plt.xlabel('time')
    plt.ylabel('population')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()


# def model_SIR(N=100, Bs=list(0.03), rs=list(1.), I0=1):
def model_SIR(N, Bs, rs, I0):
    for B, r in list(zip(Bs, rs)):
        # B = 0.03  # infectivity
        # r = 1  # recovery rate
        R0 = B * N / r  # basic reproductive ratio
        print(R0)
        # # disease free state
        # S0 = N, I0 = 0, R0 = 0

        # initial conditions
        I0 = 1
        S0 = N - I0  # initial population
        t = np.linspace(0, 10., 1000)  # time grid

        # # solve the DEs
        y0 = [S0, I0, R0]  # initial condition vector
        soln = odeint(f1, y0, t, args=(B, r))
        S = soln[:, 0]
        I = soln[:, 1]
        R = soln[:, 2]
        plot_change_in_population(t, S0, S, I0, I, R0, R)


# def model_SI(N=100, Bs=list(0.03), rs=list(1.), I0=1):
def model_SI(N, Bs, rs, I0):
    for B, r in list(zip(Bs, rs)):
        # B = 0.03  # infectivity
        # r = 1  # recovery rate
        R0 = B * N / r  # basic reproductive ratio
        print(R0)

        # initial conditions
        S0 = N - I0  # initial population
        t = np.linspace(0, 10., 1000)  # time grid

        # # solve the DEs
        y0 = [S0, I0]  # initial condition vector
        soln2 = odeint(f2, y0, t, args=(B, r))
        S = soln2[:, 0]
        I = soln2[:, 1]
        plot_change_in_population(t, S0, S, I0, I, R0)


def phase_portrait(B, r, a, b, c, d):
    # Plotting direction fields and trajectories in the phase plane

    y1 = np.linspace(a, b, 20)
    y2 = np.linspace(c, d, 20)
    Y1, Y2 = np.meshgrid(y1, y2)
    t = 0
    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
    NI, NJ = Y1.shape
    for i in range(NI):
        for j in range(NJ):
            x = Y1[i, j]
            y = Y2[i, j]
            yprime = f2([x, y], t, B, r)
            u[i, j] = yprime[0]
            v[i, j] = yprime[1]
    plt.quiver(Y1, Y2, u, v, color='r')
    plt.title("Phase portait for Beta=" + str(B) + " and r=" + str(r))
    plt.xlabel('')
    plt.ylabel('')
    plt.show()


def total_infected():
    N = 100
    Bs = np.linspace(0.03, 2., 100)  # Beta
    rs = np.linspace(1., 10., 100)  # r
    R0 = []
    total = []
    for B, r in list(zip(Bs, rs)):
        R0.append(B * N / r)  # basic reproductive ratio
        # initial conditions
        I0 = 1
        S0 = N - I0  # initial population

        # # solve the DEs
        y0 = [S0, I0, R0[-1]]  # initial condition vector
        t = np.linspace(0, 10., 1000)  # time grid
        soln = odeint(f1, y0, t, args=(B, r))
        S = soln[:, 0]
        I = soln[:, 1]
        R = soln[:, 2]
        total.append(I[-1] - I0 + R[-1] - R0[-1])

    plt.plot(R0, total, '*')
    plt.title("Total number of infections as a function of R0")
    plt.show()


if __name__ == "__main__":
    N = 100
    Bs = [0.03, 0.05, 0.9, 1.]  # Beta
    rs = [10., 1., 2., 5.]  # r
    I0 = 1

    model_SIR(N, Bs, rs, I0)
    model_SI(N, Bs, rs, I0)

    phase_portrait(Bs[0], rs[0], -20, 20, -20, 20)
    phase_portrait(Bs[-1], rs[-1], -20, 20, -20, 20)
    total_infected()
