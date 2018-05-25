import random as rnd
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pylab
from scipy.integrate import odeint


# solve the system dy/dt = f(y, t)
def f1(y, t, beta, mu):  # SIR model
    Si = y[0]  # susceptible
    Ii = y[1]  # infected
    Ri = y[2]  # recovered
    # TODO modify to work on graph
    f0 = -beta * Si * Ii
    f1 = beta * Si * Ii - mu * Ii
    f2 = mu * Ii
    return [f0, f1, f2]


def f2(y, t, beta, mu):  # SIS model
    Si = y[0]  # susceptible
    Ii = y[1]  # infected

    f0 = -beta * Si * Ii
    f1 = beta * Si * Ii - mu * Ii
    return [f0, f1]


def plot_change_in_population(t, S, I, R=None):
    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    if R is not None:
        plt.plot(t, R, label='Recovered')
        title = 'Plague model for S(0)=' + str(S[0]) + ', I(0)=' + str(I[0]) + ' and R(0)=' + str(R[0])
    else:
        title = 'Plague model for S(0)=' + str(S[0]) + ', I(0)=' + str(I[0])

    plt.xlabel('time')
    plt.ylabel('population')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()


# def model_SIR(N=100, Bs=list(0.03), rs=list(1.), I0=1):
def model_SIR(N, beta, mu, I0):
    # beta = 0.03  # infectivity
    # mu = 1  # recovery rate
    R0 = beta * N / mu  # basic reproductive ratio
    print(R0)
    # # disease free state S0 = N, I0 = 0, R0 = 0

    # initial conditions
    I0 = 1
    S0 = N - I0  # initial population
    t = np.linspace(0, 10., 1000)  # time grid

    # # solve the DEs
    y0 = [S0, I0, R0]  # initial condition vector
    soln = odeint(f1, y0, t, args=(beta, mu))
    # S = [S0] + soln[:, 0]
    # I = [I0] + soln[:, 1]
    # R = [R0] + soln[:, 2]
    S = soln[:, 0]
    I = soln[:, 1]
    R = soln[:, 2]
    # plot_change_in_population(t, S, I, R)
    return S, I, R


# def model_SI(N=100, Bs=list(0.03), rs=list(1.), I0=1):
def model_SI(N, beta, mu, I0):
    # beta = 0.03  # infectivity
    # mu = 1  # recovery rate

    # initial conditions
    S0 = N - I0  # initial population
    t = np.linspace(0, 10., 1000)  # time grid

    # # solve the DEs
    y0 = [S0, I0]  # initial condition vector
    soln2 = odeint(f2, y0, t, args=(beta, mu))
    S = soln2[:, 0]
    I = soln2[:, 1]
    # plot_change_in_population(t, S, I)
    return S, I


def SIR_graph(G, prob=0.5, save=False, pathname='graph_walk', starting=None):
    Susceptible = list(G.nodes())
    if starting is None:
        start = Susceptible[rnd.randint(0, len(Susceptible) - 1)]
    else:
        # start = starting
        start = Susceptible[0]

    Infected_now = [start]
    Susceptible.remove(start)

    Infected_next = []
    Removed = []

    SIR = {'S': [Susceptible], 'I': [Infected_now], 'R': [Removed]}

    if save:
        fig = pylab.figure(figsize=(15, 10))
        pos = nx.spring_layout(G)
        size = 300

        nx.draw_networkx_nodes(G, pos, nodelist=Susceptible, node_color='g', node_size=size, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=Infected_now, node_color='y', node_size=size, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

        nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'population'))
        # nx.draw_networkx_edge_labels(G, pos, label_pos=.75)

        pylab.savefig(pathname + '_0.png')
        pylab.clf()

    i = 1
    while Infected_now:
        for u in Infected_now:
            for v in G.neighbors(u):
                if rnd.random() < prob and v in Susceptible:
                    Infected_next.append(v)
                    Susceptible.remove(v)

            Removed.append(u)

        Infected_now = Infected_next
        Infected_next = []

        if save:
            nx.draw_networkx_nodes(G, pos, nodelist=Susceptible, node_color='g', node_size=size, alpha=0.8)
            nx.draw_networkx_nodes(G, pos, nodelist=Infected_now, node_color='y', node_size=size, alpha=0.8)
            nx.draw_networkx_nodes(G, pos, nodelist=Removed, node_color='grey', node_size=size, alpha=0.8)

            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'population'))
            # nx.draw_networkx_edge_labels(G, pos, label_pos=.75)

            s = pathname + "_" + str(i) + ".png"
            pylab.savefig(s)
            pylab.clf()

        SIR['S'].append(Susceptible)
        SIR['I'].append(Infected_now)
        SIR['R'].append(Removed)

        i += 1

    if save:
        pylab.close(fig)
    return SIR


def movie(n, pathname, moviename, duration=0.2):
    frames = []
    for i in range(n):
        path = pathname + "_" + str(i) + '.png'
        frames.append(imageio.imread(path))

    kargs = {'duration': duration}
    imageio.mimwrite(moviename + '.gif', frames, 'gif', **kargs)


def sim_infection(G, p, M, start):
    N = G.number_of_nodes()
    res = []
    for i in range(M):
        SIR = SIR_graph(G, prob=p, starting=start)
        w = []
        for vec in SIR['I']:
            # vec is a list of lists, each for different time
            # so w contains fraction of infected nodes at times t
            w.append(len(vec) / N)

        for t in range(len(w)):
            if len(res) <= t:
                res.append(w[t] / M)
            else:
                res[t] += w[t] / M
    return res


def create_graph():
    number_of_nodes = 20

    g = nx.barabasi_albert_graph(number_of_nodes, 2)
    population = {node: rnd.randint(100, 1000) for node in g.nodes()}
    area = {node: rnd.randint(1000, 1500) for node in g.nodes()}
    density = {k: population[k] / area[k] for k in population}
    nx.set_node_attributes(g, population, 'population')
    nx.set_node_attributes(g, area, 'area')
    nx.set_node_attributes(g, density, 'density')

    # initialize SIR
    starting_node = 1
    nx.set_node_attributes(g, {node: [p, p - int(node == starting_node)] for node, p in population.items()}, 'S')
    nx.set_node_attributes(g, {node: [0, int(node == starting_node)] for node in g.nodes()}, 'I')
    nx.set_node_attributes(g, {node: [0, 0] for node in g.nodes()}, 'R')

    s, i, r = lambda node, id: g.nodes[node]['S'][id], lambda node, id: g.nodes[node]['I'][id], \
              lambda node, id: g.nodes[node]['R'][id]
    n = lambda node, id: s(node, id) + i(node, id) + r(node, id)
    nx.set_node_attributes(g, {node: [n(node, 0), n(node, 1)] for node in g.nodes()}, 'N')

    # g = g.to_directed() # do we need this???
    commute = {edge: rnd.randint(50, 100) for edge in g.edges()}
    nx.set_edge_attributes(g, commute, 'com')
    return g


def SIR_on_network(G, starting=None):
    Susceptible = list(G.nodes())
    prob = 0.5
    if starting is None:
        start = Susceptible[rnd.randint(0, len(Susceptible) - 1)]
    else:
        # start = starting
        start = Susceptible[0]

    return


def sis_on_network(g):
    P = 0.8
    mv = True  # SAVE AND MAKE MOVIE
    pathname = 'fig'
    SIR = SIR_graph(g, prob=P, save=mv, pathname=pathname)
    movie(len(SIR['I']), pathname, 'sir_ba', 1.)


if __name__ == "__main__":
    g = create_graph()
    sis_on_network(g)
