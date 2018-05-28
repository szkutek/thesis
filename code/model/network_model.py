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
    dS_dt = -beta * Si * Ii
    dI_dt = beta * Si * Ii - mu * Ii
    dR_dt = mu * Ii
    return [dS_dt, dI_dt, dR_dt]


def f2(y, t, beta, mu):  # SIS model
    Si = y[0]  # susceptible
    Ii = y[1]  # infected

    dS_dt = -beta * Si * Ii
    dI_dt = beta * Si * Ii
    return [dS_dt, dI_dt]


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


def sir_on_node(g, node, t, beta, mu, i0=0):
    # beta = 0.03  # infectivity
    # mu = 1.  # recovery rate
    N = g.nodes[node]['population']
    R0 = beta * N / mu  # basic reproductive ratio
    print(R0)
    # # disease free state s0 = N, i0 = 0, R0 = 0

    # initial conditions
    s0 = N - i0  # initial number of susceptible individuals

    # # solve the DEs
    y0 = [s0, i0, 0]  # initial condition vector
    soln = odeint(f1, y0, t, args=(beta, mu))
    # S = [s0] + soln[:, 0]
    # I = [i0] + soln[:, 1]
    # R = [R0] + soln[:, 2]
    s = soln[:, 0]
    i = soln[:, 1]
    r = soln[:, 2]

    # nx.set_node_attributes(g, {node: s}, 'S')
    # nx.set_node_attributes(g, {node: i}, 'I')
    # nx.set_node_attributes(g, {node: r}, 'R')
    return s, i, r


def model_SI(N, t, beta, mu, I0):
    # beta = 0.03  # infectivity
    # mu = 1.  # recovery rate

    # initial conditions
    S0 = N - I0  # initial population

    # # solve the DEs
    y0 = [S0, I0]  # initial condition vector
    soln2 = odeint(f2, y0, t, args=(beta, mu))
    S = soln2[:, 0]
    I = soln2[:, 1]
    # plot_change_in_population(t, S, I)
    return S, I


def agent_sir_on_network():
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

    def sir_on_network(g):
        P = 0.8
        mv = True  # SAVE AND MAKE MOVIE
        pathname = 'fig'
        SIR = SIR_graph(g, prob=P, save=mv, pathname=pathname)
        movie(len(SIR['I']), pathname, 'sir_ba', 1.)

    g = create_graph()
    sir_on_network(g)


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
    # starting_node = 1
    # nx.set_node_attributes(g, {node: p for node, p in population.items()}, 'N')
    # nx.set_node_attributes(g, {node: p for node, p in population.items()}, 'S')
    # nx.set_node_attributes(g, {node: 0 for node in g.nodes()}, 'I')
    # nx.set_node_attributes(g, {node: 0 for node in g.nodes()}, 'R')

    # g = g.to_directed() # do we need this???
    commute = {edge: rnd.randint(50, 100) for edge in g.edges()}
    nx.set_edge_attributes(g, commute, 'commute')
    return g


if __name__ == "__main__":
    g = create_graph()
    # agent_sir_on_network()
    node = 1

    t = np.linspace(0, 10., 100)  # time grid
    beta, mu = 0.03, 1.
    s, i, r = sir_on_node(g, node, t, beta, mu, i0=1)
    plot_change_in_population(t, s, i, r)

    # tmp = g.nodes[node]
    # print(tmp)
    # nbrs = [*nx.neighbors(g, node)]
    # print(nbrs)
    # for nbr in nbrs:
    #     print(nbr, g.get_edge_data(node, nbr)['commute'])
