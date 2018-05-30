import random as rnd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def sir_ode_without_neighbours():
    def integrate(f, y, t, args):
        res = np.ndarray([len(t), len(y)])
        tau = t[1] - t[0]
        res[0, :] = y
        for i, _ in enumerate(t[1:]):
            res[i + 1, :] = int_iterative(f, res[i, :], tau, args)
        return res

    def int_iterative(f, y, tau, args):
        dy = f(y, t, args)
        dy = np.array(dy) * tau
        return y + dy

    def SIR_model_ODE(y, t, args):  # SIR model
        beta, mu, infected_from_nbrs = args
        Si = y[0]  # susceptible
        Ii = y[1]  # infected
        Ri = y[2]  # recovered
        N = sum(y)
        dS_dt = - beta * Si * Ii / N
        dI_dt = beta * Si * Ii / N - mu * Ii
        dR_dt = mu * Ii
        return dS_dt, dI_dt, dR_dt


def plot_change_in_population(filename, t, S, I, R=None):
    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    if R is not None:
        plt.plot(t, R, label='Recovered')
        title = 'Plague model for S(0)=' + str(S[0]) + ', I(0)=' + str(I[0]) + ' and R(0)=' + str(R[0])
    else:
        title = 'Plague model for S(0)=' + str(S[0]) + ', I(0)=' + str(I[0])

    plt.xlabel('t')
    plt.ylabel('population')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig(filename + '.png')


def sir_model_on_node(g, node, i, tau, beta, mu):
    N = g.nodes[node]['population']
    nbrs = [*nx.neighbors(g, node)]
    infection_from_nbrs = 0.
    for nbr in nbrs:
        infection_from_nbrs += g.get_edge_data(node, nbr)['commute'] * g.nodes[nbr]['I'][i - 1] \
                               / g.nodes[nbr]['population']

    y = np.array([g.nodes[node]['S'][i - 1], g.nodes[node]['I'][i - 1], g.nodes[node]['R'][i - 1]])

    def f(u):
        Si, Ii, Ri = u
        dS_dt = - beta * Si * Ii / N - beta * Si / N * infection_from_nbrs
        dI_dt = beta * Si * Ii / N + beta * Si / N * infection_from_nbrs - mu * Ii
        dR_dt = mu * Ii
        return np.array([dS_dt, dI_dt, dR_dt])

    # TODO change to Runge-Kutta method
    # # Euler method
    y += f(y) * tau

    g.nodes[node]['S'][i], g.nodes[node]['I'][i], g.nodes[node]['R'][i] = y
    return


def sir_ode_on_network(g, starting_node, I0, t, beta, mu):
    # initialize SIR
    g.nodes[starting_node]['S'][0] = g.nodes[starting_node]['S'][0] - I0
    g.nodes[starting_node]['I'][0] = I0
    tau = t[1] - t[0]

    for i, _ in enumerate(t[1:]):
        for node in g.nodes:
            sir_model_on_node(g, node, i + 1, tau, beta, mu)


def create_graph(t):
    time = len(t)
    number_of_nodes = 20

    g = nx.barabasi_albert_graph(number_of_nodes, 2)
    population = {node: rnd.randint(100, 1000) for node in g.nodes()}
    area = {node: rnd.randint(1000, 1500) for node in g.nodes()}
    density = {k: population[k] / area[k] for k in population}
    nx.set_node_attributes(g, population, 'population')
    nx.set_node_attributes(g, area, 'area')
    nx.set_node_attributes(g, density, 'density')

    # starting_node = 1
    nx.set_node_attributes(g, {node: np.zeros(time) for node in g.nodes()}, 'S')
    for node in g.nodes:
        g.nodes[node]['S'][0] = g.nodes[node]['population']
    nx.set_node_attributes(g, {node: np.zeros(time) for node in g.nodes()}, 'I')
    nx.set_node_attributes(g, {node: np.zeros(time) for node in g.nodes()}, 'R')

    # g = g.to_directed() # do we need this???
    commute = {edge: rnd.randint(50, 100) for edge in g.edges()}
    nx.set_edge_attributes(g, commute, 'commute')
    return g


if __name__ == "__main__":
    node = 1
    beta, mu = .8, 0.09
    t = np.linspace(0, 100., 1001)  # time grid

    g = create_graph(t)

    sir_ode_on_network(g, node, 1, t, beta, mu)

    for test_node in [1, 2, 3]:
        s, i, r = g.nodes[test_node]['S'], g.nodes[test_node]['I'], g.nodes[test_node]['R']
        plot_change_in_population('test2_' + str(test_node), t, s, i, r)
