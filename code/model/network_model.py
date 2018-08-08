import random as rnd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

"""
Contains necessary functions to simulate spreading of an epidemic on a random graph 
and present results for chosen nodes in simple plots.   
"""


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


def sir_model_on_node(g, nodes, results, node, i, dt, beta, mu):
    N = g.nodes[node]['population']
    nbrs = [*nx.neighbors(g, node)]
    infection_from_nbrs = 0.
    for nbr in nbrs:
        # infection_from_nbrs += g.get_edge_data(node, nbr)['commute'] * results['I'][nodes[nbr]][i - 1] \
        #                        / g.nodes[nbr]['population']
        infection_from_nbrs += g.get_edge_data(node, nbr)['commute'] * results['I'][nbr, i - 1] \
                               / g.nodes[nbr]['population']

    # y = np.array([results['S'][nodes[node]][i - 1], results['I'][nodes[node]][i - 1], results['R'][nodes[node]][i - 1]])
    y = np.array([results['S'][node, i - 1], results['I'][node, i - 1], results['R'][node, i - 1]])

    def f(u):
        Si, Ii, Ri = u
        dS_dt = - beta * Si * Ii - beta * Si * infection_from_nbrs
        dI_dt = beta * Si * Ii + beta * Si * infection_from_nbrs - mu * Ii
        dR_dt = mu * Ii
        return np.array([dS_dt, dI_dt, dR_dt])

    # # Euler method
    # y += f(y) * dt

    # # Runge-Kutta method of 4th order
    k1 = f(y)
    k2 = f(y + dt * k1 / 2.)
    k3 = f(y + dt * k2 / 2.)
    k4 = f(y + dt * k3)
    y += dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)

    # results['S'][nodes[node]][i], results['I'][nodes[node]][i], results['R'][nodes[node]][i] = y
    results['S'][node, i], results['I'][node, i], results['R'][node, i] = y
    return


def sir_ode_on_network(g, nodes, starting_node, I0, t, beta, mu):
    # initialize SIR
    number_of_nodes = len(nodes)
    results = {'S': np.zeros([number_of_nodes, len(t)]),
               'I': np.zeros([number_of_nodes, len(t)]),
               'R': np.zeros([number_of_nodes, len(t)])}
    for k, v in nodes.items():
        results['S'][v, 0] = g.nodes[v]['population']

    results['S'][nodes[starting_node], 0] -= I0
    results['I'][nodes[starting_node], 0] = I0
    dt = t[1] - t[0]

    for i, _ in enumerate(t[1:]):
        for node in g.nodes:
            # TODO simultaneous calculations
            sir_model_on_node(g, nodes, results, node, i + 1, dt, beta, mu)
    return results


def create_graph(t, number_of_nodes):
    time = len(t)

    g = nx.barabasi_albert_graph(number_of_nodes, 2)
    population = {node: rnd.randint(100, 1000) for node in g.nodes()}
    nx.set_node_attributes(g, population, 'population')

    commute = {edge: rnd.randint(10, 50) for edge in g.edges()}
    nx.set_edge_attributes(g, commute, 'commute')

    return g


if __name__ == "__main__":
    number_of_nodes = 10

    node = '1'
    beta, mu = .008, 0.5
    R0 = beta / mu
    print(R0)
    print(R0 * number_of_nodes)
    t = np.linspace(0, 5, 1001)  # time grid

    g = create_graph(t, number_of_nodes)
    nodes = {str(n): n for i, n in enumerate(g.nodes())}
    print(nodes)

    res = sir_ode_on_network(g, nodes, node, 1, t, beta, mu)

    for test_node in [1, 2, 3]:
        # s, i, r = results['S'][nodes[test_node]], results['I'][nodes[test_node]], results['R'][nodes[test_node]]
        s, i, r = res['S'][test_node], res['I'][test_node], res['R'][test_node]
        plot_change_in_population('test3_' + str(test_node), t, s, i, r)
