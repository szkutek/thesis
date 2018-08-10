import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

"""
Contains necessary functions to simulate spreading of an epidemic on a the multiscale network,
where data is organised in the following way:
1. networkx Graph (teryt number, population, work commute)
2. geopandas GeoDataFrame (teryt number, geometry, population, name, representative pt_x and pt_y)
3. dictionary results with 2D arrays for each SIR (node_id x t_i) 

The algorithm uses (1.) to fill (3.) with results. 
Then we can plot the results using GeoDataFrame with added column for the time t_i that we want to display
(defined in multiscale_network.py).  
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


def sir_model_on_node(g, results, index, node, i, dt, beta, mu):
    """index: index of node in list(g.nodes)"""
    nbrs = list(nx.neighbors(g, node))
    infection_from_nbrs = 0.

    for nbr in nbrs:
        nbr_index = list(g.nodes).index(nbr)
        infection_from_nbrs += g.get_edge_data(node, nbr)['commute'] * results['I'][nbr_index, i - 1] \
                               / g.nodes[nbr]['population']

    y = np.array([results['S'][index, i - 1], results['I'][index, i - 1], results['R'][index, i - 1]])

    # if infection_from_nbrs != 0:
    #     print(infection_from_nbrs)
    #     print(y)

    def f(u):
        Si, Ii, Ri = u
        dS_dt = - beta * Si * Ii - beta * Si * infection_from_nbrs
        dI_dt = beta * Si * Ii + beta * Si * infection_from_nbrs - mu * Ii
        dR_dt = mu * Ii
        return np.array([dS_dt, dI_dt, dR_dt])

    # # Euler method
    # y += f(y) * dt

    # # Runge-Kutta method of 4th order TEŻ GÓWNO
    k1 = f(y)
    k2 = f(y + dt * k1 / 2.)
    k3 = f(y + dt * k2 / 2.)
    k4 = f(y + dt * k3)
    y += dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)

    # if infection_from_nbrs != 0:
    #     print(y)
    results['S'][index, i], results['I'][index, i], results['R'][index, i] = y
    return


def sir_ode_on_network(g, starting_node, I0, t, beta, mu):
    # initialize SIR
    number_of_nodes = len(g.nodes())
    results = {'S': np.zeros([number_of_nodes, len(t)]),
               'I': np.zeros([number_of_nodes, len(t)]),
               'R': np.zeros([number_of_nodes, len(t)])}

    for k, node in enumerate(g.nodes):
        results['S'][k, 0] = g.nodes[node]['population']  # g.nodes('population')[node]

    starting_node_index = list(g.nodes).index(starting_node)
    results['S'][starting_node_index, 0] -= I0
    results['I'][starting_node_index, 0] = I0

    dt = t[1] - t[0]

    for i, _ in enumerate(t[1:]):
        print('i = ' + str(i))
        for k, node in enumerate(g.nodes):
            sir_model_on_node(g, results, k, node, i + 1, dt, beta, mu)

    return results
