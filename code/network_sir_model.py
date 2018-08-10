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

errs = 0


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
    global errs
    nbrs = [*nx.neighbors(g, node)]
    infection_from_nbrs = 0.

    edge_data_dict = {(n1, n2): w for n1, n2, w in g.edges.data('commute')}
    for nbr in nbrs:
        if (node, nbr) in edge_data_dict:
            infection_from_nbrs += edge_data_dict[(node, nbr)] * results['I'][nodes[nbr], i - 1] / g.nodes[nbr][
                'population']
        else:
            errs += 1

    y = np.array([results['S'][nodes[node]][i - 1], results['I'][nodes[node]][i - 1], results['R'][nodes[node]][i - 1]])

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

    results['S'][nodes[node], i], results['I'][nodes[node], i], results['R'][nodes[node], i] = y
    return


def sir_ode_on_network(g, nodes, starting_node, I0, t, beta, mu):
    # initialize SIR
    number_of_nodes = len(nodes)
    results = {'S': np.zeros([number_of_nodes, len(t)]),
               'I': np.zeros([number_of_nodes, len(t)]),
               'R': np.zeros([number_of_nodes, len(t)])}
    population_dict = dict(g.nodes.data('population'))

    for k, v in nodes.items():
        results['S'][v, 0] = population_dict[k]

    results['S'][nodes[starting_node], 0] -= I0
    results['I'][nodes[starting_node], 0] = I0
    dt = t[1] - t[0]

    for i, _ in enumerate(t[1:]):
        print('i = ' + str(i))
        for node in g.nodes:
            # TODO simultaneous calculations
            sir_model_on_node(g, nodes, results, node, i + 1, dt, beta, mu)
    return results
