import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import create_flight_connections
import create_graphs
import network_sir_model


def save_results_to_file(g, beta, mu, t, starting_node, filename='results.npy'):
    results = network_sir_model.sir_ode_on_network(g, starting_node=starting_node, I0=1, beta=beta, mu=mu, t=t)
    results['beta'] = beta
    results['mu'] = mu
    results['t'] = t
    results['starting_node'] = starting_node
    results['node'] = list(g.nodes)

    np.save('../data/results/' + filename, results)  # save results to file
    return


def plot_epidemic_map(G, filename):
    results = np.load('../data/results/' + filename)[()]  # type: dict()
    print(results.keys())

    # beta = results['beta']
    # mu = results['mu']
    # R0 = beta / mu
    # # print(R0)
    # t = results['t']  # time grid
    # starting_node = results['starting_node']
    # nodes = results['nodes']
    nodes = list(G.nodes)  # TODO check if it's the same as results['nodes']

    for test_node in [starting_node]:  # , '2403011', '2475011', '3262011']:
        test_node_index = nodes.index(test_node)
        print(test_node_index)
        s, i, r = results['S'][test_node_index], results['I'][test_node_index], results['R'][test_node_index]
        network_sir_model.plot_change_in_population('test4_' + test_node, t, s, i, r)

    # for i in [0, 200, 500]:
    i = 0
    col = 'epidemic'
    I = results['I'][:, i]
    S = results['S'][:, i]
    gminas_df.loc[:, col] = gminas_df.loc[:, 'teryt'].map(dict(zip(nodes, I / S)))
    print(gminas_df.head())
    map = gminas_df.plot(column=col, alpha=0.8, cmap='RdYlGn_r', figsize=(10, 7), legend=True)  # , scheme='quantiles')
    plt.xticks([])
    plt.yticks([])

    # plot start of the epidemic with black border
    gminas_df.loc[[starting_node], ['geometry', col]].plot(column=col, alpha=0.8, cmap='RdYlGn_r',
                                                           ax=map, edgecolor='black', linewidth=2)
    # plot airline connections
    nx.draw_networkx_edges(flights, flights_pos, edge_color='blue', width=0.5, alpha=.8, ax=map)

    plt.savefig('../data/graphs/epidemic_spread.eps', format='eps')
    plt.show()


if __name__ == '__main__':
    gminas, gminas_pos, gminas_df = create_graphs.gminas_network(create=False)
    flights, flights_pos = create_flight_connections.flights_network(create=False)

    G = nx.compose(gminas, flights)

    beta, mu = .008, 0.5
    R0 = beta / mu
    # print(R0)
    t = np.linspace(0, 1, 10001)  # time grid
    starting_node = '1465011'

    filename = 'results_test.npy'
    save_results_to_file(G, beta, mu, t, starting_node, filename)

    # plot_epidemic_map(G, filename)
