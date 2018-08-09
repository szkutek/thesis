import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import create_flight_connections
import create_graphs
import network_sir_model


def plot_epidemic_map(g):
    beta, mu = .008, 0.5
    R0 = beta / mu
    # print(R0)
    t = np.linspace(0, 5, 1001)  # time grid
    nodes = {str(n): i for i, n in enumerate(g.nodes())}
    starting_node = '1465011'
    # results = network_sir_model.sir_ode_on_network(g, nodes,
    #                                                starting_node=starting_node, I0=1, beta=beta, mu=mu, t=t)
    results = {'S': np.random.rand(len(nodes), len(t)),
               'I': np.random.rand(len(nodes), len(t)),
               'R': np.random.rand(len(nodes), len(t))}

    for test_node in ['2403011', '2475011', '3262011']:
        s, i, r = results['S'][nodes[test_node]], results['I'][nodes[test_node]], results['R'][nodes[test_node]]
        network_sir_model.plot_change_in_population('test4_' + str(test_node), t, s, i, r)

    # for i in [0, 200, 500]:
    i = 0
    col = 'epidemic_ti=' + str(t[i])
    gminas_df.loc[:, col] = results['I'][:, i]  # TODO? .map(dict())
    map = gminas_df.plot(column=col, alpha=0.8, cmap='RdYlGn_r', figsize=(10, 7), legend=True)
    plt.xticks([])
    plt.yticks([])

    # plot start of the epidemic with black border
    gminas_df.loc[[starting_node], ['geometry', col]].plot(column=col, alpha=0.8, cmap='RdYlGn_r',
                                                           ax=map, edgecolor='black', linewidth=2)
    # plot airline connections
    nx.draw_networkx_edges(flights, flights_pos, edge_color='blue', width=0.5, alpha=.8, ax=map)

    plt.savefig('../data/graphs/epidemic_spread.eps', format='eps')
    # plt.show()


if __name__ == '__main__':
    # TODO uncomment add_work_migration in create_graphs.gminas_network()
    gminas, gminas_pos, gminas_df = create_graphs.gminas_network(create=True)
    flights, flights_pos = create_flight_connections.flights_network(create=True)

    G = nx.compose(gminas, flights)

    plot_epidemic_map(G)
