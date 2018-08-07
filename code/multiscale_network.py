import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import create_flight_connections
import create_graphs
import model.network_model


def plot_epidemic_map(g):
    # TODO add airport locations and connections between them
    beta, mu = .08, 0.5
    R0 = beta / mu
    print(R0)
    t = np.linspace(0, 5, 1001)  # time grid
    nodes = {str(n): n for i, n in enumerate(g.nodes())}

    # results = model.network_model.sir_ode_on_network(g, nodes, starting_node='1465011', I0=1, beta=beta, mu=mu, t=t)
    # results = np.array()
    # for test_node in ['2403011', '2475011', '3262011']:
    #     s, i, r = results['S'][nodes[test_node]], results['I'][nodes[test_node]], results['R'][nodes[test_node]]
    #     model.network_model.plot_change_in_population('test4_' + str(test_node), t, s, i, r)
    #
    # gminas_df.loc[:, 'sir_t=']
    gminas_df.plot(column='population', alpha=0.3, figsize=(10, 10))

    plt.savefig('../data/graphs/epidemic_spread.png')
    plt.show()


if __name__ == '__main__':
    gminas, gminas_pos, gminas_df = create_graphs.gminas_network()
    flights, flights_pos = create_flight_connections.flights_network()

    G = nx.compose(gminas, flights)

    plot_epidemic_map(G)
