import matplotlib.pyplot as plt
import networkx as nx

import create_flight_connections
import create_graphs


def plot_network(G1, pos1, G2, pos2, node_labels=False, edge_labels=False):
    nx.draw_networkx_nodes(G1, pos=pos1, node_size=100, node_color='red', edge_color='k', alpha=.5,
                           with_labels=False)
    nx.draw_networkx_edges(G1, pos=pos1, edge_color='red', alpha=.3)

    nx.draw_networkx_nodes(G2, pos=pos2, node_size=100, node_color='blue', edge_color='k', alpha=.5,
                           with_labels=False)
    nx.draw_networkx_edges(G2, pos=pos2, edge_color='blue', alpha=.3)
    if node_labels:
        nx.draw_networkx_labels(G1, pos=pos1, label_pos=0.5)
        nx.draw_networkx_labels(G2, pos=pos2, label_pos=0.5)
    if edge_labels:
        nx.draw_networkx_edge_labels(G1, pos=pos1, label_pos=0.5)
        nx.draw_networkx_edge_labels(G2, pos=pos2, label_pos=0.5)

    plt.savefig('../data/graphs/multiscale_network.png')
    plt.show()


if __name__ == '__main__':
    gminas, gminas_pos = create_graphs.gminas_network()
    flights, flights_pos = create_flight_connections.flights_network()

    # TODO connect airports to the cities
    plot_network( flights, flights_pos,gminas, gminas_pos, False, False)
