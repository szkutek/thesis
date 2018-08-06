import matplotlib.pyplot as plt
import networkx as nx

import create_flight_connections
import create_graphs


def plot_two_networks(g, G1, pos1, G2, pos2, node_labels=False, edge_labels=False):
    """
    :param g: G1 and G2 combined
    :param G1: graph for gminas
    :param pos1: positions for gminas
    :param G2: for airports
    :param pos2: for airports
    """
    node_size = 50
    create_graphs.plot_bg()

    # nx.draw_networkx_nodes(G1, pos=pos1, node_size=node_size, node_color='red', edge_color='k', alpha=.5,
    #                        with_labels=False)
    nx.draw_networkx_edges(G1, pos=pos1, edge_color='red', alpha=.3)

    # nx.draw_networkx_nodes(G2, pos=pos2, node_size=node_size, node_color='blue', edge_color='k', alpha=.5,
    #                        with_labels=False)
    nx.draw_networkx_edges(G2, pos=pos2, edge_color='blue', alpha=.3)
    if node_labels:
        nx.draw_networkx_labels(G1, pos=pos1, label_pos=0.5)
        nx.draw_networkx_labels(G2, pos=pos2, label_pos=0.5)
    if edge_labels:
        nx.draw_networkx_edge_labels(G1, pos=pos1, label_pos=0.5)
        nx.draw_networkx_edge_labels(G2, pos=pos2, label_pos=0.5)

    plt.savefig('../data/graphs/two_networks.png')
    plt.show()


def plot_network(g, G1, pos1, G2, pos2, node_labels=False, edge_labels=False):
    """
    :param g: graph for gminas
    :param pos1: positions for gminas
    :param pos2: positions for airports
    """
    node_size = 50
    create_graphs.plot_bg()

    nx.draw_networkx_nodes(g, pos=pos1, node_size=node_size, node_color='red', edge_color='k', alpha=.5,
                           with_labels=False)
    nx.draw_networkx_edges(g, pos=pos1, edge_color='red', alpha=.3)

    nx.draw_networkx_nodes(G2, pos=pos2, node_size=node_size, node_color='blue', edge_color='k', alpha=.5,
                           with_labels=False)
    nx.draw_networkx_edges(G2, pos=pos2, edge_color='blue', alpha=.3)
    if node_labels:
        nx.draw_networkx_labels(g, pos=pos1, label_pos=0.5)
        nx.draw_networkx_labels(g, pos=pos2, label_pos=0.5)
    if edge_labels:
        nx.draw_networkx_edge_labels(g, pos=pos1, label_pos=0.5)
        nx.draw_networkx_edge_labels(g, pos=pos2, label_pos=0.5)

    plt.savefig('../data/graphs/two_networks.png')
    plt.show()


if __name__ == '__main__':
    gminas, gminas_pos = create_graphs.gminas_network()
    flights, flights_pos = create_flight_connections.flights_network()

    G = nx.compose(gminas, flights)

    plot_two_networks(G, gminas, gminas_pos, flights, flights_pos, False, False)
