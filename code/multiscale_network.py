import create_graphs
import create_flight_connections
import networkx as nx
import matplotlib.pyplot as plt


def plot_network(G1, G2, pos1, pos2, node_labels=False, edge_labels=False):
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
    print(gminas.nodes())
    print(gminas_pos)
    print(flights.nodes())
    print(flights_pos)

    plot_network(gminas, flights, gminas_pos, flights_pos, True, True)
