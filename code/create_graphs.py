import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    type = 'gminas'
    with open('../data/' + type + '_neighbours.json') as file:
        nbrs = json.load(file)  # dict {gminas teryt: list of nbrs teryts}
        G = nx.from_dict_of_lists(nbrs)

        work_migration = pd.read_csv('../data/l_macierz_2014_03_18.csv', delimiter=';',
                                     dtype={0: str, 1: str})
        work_migration = work_migration.loc[:, work_migration.columns[:3]]
        edges = []
        work_migration.apply(lambda x: edges.append((x[0], x[1], x[2])), axis=1)
        # TODO add weights to edges
        # TODO if edge doesn't exits -> find shortest path and accumulate weights
        edges = {(e1, e2): w for e1, e2, w in edges}
        print(edges)

        # edges = {('02', '04'): '20', ('06', '08'): '50'}
        # G.add_edges_from(edges)
        # pos = nx.spring_layout(G)
        #
        # nx.draw(G)
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edges)
        #
        # plt.savefig('graph_' + type + '.png')
