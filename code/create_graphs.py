import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    type = 'voivodeships'
    with open('../data/' + type + '_neighbours.json') as file:
        nbrs = json.load(file)  # dict {gminas teryt: list of nbrs teryts}
        g = nx.from_dict_of_lists(nbrs)

        # work_migration = pd.read_csv('../data/l_macierz_2014_03_18.csv', delimiter=';',
        #                              dtype={0: str, 1: str, 2: str})
        # work_migration = work_migration.head(10)
        # work_migration = work_migration.loc[:, work_migration.columns[:3]]
        # edges = []
        # work_migration.apply(lambda x: edges.append((x[0], x[1], x[2])), axis=1)
        # g.add_edges_from(edges)

        edges = [('02', '04', '20'), ('06', '08', '50')]
        print(edges)
        nx.draw(g)
        # nx.draw_networkx_edge_labels(g)

        plt.show()
