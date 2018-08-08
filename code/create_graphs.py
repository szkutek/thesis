import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd


def read_files(type):
    """
    :param type: 'gminas', 'powiats', 'voivodeships'
    :return: tuple of dataframe with teryt number, representative point and geometry
    and list of tuples [[n1, n2], [n1, n3], ...]
    """
    with open('../data/' + type + '_neighbours.json') as file:
        nbrs = json.load(file)  # list of tuples [[n1, n2], [n1, n3], ...]
    df = gpd.read_file('../data/' + type + '.shp', encoding='utf-8')
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df.set_index('teryt', inplace=True, drop=False)

    return df, nbrs


def create_pos(df):
    pts = df.loc[:, ('pt_x', 'pt_y')].apply(tuple, axis=1)
    df.loc[:, 'coord'] = pts
    return dict(zip(df['teryt'], pts))


def plot_bg():
    # BACKGROUND
    type_bg = 'voivodeships'  # background areas
    df_bg = gpd.read_file('../data/' + type_bg + '.shp', encoding='utf-8')
    df_bg = gpd.GeoDataFrame(df_bg, geometry='geometry')
    df_bg.plot(alpha=0.3, figsize=(10, 10))  # # df_bg.plot(column='name', cmap='GnBu')


def add_population(G, df):
    pop_df = pd.read_csv('../data/ludnosc/LUDN_2017.csv', delimiter=';', dtype={0: str})
    pop_df.rename(columns={'Kod': 'teryt',
                           'gminy bez miast na prawach powiatu;miejsce zamieszkania;stan na 31 XII;ogółem;2017;[osoba]':
                               'pop_no_cities',
                           'miasta na prawach powiatu;miejsce zamieszkania;stan na 31 XII;ogółem;2017;[osoba]':
                               'pop_cities'},
                  inplace=True)
    pop = pop_df.loc[:, ('pop_no_cities', 'pop_cities')].apply(sum, axis=1).tolist()
    population = dict(zip(pop_df.loc[:, 'teryt'].tolist(), pop))
    print(len(pop))
    print(len(population))
    nx.set_node_attributes(G, population, 'population')

    df.loc[:, 'population'] = df.loc[:, 'teryt'].map(population)


def add_work_migration(G):
    """adding weights to edges
    if edge doesn't exits -> find shortest path and accumulate weights"""
    work_migration = pd.read_csv('../data/l_macierz_2014_03_18.csv', delimiter=';', dtype={0: str, 1: str})
    work_migration = work_migration.loc[:, work_migration.columns[:3]]
    work_migration = work_migration.apply(tuple, axis=1).tolist()

    def set_work_migration(G, e1, e2, w):
        if 'work' in G[e1][e2]:
            G[e1][e2]['commute'] += w
        else:
            G[e1][e2]['commute'] = w

    path = nx.shortest_path(G)  # source and target not specified
    errs = 0
    for e1, e2, w in work_migration:
        # TODO fix key error in path
        try:
            p = path[e1][e2]  # [e1, e2, e3, ..., en]
        except:
            errs += 1
            p = []
        pairs = list(zip(p[:-1], p[1:]))  # [(e1, e2), (e2, e3), ...]
        for p1, p2 in pairs:  # add work migration to edges in path p
            set_work_migration(G, p1, p2, w)
    print('errors in work migration: ' + str(errs))


def save_graph(G, pos, node_labels=False, edge_labels=False):
    plot_bg()
    nx.draw_networkx_nodes(G, pos=pos, node_size=100, node_color='red', edge_color='k', alpha=.5,
                           with_labels=False)
    nx.draw_networkx_edges(G, pos=pos, edge_color='red', alpha=.3)
    if node_labels:
        nx.draw_networkx_labels(G, pos=pos, label_pos=0.5)
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos=pos, label_pos=0.5)
    plt.savefig('../data/graphs/graph_gminas.png')
    plt.show()


def create_gminas_graph(pos, nbrs, df):
    G = nx.DiGraph()
    G.add_nodes_from(pos.keys())
    G.add_edges_from(nbrs)
    add_population(G, df)
    # add_work_migration(G)
    return G


def gminas_network():
    df, nbrs = read_files('gminas')
    pos = create_pos(df)  # {node: (pt_x, pt_y)}
    G = create_gminas_graph(pos, nbrs, df)
    return G, pos, df


if __name__ == '__main__':
    G, pos, df = gminas_network()
    # save_graph(G, pos)
    print(list(G.nodes.data())[:3])
    print(list(G.edges.data())[:3])
    print(df.loc[:, 'teryt'].tolist())
