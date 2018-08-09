import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import networkx.readwrite.gpickle as pickle


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

    # rows with NaN population - added manually from polskawliczbach.pl
    # df.loc[:, 'population'] = df.loc[:, 'teryt'].map(population)
    # print(df[df['population'].isnull()].loc[:, 'teryt'].tolist())
    nan_population = {'0265011': 115453, '0202033': 9416, '0612023': 6743, '0603113': 6969, '0603153': 6596,
                      '0618053': 6444, '0607083': 8773, '0605063': 7088, '0804073': 6892, '1404043': 6217,
                      '1412123': 8743, '1609123': 5257, '1818053': 8675, '2213013': 3264, '2211023': 3824,
                      '2211043': 15467, '2601083': 5611, '2604073': 6954, '2601063': 7744, '2605043': 9044,
                      '2604123': 15801, '3020013': 4764, '3006013': 8346, '3007083': 10850, '3020033': 8253,
                      '3209053': 5006, '3204073': 4931}
    population.update(nan_population)

    nx.set_node_attributes(G, population, 'population')
    df.loc[:, 'population'] = df.loc[:, 'teryt'].map(population)


def add_work_migration(G):
    """adding weights to edges
    if edge doesn't exits -> find shortest path and accumulate weights"""
    work_migration = pd.read_csv('../data/l_macierz_2014_03_18.csv', delimiter=';', dtype={0: str, 1: str})
    work_migration = work_migration.loc[:, work_migration.columns[:3]]
    work_migration = work_migration.apply(tuple, axis=1).tolist()

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
            G[p1][p2]['commute'] += w

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
    G = nx.Graph()
    G.add_nodes_from(pos.keys())
    nbrs = [(n1, n2, 0) for n1, n2 in nbrs]
    G.add_weighted_edges_from(nbrs, 'commute')
    add_population(G, df)
    add_work_migration(G)
    return G


def gminas_network(create=False):
    path = '../data/pickled_graphs/gminas.pkl'
    df, nbrs = read_files('gminas')
    pos = create_pos(df)  # {node: (pt_x, pt_y)}
    if create:
        G = create_gminas_graph(pos, nbrs, df)
        pickle.write_gpickle(G, path)
    else:
        G = pickle.read_gpickle(path)
    return G, pos, df


if __name__ == '__main__':
    G, pos, df = gminas_network()
    # save_graph(G, pos)
    print(list(G.nodes.data())[:3])
    print(list(G.edges.data())[:3])
    print(df.loc[:, 'teryt'].tolist())
