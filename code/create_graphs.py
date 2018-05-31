import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd


def plot_with_folium(pos):
    """pos = {name: [lat, lng]}"""
    import folium
    m = folium.Map(location=[51.9194, 19.1451], zoom_start=6, tiles='cartodbpositron')
    for k, pt in pos.items():
        folium.Marker(location=pt, popup=k).add_to(m)
    m.save('_map.html')


def read_files(type):
    """
    :param type: 'gminas', 'powiats', 'voivodeships'
    :return: tuple of dataframe with teryt number and geometry and dict {teryt: list of neighbouring teryts}
    """
    with open('../data/' + type + '_neighbours.json') as file:
        nbrs = json.load(file)  # dict {teryt: list of neighbouring teryts}
    df = gpd.read_file('../data/' + type + '.shp', encoding='utf-8')
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df.set_index('teryt', inplace=True, drop=False)

    return df, nbrs


def create_pos(df):
    pts = df.loc[:, ('pt_x', 'pt_y')].apply(tuple, axis=1)
    pos = dict(zip(df['teryt'], pts))
    df.loc[:, 'coord'] = pts
    return pos


def plot_pos(pos):
    type_bg = 'voivodeships'  # background areas
    df_bg = gpd.read_file('../data/' + type_bg + '.shp', encoding='utf-8')
    df_bg = gpd.GeoDataFrame(df_bg, geometry='geometry')

    x, y = zip(*pos.values())
    df_bg.plot()  # # df_bg.plot(column='name', cmap='GnBu')
    plt.plot(x, y, 'r.')
    plt.show()


def add_work_migration(G):
    """adding weights to edges
    if edge doesn't exits -> find shortest path and accumulate weights"""
    work_migration = pd.read_csv('../data/l_macierz_2014_03_18.csv', delimiter=';', dtype={0: str, 1: str})
    work_migration = work_migration.loc[:, work_migration.columns[:3]]
    migration = []
    work_migration.apply(lambda x: migration.append((x[0][:2], x[1][:2], x[2])),
                         axis=1)  # TODO remove [:2] (so that we calc for gminas)

    def set_work_migration(G, e1, e2, w):
        if 'work' in G[e1][e2]:
            G[e1][e2]['work'] += w
        else:
            G[e1][e2]['work'] = w

    path = nx.shortest_path(G)  # source and target not specified
    for e1, e2, w in migration:
        p = path[e1][e2]  # [e1, e2, e3, ..., en]
        pairs = list(zip(p[:-1], p[1:]))  # [(e1, e2), (e2, e3), ...]
        for p1, p2 in pairs:  # add work migration to edges in pathfile p
            set_work_migration(G, p1, p2, w)


def save_graph(G, node_labels=False, edge_labels=False):
    nx.draw_networkx_nodes(G, pos=pos, node_size=100, node_color='red', edge_color='k', alpha=.5,
                           with_labels=False)
    nx.draw_networkx_edges(G, pos=pos, edge_color='red', alpha=.3)
    if node_labels:
        nx.draw_networkx_labels(G, pos=pos, label_pos=0.5)
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos=pos, label_pos=0.5)
    plt.savefig('../data/graphs/graph_' + type + '.png')
    plt.show()


def create_gminas_graph(pos, nbrs):
    G = nx.DiGraph()
    G.add_nodes_from(pos.keys())
    nbrs = sum([list(map(lambda el: (k, el), v)) for k, v in nbrs.items()], [])  # list of tuples
    G.add_edges_from(nbrs)
    add_work_migration(G)
    return G


def gminas_network():
    # type = 'gminas'
    type = 'voivodeships'
    df, nbrs = read_files(type)
    pos = create_pos(df)  # {node: (pt_x, pt_y)}
    G = create_gminas_graph(pos, nbrs)
    return G, pos


if __name__ == '__main__':
    # type = 'gminas'
    type = 'voivodeships'
    df, nbrs = read_files(type)
    df.plot(alpha=0.3)
    # plt.show()
    pos = create_pos(df)  # {node: (pt_x, pt_y)} (coordinates)
    G = create_gminas_graph(pos, nbrs)
    save_graph(G, True, True)
