import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import geopandas as gpd


def find_gmina_in_voiv(gmina, voiv, gminas_df):
    voiv_df = gpd.read_file('../data/voivodeships.shp', encoding='utf-8')
    voiv_df = gpd.GeoDataFrame(voiv_df, geometry='geometry')
    voiv_df.set_index('name', inplace=True, drop=False)
    for i, r in gminas_df.iterrows():
        if r['name'] == gmina and r['teryt'][:2] == voiv_df.loc[voiv, 'teryt']:
            return r['teryt'], r['pt_x'], r['pt_y']


def add_airport_coordinates(table):
    coord = [(52.1672, 20.9679), (50.0770, 19.7881), (54.3788, 18.4681), (50.4728, 19.0759), (52.4493, 20.6512),
             (51.1042, 16.8809), (52.4200, 16.8286), (50.1148, 22.0246), (53.5859, 14.9028), (51.2358, 22.7151),
             (53.0980, 17.9727), (51.7197, 19.3908), (53.4806, 20.9362), (52.1372, 15.7781), (51.3931, 21.1994)]
    table.set_index('IATA', drop=False, inplace=True)

    df = gpd.read_file('../data/gminas.shp', encoding='utf-8')
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df.set_index('teryt', inplace=True, drop=False)

    coord = []
    teryts = []
    for i, r in table.iterrows():
        teryt, pt_x, pt_y = find_gmina_in_voiv(r['gmina'], r['voivodeship'], df)
        teryts.append(teryt)
        coord.append((pt_x, pt_y))
    table.loc[:, 'teryt'] = teryts
    c_x, c_y = zip(*coord)
    table.loc[:, 'coord_x'] = c_x
    table.loc[:, 'coord_y'] = c_y


def create_df_airports(link):
    table = 'https://pl.wikipedia.org/wiki/Porty_lotnicze_w_Polsce'

    table = pd.read_html(table, header=0)
    table = pd.DataFrame(table[0][:15],
                         columns=('Główne miasto dla portu', 'Województwo', 'Gmina/miasto',
                                  # 'Nazwa toponomiczna (miejscowość/dzielnica)',
                                  # 'ICAO',
                                  'IATA',
                                  'Nazwa portu lotniczego',
                                  'Ofic. liczba pasażerów (2016)[2]',
                                  'Ofic. liczba pasażerów (2017)[2]'))
    table.rename(columns={'Główne miasto dla portu': 'city',
                          'Województwo': 'voivodeship', 'Gmina/miasto': 'gmina',
                          'Nazwa portu lotniczego': 'airport name',
                          'Ofic. liczba pasażerów (2016)[2]': 'passengers 2016',
                          'Ofic. liczba pasażerów (2017)[2]': 'passengers 2017'}, inplace=True)
    add_airport_coordinates(table)
    table.to_csv(link, sep=';', encoding='utf-8', index=False)


def read_airports(create=False):
    link = '../data/airports_poland.csv'
    if create:
        create_df_airports(link)
    table = pd.read_csv(link, sep=';', encoding='utf-8', dtype={'teryt': str})
    table.set_index('IATA', drop=False, inplace=True)
    return table


def create_pos_for_shp(df):
    pts = df.loc[:, ('coord_x', 'coord_y')].apply(tuple, axis=1)
    # # for folium replace y and x:
    # pts = df.loc[:, ('coord_x', 'coord_y')].apply(tuple, axis=1)
    pos = dict(zip(df['IATA'], pts))
    df.loc[:, 'coords'] = pts
    return pos


def save_graph(G, node_labels=False, edge_labels=False):
    nx.draw_networkx_nodes(G, pos=pos, node_size=100, node_color='red', edge_color='k', alpha=.5,
                           with_labels=False)
    nx.draw_networkx_edges(G, pos=pos, edge_color='red', alpha=.3)
    if node_labels:
        nx.draw_networkx_labels(G, pos=pos, label_pos=0.5)
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos=pos, label_pos=0.5)
    plt.savefig('../data/graphs/graph_airports.png')
    plt.show()


def plot_with_folium(pos):
    """pos = {name: [lat, lng]}"""
    import folium
    m = folium.Map(location=[51.9194, 19.1451], zoom_start=6, tiles='cartodbpositron')
    for k, pt in pos.items():
        folium.Marker(location=pt, popup=k).add_to(m)
    m.save('../data/graphs/airports.html')


def create_airports_graph(pos):
    airports = nx.DiGraph()
    airports.add_nodes_from(pos.keys())
    number_of_passengers = 70
    flights = {('WAW', 'WRO'): 6, ('WAW', 'KTW'): 2, ('WAW', 'POZ'): 1, ('WAW', 'RZE'): 4,
               ('WAW', 'SZZ'): 4, ('WAW', 'IEG'): 1, ('GDN', 'WRO'): 0.5, ('WAW', 'KRK'): 2,
               ('WAW', 'GDN'): 3}
    edges = [(k[0], k[1], flights_per_day * number_of_passengers) for k, flights_per_day in flights.items()]
    edges += [(k[1], k[0], flights_per_day * number_of_passengers) for k, flights_per_day in flights.items()]
    airports.add_weighted_edges_from(edges)
    return airports


def flights_network():
    df = read_airports()
    pos = create_pos_for_shp(df)
    G = create_airports_graph(pos)
    return G, pos


if __name__ == '__main__':
    df = read_airports(create=False)
    # print(df)
    pos = create_pos_for_shp(df)
    G = create_airports_graph(pos)
    save_graph(G, True, True)
    plot_with_folium(pos)
