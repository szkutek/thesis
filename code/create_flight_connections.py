import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import geopandas as gpd
import networkx.readwrite.gpickle as pickle


def find_gmina_in_voiv(gmina, voiv, gminas_df, voiv_df):
    for i, r in gminas_df.iterrows():
        if r['name'] == gmina and r['teryt'][:2] == voiv_df.loc[voiv, 'teryt']:
            return r['teryt'], r['pt_x'], r['pt_y']


def add_airport_coordinates(table):
    coord = [(52.1672, 20.9679), (50.0770, 19.7881), (54.3788, 18.4681), (50.4728, 19.0759), (52.4493, 20.6512),
             (51.1042, 16.8809), (52.4200, 16.8286), (50.1148, 22.0246), (53.5859, 14.9028), (51.2358, 22.7151),
             (53.0980, 17.9727), (51.7197, 19.3908), (53.4806, 20.9362), (52.1372, 15.7781), (51.3931, 21.1994)]
    table.set_index('IATA', drop=False, inplace=True)

    gminas_df = gpd.read_file('../data/gminas.shp', encoding='utf-8')
    gminas_df = gpd.GeoDataFrame(gminas_df, geometry='geometry')
    gminas_df.set_index('teryt', inplace=True, drop=False)

    voiv_df = gpd.read_file('../data/voivodeships.shp', encoding='utf-8')
    voiv_df = gpd.GeoDataFrame(voiv_df, geometry='geometry')
    voiv_df.set_index('name', inplace=True, drop=False)

    coord = []
    teryts = []
    for i, r in table.iterrows():
        teryt, pt_x, pt_y = find_gmina_in_voiv(r['gmina'], r['voivodeship'], gminas_df, voiv_df)
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
    pts = df.loc[:, ('coord_x', 'coord_y')].apply(tuple, axis=1)  # for folium replace y and x
    pos = dict(zip(df['IATA'], pts))
    pos = {df.loc[k, 'teryt']: v for k, v in pos.items()}
    df.loc[:, 'coords'] = pts
    return pos


def save_graph(G, pos, node_labels=False, edge_labels=False):
    nx.draw_networkx_nodes(G, pos=pos, node_size=100, node_color='red', edge_color='k', alpha=.5,
                           with_labels=False)
    nx.draw_networkx_edges(G, pos=pos, edge_color='red', alpha=.3)
    if node_labels:
        nx.draw_networkx_labels(G, pos=pos, label_pos=0.5)
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos=pos, label_pos=0.5)
    plt.savefig('../data/graphs/graph_airports.png')
    plt.show()


def create_airports_graph(df, pos):
    number_of_passengers = 70
    flights = {('WAW', 'WRO'): 6, ('WAW', 'KTW'): 2, ('WAW', 'POZ'): 1, ('WAW', 'RZE'): 4,
               ('WAW', 'SZZ'): 4, ('WAW', 'IEG'): 1, ('GDN', 'WRO'): 0.5, ('WAW', 'KRK'): 2,
               ('WAW', 'GDN'): 3}
    flights = {(df.loc[k[0], 'teryt'], df.loc[k[1], 'teryt']): v for k, v in flights.items()}
    edges = [(k[0], k[1], flights_per_day * number_of_passengers) for k, flights_per_day in flights.items()]
    # for DiGraph:
    # edges += [(k[1], k[0], flights_per_day * number_of_passengers) for k, flights_per_day in flights.items()]

    airports = nx.Graph()
    airports.add_nodes_from(pos.keys())
    airports.add_weighted_edges_from(edges, weight='commute')
    return airports


def flights_network(create=False):
    path = '../data/pickled_graphs/flights.pkl'
    df = read_airports()
    pos = create_pos_for_shp(df)
    if create:
        G = create_airports_graph(df, pos)
        pickle.write_gpickle(G, path)
    else:
        G = pickle.read_gpickle(path)
    return G, pos


if __name__ == '__main__':
    df = read_airports(create=False)
    # print(df)
    pos = create_pos_for_shp(df)
    G, pos = create_airports_graph(df, pos)
    save_graph(G, pos, True, True)
