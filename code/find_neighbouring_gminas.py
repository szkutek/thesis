import json
import subprocess
import geopandas as gpd
import shapely.geometry

import create_shapefiles


def convert_to_kml_coords():
    path = '../data/simplified_geometry/shp/'
    df = gpd.read_file(path + 'gminy.shp', encoding='windows-1250')
    df = gpd.GeoDataFrame(df, geometry='geometry')
    print(df.head())

    subprocess.call("ogr2ogr -f KML " + path + 'gminy.kml ' + path + 'gminas.shp', shell=True)


def save_neighbours_to_json(type):
    df = gpd.read_file('../data/' + type + '.shp', encoding='utf-8')
    df = gpd.GeoDataFrame(df, geometry='geometry')
    neighbours = {}
    for i1, r1 in df.iterrows():
        nbrs = set()
        for i2, r2 in df.iterrows():
            try:
                if i1 != i2 and shapely.geometry.shape(r1['geometry']).touches(r2['geometry']):
                    nbrs.add(r2['teryt'])
            except:
                print(r1['name'], r2['name'])
                nbrs.add(r2['teryt'])
        neighbours[r1['teryt']] = list(nbrs)

    neighbours = sum([list(map(lambda el: (k, el), v)) for k, v in neighbours.items()],
                     [])  # list of tuples (teryt1, teryt2)
    json.dump(neighbours, open('../data/' + type + '_neighbours.json', 'w'))


if __name__ == '__main__':
    # check which polygons are neighbours -> shapely.geometry.shape(poly1).touches(poly2)
    type = 'gminas'
    # type = 'voivodeships'
    create_shapefiles.create_shp(type)
    # save_neighbours_to_json(type)
    with open('../data/' + type + '_neighbours.json') as file:
        nbrs = json.load(file)
        print(nbrs[:10])
