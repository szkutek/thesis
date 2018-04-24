import geopandas as gpd
import pandas as pd
import shapely.geometry
import json


def create_df_voivodeships(df_terc):
    shp_link = '../data/PRG_jednostki_administracyjne_v22/wojewodztwa.shp'
    shp = gpd.read_file(shp_link)
    df_shp = pd.DataFrame(shp, columns=['jpt_kod_je', 'geometry'])
    df_shp.rename(columns={'jpt_kod_je': 'teryt'}, inplace=True)

    cond = df_terc['POW'].isnull()
    df_names = pd.DataFrame(df_terc[cond], columns=['WOJ', 'NAZWA'])
    df_names.rename(columns={'WOJ': 'teryt', 'NAZWA': 'name'}, inplace=True)
    df_names['name'] = df_names['name'].map(lambda x: x.lower())

    df = pd.merge(df_names, df_shp, on='teryt')
    # df.to_csv('../data/voivodeships.csv', sep='\t', encoding='utf-8', index=False)
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df.to_file('../data/voivodeships.shp', driver='ESRI Shapefile', encoding='utf-8')


def create_df_powiats(df_terc):
    shp_link = '../data/PRG_jednostki_administracyjne_v22/powiaty.shp'
    shp = gpd.read_file(shp_link)
    df_shp = shp.loc[:, ('jpt_kod_je', 'geometry')]
    df_shp.rename(columns={'jpt_kod_je': 'teryt'}, inplace=True)

    df_terc = df_terc[df_terc.loc[:, 'POW'].notnull()]
    df_terc = df_terc[df_terc.loc[:, 'GMI'].isnull()]

    teryts = df_terc.apply(lambda x: x['WOJ'] + x['POW'], axis=1)
    df_terc['teryt'] = teryts

    df_names = df_terc.loc[:, ('teryt', 'NAZWA')]
    df_names.rename(columns={'NAZWA': 'name'}, inplace=True)

    df = pd.merge(df_names, df_shp, on='teryt')
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df.to_file('../data/powiats.shp', driver='ESRI Shapefile', encoding='utf-8')


def create_df_gminas(df_terc):
    shp_link = '../data/PRG_jednostki_administracyjne_v22/gminy.shp'
    shp = gpd.read_file(shp_link)
    df_shp = shp.loc[:, ('jpt_kod_je', 'geometry')]
    df_shp.rename(columns={'jpt_kod_je': 'teryt'}, inplace=True)

    cond = df_terc.loc[:, 'RODZ'].notnull()
    df_terc = df_terc[cond]

    teryts = df_terc.apply(lambda x: x['WOJ'] + x['POW'] + x['GMI'] + x['RODZ'], axis=1)
    df_terc.loc[:, 'teryt'] = teryts

    df_names = df_terc.loc[:, ('teryt', 'NAZWA')]
    df_names.rename(columns={'NAZWA': 'name'}, inplace=True)

    df = pd.merge(df_names, df_shp, on='teryt')
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df.to_file('../data/gminas.shp', driver='ESRI Shapefile', encoding='utf-8')


def create_shp(type):
    terc_link = '../data/TERC_Urzedowy_2018-04-17/TERC_Urzedowy_2018-04-17.csv'
    df_terc = pd.read_csv(terc_link, sep=';', dtype={'WOJ': str, 'POW': str, 'GMI': str, 'RODZ': str})

    if type == 'voivodeships':
        create_df_voivodeships(df_terc)
    elif type == 'powiats':
        create_df_powiats(df_terc)
    elif type == 'gminas':
        create_df_gminas(df_terc)


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
    json.dump(neighbours, open('../data/' + type + '_neighbours.json', 'w'))


# f, ax = plt.subplots(1)
# ax = shp.plot(ax=ax)
# ax.set_axis_off()
# plt.show()

# Plot one Polygon
# poly = shp.loc[0, 'geometry']
# x,y = poly.exterior.xy
# plt.plot(x,y)


if __name__ == '__main__':
    # check which polygons are neighbours -> shapely.geometry.shape(poly1).touches(poly2)

    type = 'gminas'
    # create_shp(type)
    # save_neighbours_to_json(type)
    with open('../data/' + type + '_neighbours.json') as file:
        nbrs = json.load(file)
        print(list(nbrs.items())[:5])
