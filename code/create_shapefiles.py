import geopandas as gpd
import pandas as pd


def add_centerpoint(df):
    """Add centerpoint do DataFrame and set index on teryt number"""
    # coordinates of representative points of Polygons
    # we can use obj.representative_point() or obj.centroid
    centerpoints = df.apply(lambda r: r['geometry'].representative_point(), axis=1)
    centerpoints_x = centerpoints.apply(lambda r: r.x)
    centerpoints_y = centerpoints.apply(lambda r: r.y)

    df.loc[:, 'pt_x'] = centerpoints_x
    df.loc[:, 'pt_y'] = centerpoints_y


def create_df_voivodeships(shp_link):
    shp = gpd.read_file(shp_link, encoding='windows-1250')
    df = pd.DataFrame(shp, columns=['jpt_kod_je', 'jpt_nazwa_', 'geometry'])
    df.rename(columns={'jpt_kod_je': 'teryt', 'jpt_nazwa_': 'name'}, inplace=True)

    add_centerpoint(df)
    df.to_file('../data/voivodeships.shp', driver='ESRI Shapefile', encoding='utf-8')


def create_df_powiats(shp_link):
    shp = gpd.read_file(shp_link)
    df = shp.loc[:, ('jpt_kod_je', 'jpt_nazwa_', 'geometry')]
    df.rename(columns={'jpt_kod_je': 'teryt', 'jpt_nazwa_': 'name'}, inplace=True)

    add_centerpoint(df)
    df.to_file('../data/powiats.shp', driver='ESRI Shapefile', encoding='utf-8')


def create_df_gminas(shp_link):
    shp = gpd.read_file(shp_link, encoding='windows-1250')
    df = shp.loc[:, ('jpt_kod_je', 'jpt_nazwa_', 'geometry')]
    df.rename(columns={'jpt_kod_je': 'teryt', 'jpt_nazwa_': 'name'}, inplace=True)

    add_centerpoint(df)
    df.to_file('../data/gminas.shp', driver='ESRI Shapefile', encoding='utf-8')


def create_shp(type):
    if type == 'voivodeships':
        create_df_voivodeships(pathfile + 'wojewodztwa.shp')
    elif type == 'powiats':
        create_df_powiats(pathfile + 'powiaty.shp')
    elif type == 'gminas':
        create_df_gminas(pathfile + 'gminy.shp')


if __name__ == '__main__':
    pathfile = '../data/simplified_geometry/shp/'

    type = 'gminas'
    # type = 'voivodeships'
    create_shp(type)

    df = gpd.read_file('../data/' + type + '.shp', encoding='utf-8')
    df = gpd.GeoDataFrame(df, geometry='geometry')

    print(df.head())
