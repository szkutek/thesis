import matplotlib.pyplot as plt
import geopandas as gpd
import fiona
import pysal as ps
from pysal.contrib.viz import mapping as maps

# # http://darribas.org/gds15/content/labs/lab_03.html
lsoas_link = '../data/PRG_jednostki_administracyjne_v22/gminy.shp'
lsoas = gpd.read_file(lsoas_link)

print(lsoas['jpt_nazwa_'])
print(lsoas.head())

# TODO check if point is inside polygon (to match polygon with gmina in the correct powiat and voivodship name)

# f, ax = plt.subplots(1)
# ax = lsoas.plot(ax=ax)
# ax.set_axis_off()
# plt.show()
