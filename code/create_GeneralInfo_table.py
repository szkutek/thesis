import sqlite3
from pprint import pprint
import pandas as pd
import requests
from bs4 import BeautifulSoup


def create_GeneralInfo_table(db):
    tables = pd.read_html('http://www.polskawliczbach.pl/Gminy')
    df1 = pd.DataFrame(tables[0], columns=['Gmina', 'Powiat', 'Województwo', 'Populacja'])

    cursor = db.cursor()

    for id, row in df1.iterrows():
        gmina, powiat, voivodship, population = row['Gmina'], row['Powiat'], row['Województwo'], row['Populacja']
        gmina = gmina.split('(')[0]
        powiat = powiat[7:]
        population = int(population.replace(' ', ''))

        cursor.execute(
            'INSERT INTO GeneralInfo(gmina, powiat, voivodship, population) VALUES (?,?,?,?)',
            (gmina, powiat, voivodship, population))
        db.commit()

    cursor.close()


def print_GeneralInfo_table(db):
    cursor = db.cursor()
    cursor.execute('SELECT * FROM GeneralInfo')

    a = cursor.fetchall()
    pprint(a)
    cursor.close()


if __name__ == '__main__':
    # # connect to sqlite
    db = sqlite3.connect('gminas.db')

    # create_GeneralInfo_table(db)

    print_GeneralInfo_table(db)

    db.close()
