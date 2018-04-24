import json
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':
    type = 'voivodeships'
    with open('../data/' + type + '_neighbours.json') as file:
        nbrs = json.load(file)
        g = nx.from_dict_of_lists(nbrs)
        nx.draw(g)
        plt.show()
