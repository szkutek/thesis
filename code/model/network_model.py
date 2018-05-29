import random as rnd
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pylab


def integrate(f, y0, t, args):
    res = np.ndarray([len(t), len(y0)])
    y = np.array([0., 0., 0.])
    tau = t[1] - t[0]
    for i, _ in enumerate(t):
        y0 += np.array(y) * tau
        res[i, :] = y0
        y = f(y0, t, args)
    return res


def SIR_model_ODE(y, t, args):  # SIR model
    beta, mu, infected_from_nbrs = args
    Si = y[0]  # susceptible
    Ii = y[1]  # infected
    Ri = y[2]  # recovered
    N = sum(y)
    # TODO modify to work on graph
    dS_dt = - beta * Si * Ii / N
    dI_dt = beta * Si * Ii / N - mu * Ii
    dR_dt = mu * Ii
    return dS_dt, dI_dt, dR_dt


def plot_change_in_population(filename, t, S, I, R=None):
    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    if R is not None:
        plt.plot(t, R, label='Recovered')
        title = 'Plague model for S(0)=' + str(S[0]) + ', I(0)=' + str(I[0]) + ' and R(0)=' + str(R[0])
    else:
        title = 'Plague model for S(0)=' + str(S[0]) + ', I(0)=' + str(I[0])

    plt.xlabel('time')
    plt.ylabel('population')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig(filename + '.png')


def sir_on_node(g, node, t, beta, mu, nbr_inf=1, i0=0):
    N = g.nodes[node]['population']
    R0 = beta * N / mu  # basic reproductive ratio
    print(R0)

    infection_from_nbrs = 0.
    if nbr_inf:
        nbrs = [*nx.neighbors(g, node)]
        for nbr in nbrs:
            infection_from_nbrs += g.get_edge_data(node, nbr)['commute'] / g.nodes[nbr]['population']

    y0 = [N - i0, i0, 0]  # initial condition vector
    soln = integrate(SIR_model_ODE, y0, t, args=(beta, mu, infection_from_nbrs))
    s = soln[:, 0]
    i = soln[:, 1]
    r = soln[:, 2]

    # nx.set_node_attributes(g, {node: s}, 'S')
    # nx.set_node_attributes(g, {node: i}, 'I')
    # nx.set_node_attributes(g, {node: r}, 'R')
    return s, i, r


def agent_sir_on_network():
    def SIR_graph(G, prob=0.5, save=False, pathname='graph_walk', starting=None):
        Susceptible = list(G.nodes())
        if starting is None:
            start = Susceptible[rnd.randint(0, len(Susceptible) - 1)]
        else:
            # start = starting
            start = Susceptible[0]

        Infected_now = [start]
        Susceptible.remove(start)

        Infected_next = []
        Removed = []

        SIR = {'S': [Susceptible], 'I': [Infected_now], 'R': [Removed]}

        if save:
            fig = pylab.figure(figsize=(15, 10))
            pos = nx.spring_layout(G)
            size = 300

            nx.draw_networkx_nodes(G, pos, nodelist=Susceptible, node_color='g', node_size=size, alpha=0.8)
            nx.draw_networkx_nodes(G, pos, nodelist=Infected_now, node_color='y', node_size=size, alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

            nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'population'))
            # nx.draw_networkx_edge_labels(G, pos, label_pos=.75)

            pylab.savefig(pathname + '_0.png')
            pylab.clf()

        i = 1
        while Infected_now:
            for u in Infected_now:
                for v in G.neighbors(u):
                    if rnd.random() < prob and v in Susceptible:
                        Infected_next.append(v)
                        Susceptible.remove(v)

                Removed.append(u)

            Infected_now = Infected_next
            Infected_next = []

            if save:
                nx.draw_networkx_nodes(G, pos, nodelist=Susceptible, node_color='g', node_size=size, alpha=0.8)
                nx.draw_networkx_nodes(G, pos, nodelist=Infected_now, node_color='y', node_size=size, alpha=0.8)
                nx.draw_networkx_nodes(G, pos, nodelist=Removed, node_color='grey', node_size=size, alpha=0.8)

                nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
                nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'population'))
                # nx.draw_networkx_edge_labels(G, pos, label_pos=.75)

                s = pathname + "_" + str(i) + ".png"
                pylab.savefig(s)
                pylab.clf()

            SIR['S'].append(Susceptible)
            SIR['I'].append(Infected_now)
            SIR['R'].append(Removed)

            i += 1

        if save:
            pylab.close(fig)
        return SIR

    def movie(n, pathname, moviename, duration=0.2):
        frames = []
        for i in range(n):
            path = pathname + "_" + str(i) + '.png'
            frames.append(imageio.imread(path))

        kargs = {'duration': duration}
        imageio.mimwrite(moviename + '.gif', frames, 'gif', **kargs)

    def sir_on_network(g):
        P = 0.8
        mv = True  # SAVE AND MAKE MOVIE
        pathname = 'fig'
        SIR = SIR_graph(g, prob=P, save=mv, pathname=pathname)
        movie(len(SIR['I']), pathname, 'sir_ba', 1.)

    g = create_graph()
    sir_on_network(g)


def create_graph():
    number_of_nodes = 20

    g = nx.barabasi_albert_graph(number_of_nodes, 2)
    population = {node: rnd.randint(100, 1000) for node in g.nodes()}
    area = {node: rnd.randint(1000, 1500) for node in g.nodes()}
    density = {k: population[k] / area[k] for k in population}
    nx.set_node_attributes(g, population, 'population')
    nx.set_node_attributes(g, area, 'area')
    nx.set_node_attributes(g, density, 'density')

    # initialize SIR
    # starting_node = 1
    # nx.set_node_attributes(g, {node: p for node, p in population.items()}, 'N')
    # nx.set_node_attributes(g, {node: p for node, p in population.items()}, 'S')
    # nx.set_node_attributes(g, {node: 0 for node in g.nodes()}, 'I')
    # nx.set_node_attributes(g, {node: 0 for node in g.nodes()}, 'R')

    # g = g.to_directed() # do we need this???
    commute = {edge: rnd.randint(50, 100) for edge in g.edges()}
    nx.set_edge_attributes(g, commute, 'commute')
    return g


if __name__ == "__main__":
    g = create_graph()
    node = 1

    t = np.linspace(0, 100., 1001)  # time grid
    beta, mu = 1., 0.03

    nbr_inf = 0
    s, i, r = sir_on_node(g, node, t, beta, mu, nbr_inf=nbr_inf, i0=1)
    plot_change_in_population('test2_' + str(nbr_inf), t, s, i, r)
    nbr_inf = 1
    s, i, r = sir_on_node(g, node, t, beta, mu, nbr_inf=nbr_inf, i0=1)
    plot_change_in_population('test2_' + str(nbr_inf), t, s, i, r)

    # tmp = g.nodes[node]
    # print(tmp)
    # nbrs = [*nx.neighbors(g, node)]
    # print(nbrs)
    # for nbr in nbrs:
    #     print(nbr, g.get_edge_data(node, nbr)['commute'])
