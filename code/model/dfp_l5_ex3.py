import random as rnd
import matplotlib.pyplot as plt
import imageio
import pylab
import networkx as nx


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
        fig = pylab.figure()
        pos = nx.spring_layout(G)
        size = 200

        nx.draw_networkx_nodes(G, pos, nodelist=Susceptible, node_color='g', node_size=size, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=Infected_now, node_color='y', node_size=size, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

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
            nx.draw_networkx_nodes(G, pos, nodelist=Removed, node_color='k', node_size=size, alpha=0.8)

            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

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


def sim_infection(G, p, M, start):
    N = G.number_of_nodes()
    res = []
    for i in range(M):
        SIR = SIR_graph(G, prob=p, starting=start)
        w = []
        for vec in SIR['I']:
            # vec is a list of lists, each for different time
            # so w contains fraction of infected nodes at times t
            w.append(len(vec) / N)

        for t in range(len(w)):
            if len(res) <= t:
                res.append(w[t] / M)
            else:
                res[t] += w[t] / M
    return res


def zad3a():
    number_of_nodes = 30
    P = 0.8
    mv = True  # SAVE AND MAKE MOVIE
    pathname = 'fig'

    g = nx.barabasi_albert_graph(number_of_nodes, 2)
    SIR = SIR_graph(g, prob=P, save=mv, pathname=pathname)
    n = len(SIR['I'])
    movie(n, pathname, 'sir_ba', 1.)


def zad3bc():
    # G = nx.watts_strogatz_graph(100, 4, 0.5)
    G = nx.barabasi_albert_graph(100, 2)
    M = 1000
    P = [0.2, 0.5, 0.8]
    fig = plt.figure()
    for p in P:
        res = sim_infection(G, p, M, G.nodes()[0])
        plt.plot(res, linestyle='-', marker='o')

    plt.title('Fraction of infected nodes for P = ' + str(P))
    plt.legend(P)
    plt.xlabel('time')
    plt.ylabel('infected nodes')
    plt.show()


def zad3d(G, P, M=100):
    measure_prop_total_infected = []
    measure_time_to_most_i = []
    measure_time_to_clear_i = []
    # i = 0
    for p in P:
        mean_total_infected = 0
        mean_time_to_most_i = 0
        mean_time_to_clear_i = 0
        for m in range(M):
            SIR = SIR_graph(G, prob=p)
            # total proportion of the network that becomes infected
            total_infected = list(set(pylab.concatenate(SIR['I'])))
            mean_total_infected += len(total_infected) / M

            # time to clear infection
            time_to_clear_i = len(SIR['I'])
            mean_time_to_clear_i += time_to_clear_i / M

            # time to the largest number of infected nodes
            time_to_most_i = SIR['I'].index(max(SIR['I'], key=len))
            mean_time_to_most_i += time_to_most_i / M

        measure_prop_total_infected.append(mean_total_infected / G.number_of_nodes())
        measure_time_to_clear_i.append(mean_time_to_clear_i)
        measure_time_to_most_i.append(mean_time_to_most_i)

    return measure_prop_total_infected, measure_time_to_clear_i, measure_time_to_most_i


def plot_measures():
    N = 100
    g = nx.barabasi_albert_graph(N, 2)

    P = pylab.linspace(0.01, 1., 30)
    M = 100
    measure1, measure2, measure3 = [], [], []
    measure_prop_total_infected, measure_time_to_clear_i, measure_time_to_most_i = zad3d(g, P, M)
    measure1.append(measure_prop_total_infected)
    measure2.append(measure_time_to_clear_i)
    measure3.append(measure_time_to_most_i)

    fig1 = plt.figure()
    for m1 in measure1:
        plt.plot(P, m1, linestyle='-', marker='o')
    plt.title('total proportion of the network that becomes infected')
    plt.legend(['Barabasi-Albert'])
    plt.xlabel('probability')
    plt.ylabel('stats')

    fig2 = plt.figure()
    for m2 in measure2:
        plt.plot(P, m2, linestyle='-', marker='o')
    plt.title('time to clear infection')
    plt.legend(['Barabasi-Albert'])
    plt.xlabel('probability')
    plt.ylabel('stats')

    fig3 = plt.figure()
    for m3 in measure3:
        plt.plot(P, m3, linestyle='-', marker='o')
    plt.title('time to the largest number of infected nodes')
    plt.legend(['Barabasi-Albert'])
    plt.xlabel('probability')
    plt.ylabel('stats')

    plt.show()


if __name__ == "__main__":
    zad3a()  # and 3f
    # zad3bc()
    # # zad 3d
    # plot_measures()
