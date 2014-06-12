from __future__ import division
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy

colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']

def visualiseTime(x_lst, y_lst, filename="tempTime.png"):
    matplotlib.rcParams.update({'font.size': 32})
    fig = plt.figure()
    ax = fig.gca()

    ax.set_yscale('log')

    plots = []
    for i in range(len(x_lst)):
        plt.plot(x_lst[i], y_lst[i], colors[i] + '--')
        p, = plt.plot(x_lst[i], y_lst[i], colors[i] + 'o', markersize=10) # plot dots
        plots.append(p)

    plt.legend(plots, ["CCWP", "DD"], loc=2)
    plt.grid()
    plt.xlabel('Seed set size')
    plt.ylabel('Time (secs)')
    plt.title('HepNEPT data. p = .01')
    fig.set_size_inches(18.5,10.5)

    fig.savefig(filename, dpi=fig.dpi)
    # plt.show()

def visualiseResults(x_lst, y_lst, filename="tempResults.png"):
    matplotlib.rcParams.update({'font.size': 32})

    fig = plt.figure()
    ax = fig.gca()

    # length of axes
    # max_x = max(x_lst, key=lambda x: x[-1][0])
    # max_y = max(y_lst, key=lambda y: y[-1][0])

    # ax.set_xticks(numpy.arange(0, max_x, 25))
    # ax.set_yticks(numpy.arange(0, max_y, 200))

    plots = []
    for i in range(len(x_lst)):
        plt.plot(x_lst[i], y_lst[i], colors[i] + '--')
        p, = plt.plot(x_lst[i], y_lst[i], colors[i] + 'o', markersize=10)
        plots.append(p)

    plt.legend(plots, ["CCWP", "DD", "GDD"], loc=2)
    plt.grid()
    plt.xlabel('Seed set size')
    plt.ylabel('Influence spread')
    plt.title('HepNEPT data. p = [.01, .02, .04, .08]')
    fig.set_size_inches(18.5,10.5)

    fig.savefig(filename, dpi=fig.dpi)
    # plt.show()


if __name__ == "__main__":
    print "Plotting results..."
    DD_length_to_coverage = json.load(open("plotdata/plotDirectDDforDirect1.txt"))
    CCWP_length_to_coverage = json.load(open("plotdata/plotDirectCCWPforDirect1_v2.txt"))
    GDD_length_to_coverage = json.load(open("plotdata/plotDirectGDDforDirect1.txt"))
    [DDx, DDy] = zip(*DD_length_to_coverage)
    [CCWPx, CCWPy] = zip(*CCWP_length_to_coverage)
    [GDDx, GDDy] = zip(*GDD_length_to_coverage)

    visualiseResults([CCWPx, DDx, GDDx], [CCWPy, DDy, GDDy], "direct1_v3.png")

    # print "Plotting timing..."
    # CCWP_k = []
    # CCWP_time = []
    # with open("plotdata/timeDirectCCWPforDirect3.txt") as fp:
    #     for line in fp:
    #         data = line.split()
    #         CCWP_k.append(int(data[0]))
    #         CCWP_time.append(float(data[1]))
    # DD_k = []
    # DD_time = []
    # with open("plotdata/timeDirectDDforDirect3.txt") as fp:
    #     for line in fp:
    #         data = line.split()
    #         DD_k.append(int(data[0]))
    #         DD_time.append(float(data[1]))
    #
    # visualiseTime([CCWP_k, DD_k], [CCWP_time, DD_time], "timeDirect3.png")

