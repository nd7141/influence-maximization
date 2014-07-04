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

    plt.legend(plots, ["CCWP", "GDD", "PMIA"], loc=4)
    plt.grid()
    plt.xlabel('Seed set size')
    plt.ylabel('Influence spread')
    # plt.title('HepNEPT data. p = [.01, .02, .04, .08]')
    plt.title('HepNEPT -- Categories')
    fig.set_size_inches(18.5,10.5)
    fig.savefig(filename, dpi=fig.dpi)
    # plt.show()

def plotCCsSizeDistribution (histogram, bluedots, T, filename="plots/CCs_sizes.png",
                             L = None, TotalCCs = None, model="Model",
                             xlog = True, ylog = True):
    '''
     histogram: [(500, 1), (250, 1), ..., (6, 12), ..., (1, 1500)] -- [(size of CC, number of CCs of that size),...]
     bluedots: number of qualified sizes
     T -- Targeted spread
     L -- minimum number of CCs that achieves T
     TotalCCs -- total number of CCs
     model -- name of Ep model (eg. MultiValency, Random, Uniform, WC, Categories)
    '''

    [x,y] = zip(*histogram)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    p1 = plt.scatter(x[:bluedots],y[:bluedots], s=80, c="b")
    p2 = plt.scatter(x[bluedots:],y[bluedots:], s=80, c="r")

    x1,x2,y1,y2 = plt.axis()
    plt.axis((1, x2, 10**(-1), y2))
    plt.legend([p1, p2], ["Qualified", "Non-Qualified"], loc=1)
    plt.xlabel('Size of Connected Components')
    plt.ylabel('Number of Connected Components')
    if L != None and TotalCCs != None:
        plt.title('L = %s out of %s CCs' %(L, TotalCCs))
    plt.text( 10**2, 10**4, '')
    # plt.title('HepNEPT data. p <= .1')
    fig.suptitle("HepNEPT -- %s. T = %s" %(model, T))
    # fig.set_size_inches(18.5,10.5)
    fig.savefig(filename, dpi=fig.dpi)

    # plt.show()

def plotLvsT (Lrange, Trange, TotalCCs, filename = "plots/LvsT_.png",
              model="Model"):

    x = Trange
    y = Lrange

    fig = plt.figure()
    ax = fig.gca()

    plt.plot(x, y, 'r' + '--', linewidth=2)
    p, = plt.plot(x, y, 'r' + 'o', markersize=6)

    plt.xlabel('T')
    plt.ylabel('L')

    fig.suptitle('HepNEPT -- %s' %(model))
    plt.title('Total CCs: %s' %TotalCCs)

    fig.savefig(filename, dpi=fig.dpi)
    # plt.show()

def plotTvsR (Trange, Rrange, k, filename = "plots/TvsR_.png",
              model="Model"):

    x = Rrange
    y = Trange

    fig = plt.figure()
    ax = fig.gca()

    plt.plot(x, y, 'r' + '--', linewidth=2)
    p, = plt.plot(x, y, 'r' + 'o', markersize=6)

    plt.xlabel('R')
    plt.ylabel('T')

    fig.suptitle('HepNEPT -- %s' %(model))
    plt.title('k = %s' %k)

    fig.savefig(filename, dpi=fig.dpi)
    # plt.show()

def plotkvsR (krange, Rrange, T, filename = "plots/kvsR_.png",
              model="Model"):

    x = Rrange
    y = krange

    fig = plt.figure()
    ax = fig.gca()

    plt.plot(x, y, 'r' + '--', linewidth=2)
    p, = plt.plot(x, y, 'r' + 'o', markersize=6)

    plt.xlabel('R')
    plt.ylabel('k')

    fig.suptitle('HepNEPT -- %s' %(model))
    plt.title('T = %s' %T)

    fig.savefig(filename, dpi=fig.dpi)
    # plt.show()

if __name__ == "__main__":
    # print "Plotting results..."
    # # DD_length_to_coverage = json.load(open("plotdata/plotDirectDDforDirect5.txt"))
    # CCWP_length_to_coverage = json.load(open("plotdata/plotDirectCCWPforDirect6.txt"))
    # GDD_length_to_coverage = json.load(open("plotdata/plotDirectGDDforDirect6.txt"))
    # PMIA_length_to_coverage = json.load(open("plotdata/plotDirectPMIAforDirect6.txt"))
    # # [DDx, DDy] = zip(*DD_length_to_coverage)
    # [CCWPx, CCWPy] = zip(*CCWP_length_to_coverage)
    # [GDDx, GDDy] = zip(*GDD_length_to_coverage)
    # [PMIAx, PMIAy] = zip(*PMIA_length_to_coverage)
    #
    # visualiseResults([CCWPx, GDDx, PMIAx], [CCWPy, GDDy, PMIAy], "plots/direct6.png")

    # print "Plotting CCs sizes distribution..."
    # with open("plotdata/CCs_sizes_Categories1.txt") as fp:
    #     bluedots = int(fp.readline())
    #     T = int(fp.readline())
    #     L = int(fp.readline())
    #     TotalCCs = int(fp.readline())
    #     histogram = json.loads(fp.readline())
    #
    # model = "Categories"
    # plotCCsSizeDistribution(histogram, bluedots, T, "plots/CCs_sizes_categories.png", L, TotalCCs, model, xlog=False)

    # print "Plotting L vs T..."
    # model = "Random"
    # with open("plotdata/LvsT_%s.txt" %model) as fp:
    #     Trange = json.loads(fp.readline())
    #     Lrange = json.loads(fp.readline())
    #     TotalCCs = int(fp.readline())
    #
    # plotLvsT(Lrange, Trange, TotalCCs, "plots/LvsT_%s.png" %model, model)

    # print "Plotting T vs R..."
    # model = "Categories"
    # with open("plotdata/TvsR_%s.txt" %model) as fp:
    #     k = int(fp.readline())
    #     R2T = json.loads(fp.readline())
    # [Rrange, Trange] = zip(*R2T)
    #
    # plotTvsR(Trange, Rrange, k, "plots/TvsR_%s.png" %model, model)

    print "Plotting k vs R..."
    model = "Categories"
    with open("plotdata/kvsR_%s0.txt" %model) as fp:
        T = int(fp.readline())
        R2k = json.loads(fp.readline())
    [Rrange, krange] = zip(*R2k)

    plotkvsR(krange, Rrange, T, "plots/kvsR_%s.png" %model, model)

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

