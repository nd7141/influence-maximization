from __future__ import division
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy


if __name__ == "__main__":
    print "Plotting results..."
    DD_length_to_coverage = json.load(open("plotDDfor2.txt"))
    CCWP_length_to_coverage = json.load(open("plotCCWPfor2.txt"))
    [DDx, DDy] = zip(*DD_length_to_coverage)
    [CCWPx, CCWPy] = zip(*CCWP_length_to_coverage)

    matplotlib.rcParams.update({'font.size': 32})

    fig = plt.figure()
    ax = fig.gca()
    # length of axes
    max_x = max(CCWP_length_to_coverage[-1][0], DD_length_to_coverage[-1][0])
    max_y = max(CCWP_length_to_coverage[-1][1], DD_length_to_coverage[-1][1])

    # ax.set_xticks(numpy.arange(0, max_x, max_x//10))
    ax.set_yticks(numpy.arange(0, max_y, max_y//20))

    plt.plot(CCWPx, CCWPy, 'b--') # plot dashed line
    p1, = plt.plot(CCWPx, CCWPy, 'bo', markersize=10) # plot dots

    plt.plot(DDx, DDy, 'r--')
    p2, = plt.plot(DDx, DDy, 'ro', markersize=10)

    plt.legend([p1, p2], ["CCWP", "DD"], loc=2)
    plt.grid()
    plt.xlabel('Seed set size')
    plt.ylabel('Influence spread')
    plt.title('HepNEPT data. p = [.01, .02, .04, .08]')
    fig.set_size_inches(18.5,10.5)
    fig.savefig('temp.png', dpi=fig.dpi)
    # plt.show()
