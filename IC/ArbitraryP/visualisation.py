from __future__ import division
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

colors = ['b', 'r', 'g', 'm', 'k', 'c', 'y']
marks = ["o", "s", "^", "v", "*", "h", "D", "x"]

def visualiseTime(x_lst, y_lst, filename="tempTime.png", model="Model"):
    # matplotlib.rcParams.update({'font.size': 32})
    fig = plt.figure()
    ax = fig.gca()

    # ax.set_yscale('log')

    plots = []
    for i in range(len(x_lst)):
        plt.plot(x_lst[i], y_lst[i], colors[i] + '--')
        p, = plt.plot(x_lst[i], y_lst[i], colors[i] + marks[i], markersize=10) # plot dots
        plots.append(p)

    # plt.legend(plots, ["Preprocessing"], loc=2)
    plt.grid()
    plt.xlabel("k")
    plt.ylabel('Time (secs)')
    fig.suptitle('HepNEPT -- %s' %(model))
    plt.title("T = 2000")
    fig.set_size_inches(18.5,10.5)

    fig.savefig(filename, dpi=fig.dpi)
    # plt.show()

def visualiseResults(x_lst, y_lst, filename="tempResults.png", model="Model", dataset="Dataset"):
    matplotlib.rcParams.update({'font.size': 24})
    fig = plt.figure(figsize=(18, 10))
    ax = fig.gca()

    # length of axes
    # max_x = max(x_lst, key=lambda x: x[-1][0])
    # max_y = max(y_lst, key=lambda y: y[-1][0])

    # reduce font of first 4 xtciks
    # ax.set_xticks(map(float, x_lst[0]))
    # for i, tick in enumerate(ax.xaxis.get_major_ticks()):
    #     if i < 4:
    #         tick.label.set_fontsize(10)
    #     else:
    #         break
    # [tick.label.set_fontsize(8) for tick in ax.xaxis.get_major_ticks()]
    # ax.set_yticks([150,250,350,550,750,850,950,1050,1150,1250,1350,1450,1550,1750,1950,2150])
    # xlim = 151
    ax.set_xticks(range(10, 210, 10))

    # ax.set_xscale("log")
    ax.set_yscale("log")


    # legends = ["Harvester", "GDD", "PMIA", "Degree", "Random"]
    # legends = ["Harvester", "HarvesterABM", "HarvesterADR",
    #            "HarvesterGAME", "HarvesterGC", "PMIA"]
    legends = ["Random", "Harvester400", "HarvesterADR", "PMIA"]
    legends = ["Random", "MP", "MP+", "SF", "ADR"]
    legends = map(lambda v: "K%s" %v, range(20, 110, 20))
    legends = map(lambda v: "MPST%s" %v, range(10,110,10))
    # legends = ["MP20","MP50","MP100","MPST20","MPST50","MPST100","Spine20","Spine50","Spine100"]
    # legends = ["MP50","MPST50","Spine50"]
    # legends = ["MP+LP", "MPST+LP"]
    colors = ['b', 'r', 'g', 'm', 'k', 'y', 'c', u'#fe2fb3', u'#abfeaa', u'#cccabc', u'#1111ee']
    marks = ["o", "s", "^", "v", 'x', "<", ">", '8', "<", ">", '8']
    colors = colors[::1]
    marks = marks[::1]
    x_lst.reverse()
    y_lst.reverse()
    legends.reverse()
    colors.reverse()
    marks.reverse()

    # xlim_pos = (xlim-1)//5
    # sort legends accroding to their values
    # [x_lst, y_lst, legends, colors, marks] = zip(*sorted(zip(x_lst, y_lst, legends, colors, marks), key = lambda(x,y,l,c,m): y[-1], reverse=True))

    plots = []
    # print colors
    for i in range(len(x_lst)):
        plt.plot(x_lst[i], y_lst[i], color=colors[i], linewidth=3)
        p, = plt.plot(x_lst[i], y_lst[i], color = colors[i], marker = marks[i], markersize=10)
        plots.append(p)
    # for i in range(len(x_lst)):
    #     plt.plot(x_lst[i], y_lst[i], linewidth=3)
    #     p, = plt.plot(x_lst[i], y_lst[i], markersize=10)
    #     plots.append(p)

    # plt.xlim([9, 200])
    # plt.ylim([770, 820])
    # plt.ylim([100,600])
    # plt.ylim([400,1025])
    # plt.ylim([300,1000])
    # plt.ylim([800,1300])
    # plt.ylim([1500,2200])

    plt.legend(plots, legends, loc=4, prop={'size': 18})
    # plt.grid()
    plt.xlabel('Seed set')
    plt.ylabel('Spread (nodes)')
    # ax = plt.gca()
    # ax.set_xlabel('Number of PW', fontsize = 42)
    # ax.xaxis.set_label_coords(.5, -0.03)
    # ax.set_ylabel('Influence Spread', fontsize = 42)
    # ax.yaxis.set_label_coords(-0.125, 0.5)
    # plt.title('HepNEPT data. p = [.01, .02, .04, .08]')
    plt.title('%s' %(dataset), fontsize = 18)
    # fig.set_size_inches(18.5,10.5)
    # fig.savefig(filename, dpi=fig.dpi)
    plt.show()

def visualiseReverse(x_lst, y_lst, filename="tempResults.png", model="Model", dataset="Dataset"):
    matplotlib.rcParams.update({'font.size': 56})
    fig = plt.figure()
    ax = fig.gca()

    # length of axes
    # max_x = max(x_lst, key=lambda x: x[-1][0])
    # max_y = max(y_lst, key=lambda y: y[-1][0])

    ax.set_xticks([500, 1400])
    ax.set_xticks([900, 1600])
    ax.set_xticks([1700, 2900])

    # ax.set_yticks([100,1000])
    # ax.set_yticks([100,900])
    # ax.set_yticks([100,700])
    # ax.set_yticks([100,800])
    # ax.set_yticks([100,750])
    ax.set_yticks([100, 1700])
    xlim = 2900

    legends = ["Harvester", "GDD", "PMIA", "Degree", "Random"]
    colors = ['b', 'r', 'g', 'm', 'k']
    marks = ["o", "s", "^", "v", 'x']
    x_lst.reverse()
    y_lst.reverse()
    legends.reverse()
    colors.reverse()
    marks.reverse()

    xlim_pos = xlim//100 - 1
    # sort legends accroding to their values
    # [x_lst, y_lst, legends, colors, marks] = zip(*sorted(zip(x_lst, y_lst, legends, colors, marks), key = lambda(x,y,l,c,m): y[xlim_pos]))

    plots = []
    for i in range(len(x_lst)):
        plt.plot(x_lst[i], y_lst[i], colors[i] + '--')
        p, = plt.plot(x_lst[i], y_lst[i], colors[i] + marks[i], markersize=20)
        plots.append(p)
    plt.xlim([1700, xlim])

    # plt.ylim([50, 1050])
    # plt.ylim([50, 900])
    # plt.ylim([0, 700])
    # plt.ylim([0, 800])
    # plt.ylim([0, 770])
    plt.ylim([0, 1700])

    plt.legend(plots, legends, loc=1, prop={'size': 32})
    # plt.grid()
    ax = plt.gca()
    ax.set_xlabel('influence spread', fontsize = 42, color='k')
    ax.xaxis.set_label_coords(.5, -0.03)
    ax.set_ylabel('seed set size', fontsize = 42)
    ax.yaxis.set_label_coords(-0.125, 0.5)
    # plt.title('HepNEPT data. p = [.01, .02, .04, .08]')
    plt.title('%s' %(dataset), fontsize = 36)
    fig.set_size_inches(18.5,10.5)
    fig.savefig(filename, dpi=fig.dpi)
    # plt.show()

def visualiseSteps(x_lst, y_lst, filename="tempResults.png", model="Model", dataset="Dataset"):
    matplotlib.rcParams.update({'font.size': 56})
    fig = plt.figure()
    ax = fig.gca()

    # length of axes
    # max_x = max(x_lst, key=lambda x: x[-1][0])
    # max_y = max(y_lst, key=lambda y: y[-1][0])

    ax.set_xticks([500, 1400])
    ax.set_xticks([900, 1600])
    ax.set_xticks([1700,2900])
    ax.set_yticks([5, 20])
    xlim = 2900

    legends = ["Harvester", "GDD", "PMIA", "Degree", "Random"]
    colors = ['b', 'r', 'g', 'm', 'k']
    marks = ["o", "s", "^", "v", 'x']
    x_lst.reverse()
    y_lst.reverse()
    legends.reverse()
    colors.reverse()
    marks.reverse()

    xlim_pos = xlim//5
    # sort legends accroding to their values
    # [x_lst, y_lst, legends, colors, marks] = zip(*sorted(zip(x_lst, y_lst, legends, colors, marks), key = lambda(x,y,l,c,m): y[-1]))

    plots = []
    for i in range(len(x_lst)):
        plt.plot(x_lst[i], y_lst[i], colors[i] + '--')
        p, = plt.plot(x_lst[i], y_lst[i], colors[i] + marks[i], markersize=20)
        plots.append(p)
    plt.xlim([500, xlim])
    plt.xlim([900, xlim])
    plt.xlim([1700, xlim])
    plt.ylim([0, 25])

    plt.legend(plots, legends, loc=4, prop={'size': 24})
    # plt.grid()
    ax = plt.gca()
    ax.set_xlabel('influence spread', fontsize = 42)
    ax.xaxis.set_label_coords(.5, -0.03)
    ax.set_ylabel('number of steps', fontsize = 42)
    ax.yaxis.set_label_coords(-0.09, 0.5)
    # plt.title('HepNEPT data. p = [.01, .02, .04, .08]')
    plt.title('%s' %(dataset), fontsize = 36)
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
    # fig.savefig(filename, dpi=fig.dpi)
    #
    plt.show()

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

def plotWeightedScore (k, Trange, xticks, filename="plots/UpdateFunctions_.png",
                       model = "Model", title_dataset="Dataset"):
    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 2

    x = np.arange(0, 4*len(Trange), 4)
    y = Trange

    # plt.xlabel('R')
    plt.ylabel('running time (in sec)', fontsize = 24)

    plt.title('%s' %(title_dataset), fontsize = 36)

    plt.grid(axis="y", linestyle='-', linewidth=2)



    rects1 = ax.bar(x, y, width = 3, bottom=0, log=True, color = "k")
    plt.xticks(x + 1.5, xticks, fontsize=17)

    # add text label at the top of each bar
    # solution found at http://matplotlib.org/examples/api/barchart_demo.html
    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.4f'%height,
                    ha='center', va='bottom')

    autolabel(rects1)
    for i, rect in enumerate(rects1):
        rect.set_color(colors[i])
    # fig.set_size_inches(15.5,11.5)
    fig.savefig(filename, dpi=fig.dpi)
    # plt.show()

def read_spread (spread_filename):
    x_lst = []
    y_lst = []
    with open(spread_filename) as fp:
        for line in fp:
            [x,y] = line.split()
            x_lst.append(x)
            y_lst.append(y)
    return x_lst, y_lst

def read_reverse (reverse_filename):
    x_lst = []
    y_lst = []
    with open(reverse_filename) as fp:
        for line in fp:
            [x,y] = line.split()
            x_lst.append(float(x))
            y_lst.append(float(y))
    return x_lst, y_lst

def read_time (time_filename):
    y_lst = []
    with open(time_filename) as fp:
        for line in fp:
            y = float(line)
            y_lst.append(y)
    return y_lst

def percentage_difference (y_lst1, y_lst2, length):
    avg = 0
    for i in range(length):
        avg += (y_lst1[i] - y_lst2[i])/(y_lst2[i]*length)
    return round(avg, 3)*100

if __name__ == "__main__":
    ############################### Influence Maximization problem ##################################
    INPUT_FOLDER = "Data4InfMax/Spread/"
    # OUTPUT_FOLDER = "Data4InfMax/Spread_Plots/"
    OUTPUT_FOLDER = "./"
    dataset = "hep"
    directed="U"
    model = "MultiValency"
    if dataset == "hep":
        title_dataset = "NetHEPT "
    elif dataset == "gnu09":
        title_dataset = "Gnutella "
    elif dataset == "fb":
        title_dataset = "Facebook "

    x_lst = []
    y_lst = []
    MC = 100

    # Methods = ["Random", "Top", "Top3", "MPST", "ADR"]
    # Methods = Methods[:10]
    # for met in Methods:
    #     with open("Sparsified_results/Redistributed_%s_MC%s_avg.txt" %(met, MC)) as fp:
    #         obj_random = json.load(fp)
    #         print obj_random
    #         random_per = []
    #         random_obj = []
    #         for (percentage, obj) in sorted(obj_random.items(), key = lambda (k, _): float(k)):
    #             random_per.append(percentage)
    #             random_obj.append(obj)
    #         x_lst.append(random_per)
    #         y_lst.append(random_obj)

    x_lst = []
    y_lst = []
    for i in range(1,11):
        with open("Flickr2/spread/K%s.txt" %(i*10)) as f:
            x = []
            y = []
            for line in f:
                d = map(float, line.split())
                x.append(d[0])
                y.append(d[1])
            x_lst.append(x)
            y_lst.append(y)
    visualiseResults(x_lst, y_lst, "Sparsified_results/Memory.png", model, "Flickr: n=5K; m=800K")

    # x_lst = []
    # y_lst = []
    # for s in ["MP", "MPST"]:
    #     with open("LP/mae01%s.dat" %s) as f:
    #         x = []
    #         y = []
    #         for line in f:
    #             k, spread = map(float, line.split(','))
    #             x.append(k)
    #             y.append(spread)
    #         x_lst.append(x)
    #         y_lst.append(y)
    # visualiseResults(x_lst, y_lst, "Sparsified_results/Spread.png", model, "Flickr: n=5K; m=800K")

    console = []


    # score_i = 1
    # Harvester_R, Harvester_spread = read_spread("./HarvesterPW_results/3/Spread/Harvester_adj_Hep_Spread.txt")
    # x_lst.append(Harvester_R); y_lst.append(Harvester_spread)
    #
    # Harvester400_R, Harvester400_spread = read_spread("./HarvesterPW_results/3/Spread/Harvester400_adj_Hep_Spread.txt")
    # x_lst.append(Harvester400_R); y_lst.append(Harvester400_spread)

    # HarvesterABM_R, HarvesterABM_spread = read_spread("./HarvesterPW_results/2/Spread/HarvesterABM_adj_Hep_Spread.txt")
    # x_lst.append(HarvesterABM_R); y_lst.append(HarvesterABM_spread)

    # HarvesterADR_R, HarvesterADR_spread = read_spread("./HarvesterPW_results/3/Spread/HarvesterADR_adj_Hep_Spread.txt")
    # x_lst.append(HarvesterADR_R); y_lst.append(HarvesterADR_spread)

    # HarvesterGAME_R, HarvesterGAME_spread= read_spread("./HarvesterPW_results/2/Spread/HarvesterGAME_adj_Hep_Spread.txt")
    # x_lst.append(HarvesterGAME_R); y_lst.append(HarvesterGAME_spread)
    #
    # HarvesterGC_R, HarvesterGC_spread = read_spread("./HarvesterPW_results/2/Spread/HarvesterGC_adj_Hep_Spread.txt")
    # x_lst.append(HarvesterGC_R); y_lst.append(HarvesterGC_spread)

    # PMIA_R, PMIA_spread = read_spread("./HarvesterPW_results/3/Spread/PMIA_adj_Hep_Spread.txt")
    # x_lst.append(PMIA_R); y_lst.append(PMIA_spread)



    # visualiseResults(x_lst, y_lst, "./HarvesterPW_results/3/HEP_Spread_with_PWs.pdf", model, "HEP: Spread vs. |PWs|. k = 100")

    ###################################### Seed minimization problem ####################################
    # INPUT_FOLDER = "Data4InfMax/Reverse/"
    # OUTPUT_FOLDER = "Data4InfMax/Reverse_Plots/"
    # dataset = "hep"
    # model = "Categories"
    # if dataset == "hep":
    #     title_dataset = "NetHEPT"
    # elif dataset == "gnu09":
    #     title_dataset = "Gnutella"
    # elif dataset == "fb":
    #     title_dataset = "Facebook"
    #
    # x_lst = []
    # y_lst = []
    # CCWPx, CCWPy = read_reverse(INPUT_FOLDER + "Reverse_CCWP_%s_%s.txt" %(dataset, model));
    # CCWPx = map(float, CCWPx)
    # CCWPy = map(float, CCWPy)
    # x_lst.append(CCWPx); y_lst.append(CCWPy)
    # GDDx, GDDy = read_reverse(INPUT_FOLDER + "Reverse_GDD_%s_%s.txt" %(dataset, model));
    # GDDx = map(float, GDDx)
    # GDDy = map(float, GDDy)
    # x_lst.append(GDDx); y_lst.append(GDDy)
    # PMIAx, PMIAy = read_reverse(INPUT_FOLDER + "Reverse_PMIA_%s_%s.txt" %(dataset, model));
    # PMIAx = map(float, PMIAx)
    # PMIAy = map(float, PMIAy)
    # x_lst.append(PMIAx); y_lst.append(PMIAy)
    # Degreex, Degreey = read_reverse(INPUT_FOLDER + "Reverse_Degree_%s_%s.txt" %(dataset, model));
    # Degreex = map(float, Degreex)
    # Degreey = map(float, Degreey)
    # x_lst.append(Degreex); y_lst.append(Degreey)
    # RDMx, RDMy = read_reverse(INPUT_FOLDER + "Reverse_RDM_%s_%s.txt" %(dataset, model));
    # RDMx = map(float, RDMx)
    # RDMy = map(float, RDMy)
    # x_lst.append(RDMx); y_lst.append(RDMy)
    #
    # visualiseReverse(x_lst, y_lst, OUTPUT_FOLDER + "reverse_%s_%s.pdf" %(dataset, model), model, title_dataset)

    # console = []

    ###################################### Steps plots ####################################
    # INPUT_FOLDER = "Data4InfMax/Steps/"
    # OUTPUT_FOLDER = "Data4InfMax/Steps_Plots/"
    # dataset = "hep"
    # model = "Categories"
    # if dataset == "hep":
    #     title_dataset = "NetHEPT"
    # elif dataset == "gnu09":
    #     title_dataset = "Gnutella"
    # elif dataset == "fb":
    #     title_dataset = "Facebook"
    #
    # x_lst = []
    # y_lst = []
    # CCWPx, CCWPy = read_reverse(INPUT_FOLDER + "Steps_CCWP_%s_%s.txt" %(dataset, model));
    # CCWPx = map(float, CCWPx)
    # CCWPy = map(float, CCWPy)
    # x_lst.append(CCWPx); y_lst.append(CCWPy)
    # GDDx, GDDy = read_reverse(INPUT_FOLDER + "Steps_GDD_%s_%s.txt" %(dataset, model));
    # GDDx = map(float, GDDx)
    # GDDy = map(float, GDDy)
    # x_lst.append(GDDx); y_lst.append(GDDy)
    # PMIAx, PMIAy = read_reverse(INPUT_FOLDER + "Steps_PMIA_%s_%s.txt" %(dataset, model));
    # PMIAx = map(float, PMIAx)
    # PMIAy = map(float, PMIAy)
    # x_lst.append(PMIAx); y_lst.append(PMIAy)
    # Degreex, Degreey = read_reverse(INPUT_FOLDER + "Steps_Degree_%s_%s.txt" %(dataset, model));
    # Degreex = map(float, Degreex)
    # Degreey = map(float, Degreey)
    # x_lst.append(Degreex); y_lst.append(Degreey)
    # RDMx, RDMy = read_reverse(INPUT_FOLDER + "Steps_RDM_%s_%s.txt" %(dataset, model));
    # RDMx = map(float, RDMx)
    # RDMy = map(float, RDMy)
    # x_lst.append(RDMx); y_lst.append(RDMy)
    #
    # visualiseSteps(x_lst, y_lst, OUTPUT_FOLDER + "steps_%s_%s.pdf" %(dataset, model), model, title_dataset)

    # console = []

    ########################################### Time plots ###########################################
    # INPUT_FOLDER = "Data4InfMax/Time/"
    # OUTPUT_FOLDER = "Data4InfMax/Time_Plots/"
    # dataset = "hep"
    # model = "Random"
    # if dataset == "hep":
    #     title_dataset = "NetHEPT"
    # elif dataset == "gnu09":
    #     title_dataset = "Gnutella"
    # elif dataset == "fb":
    #     title_dataset = "Facebook"
    #
    # y_lst = []
    # step = 0
    # xticks = []
    # CCWPy = read_time(INPUT_FOLDER + "Time_CCWP_%s_%s.txt" %(dataset, model));
    # y_lst.append(CCWPy[0])
    # xticks.append("Harvester")
    # GDDy = read_time(INPUT_FOLDER + "Time_GDD_%s_%s.txt" %(dataset, model));
    # y_lst.append(GDDy[step])
    # xticks.append("GDD")
    # PMIAy = read_time(INPUT_FOLDER + "Time_PMIA_%s_%s.txt" %(dataset, model));
    # y_lst.append(PMIAy[0])
    # xticks.append("PMIA")
    # Degreey = read_time(INPUT_FOLDER + "Time_Degree_%s_%s.txt" %(dataset, model));
    # y_lst.append(Degreey[step])
    # xticks.append("Degree")
    # RDMy = read_time(INPUT_FOLDER + "Time_RDM_%s_%s.txt" %(dataset, model));
    # y_lst.append(RDMy[step])
    # xticks.append("Random")
    # NGRy = read_time(INPUT_FOLDER + "Time_NGR_%s_%s.txt" %(dataset, model));
    # y_lst.append(NGRy[0])
    # xticks.append("NGIC")
    #
    # xlim = (step-1)*5 + 1
    # plotWeightedScore(xlim, y_lst, xticks, OUTPUT_FOLDER + "time_%s_%s.pdf" %(dataset, model), model, title_dataset)



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

    # print "Plotting k vs R..."
    # model = "MultiValency"
    # with open("plotdata/kvsR_%s.txt" %model) as fp:
    #     T = int(fp.readline())
    #     R2k = json.loads(fp.readline())
    # [Rrange, krange] = zip(*R2k)
    #
    # plotkvsR(krange, Rrange, T, "plots/kvsR_%s.png" %model, model)

    # print "Plotting resutls for different update functions..."
    # model = "MultiValency"
    # updatef = [r'$\frac{1}{|CC|}$',
    #            '1',
    #            r'$|CC|$',
    #            r'$\frac{1}{\sqrt{|CC|}}$',
    #            r'$\frac{1}{|CC|^2}$',
    #            r'$\frac{L}{|CC|}$',
    #             r'$\frac{1}{|CC|\cdot L}$',
    #             r'$\frac{1 - L}{|CC| - L + i - |CC|\cdot i}$',
    #             r'$\frac{|CC|\cdot (i - L) + 1 - i}{|CC|\cdot (1 - L)}$',
    #             r'$\frac{1}{QN}$']
    # T = 1000
    # with open("plotdata/T1000_MultiValency.txt") as fp:
    #     xticks = []
    #     krange = []
    #     for line in fp:
    #         [idx, ky] = map(float, line.split())
    #         xticks.append(updatef[int(idx) - 1])
    #         krange.append(ky)
    # plotWeightedScore(T, krange, xticks, "plots/UpdateFunctionsReverse_%s.png" %model, model)

    # print "Plotting time for k vs R..."
    # model = "MultiValency"
    # with open("plotdata/timekvsR_MultiValency.txt") as fp:
    #     [p_timex, p_timey] = zip(*json.loads(fp.readline()))
    #     [a_timex, a_timey] = zip(*json.loads(fp.readline()))
    #     [s_timex, s_timey] = zip(*json.loads(fp.readline()))
    #     [t_timex, t_timey] = zip(*json.loads(fp.readline()))
    #
    # x_lst = [p_timex, s_timex, a_timex, t_timex]
    # y_lst = [p_timey, s_timey, a_timey, t_timey]
    # visualiseTime(x_lst, y_lst, "plots/timekvsR_%s.png" %model, model)

    # print "Plotting T vs steps..."
    # model = "Categories"
    # with open("plotdata/reverseCCWPrmaxL_%s.txt" %model) as fp:
    #     CCWPx = []
    #     CCWPy = []
    #     for line in fp:
    #         [r, x, y] = map(int, line.split())
    #         CCWPx.append(x)
    #         CCWPy.append(y)
    #
    # with open("plotdata/reversePMIA_%s.txt" %model) as fp:
    #     PMIAx = []
    #     PMIAy = []
    #     for line in fp:
    #         [x, y] = map(int, line.split())
    #         PMIAx.append(x)
    #         PMIAy.append(y)
    #
    # with open("plotdata/reverseGDD_%s.txt" %model) as fp:
    #     GDDx = []
    #     GDDy = []
    #     for line in fp:
    #         [x, y] = map(int, line.split())
    #         GDDx.append(x)
    #         GDDy.append(y)
    #
    # visualiseResults([CCWPx, PMIAx, GDDx], [CCWPy, PMIAy, GDDy], "plots/reverse_%s.png" %model, model)

    # print "Plotting Time vs k..."
    # with open("plotdata/time2k_MultiValency.txt") as fp:
    #     k2time = []
    #     for line in fp:
    #         k2time.append(map(float, line.split()))
    # k2time.sort()
    # [x, y] = zip(*k2time)
    #
    # visualiseTime([x], [y], "plots/timevsk_MultiValency.png", "MultiValency")


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

    console = []

