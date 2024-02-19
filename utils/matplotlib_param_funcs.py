import matplotlib.pyplot as plt

def set_matplotlib_params():
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'STIXGeneral'

    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['xtick.minor.size'] = 6
    plt.rcParams['ytick.minor.size'] = 6

    plt.rcParams['hatch.linewidth'] = 0.5

    plt.rcParams["figure.figsize"] = [12,8]
    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.titlesize"] = "medium"

def reset_rcParams():
    plt.rcParams.update(plt.rcParamsDefault)