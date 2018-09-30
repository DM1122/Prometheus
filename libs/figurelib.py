import os
import matplotlib
from matplotlib import pyplot as plt
import skopt

def save_fig(fig, name, date, id):
    '''
    Saves the specified figure to disk.

    Args:
      fig : matplotlib figure
      name : name of figure
      date : formated timestamp
      id : script name
    '''

    save_dir = './plots/'+id+'/'+date+'/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        fig.savefig(save_dir+name)
        print('Saved '+name+' figure to disk')
    except:
        print('Warning: figurelib.py was unable to save figure '+name)


def plot_opt(search, dims):
    '''
    Plots skopt hyperam optimization results.

    Args:
      search : skopt search
      dims (list) : dimension names
    '''

    matplotlib.style.use('classic')

    fig1 = skopt.plots.plot_convergence(search)
    fig2, ax = skopt.plots.plot_objective(result=search, dimension_names=dims)
    fig3, ax = skopt.plots.plot_evaluations(result=search, dimension_names=dims)

    fig2.tight_layout()
    fig3.tight_layout()

    return fig1, fig2, fig3


def plot_opt_single(search, dim):
    '''
    Plots skopt hyperam optimization results for single dim

    Args:
      search : skopt search
      dims (str) : dimension name
    '''

    matplotlib.style.use('classic')

    fig1 = skopt.plots.plot_convergence(search)
    fig2, ax = skopt.plots.plot_objective(result=search, dimension_names=[dim,dim])
    fig3, ax = skopt.plots.plot_evaluations(result=search, dimension_names=[dim,dim])

    fig2.tight_layout()
    fig3.tight_layout()

    return fig1, fig2, fig3


def plot_pred(lbl, out, xlbl, ylbl):
    
    matplotlib.style.use('classic')

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    
    # Plot and compare two signals
    plt.plot(lbl, label='Label')
    plt.plot(out, label='Output')

    fig.tight_layout()
    plt.legend()

    return fig


def plot_pred_warmup(lbl, out, warmup, xlbl, ylbl):
    
    matplotlib.style.use('classic')

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    
    # Plot and compare two signals
    plt.plot(lbl, label='Label')
    plt.plot(out, label='Output')
    plt.axvspan(0, warmup, facecolor='black', alpha=0.15)

    fig.tight_layout()
    plt.legend()

    return fig
