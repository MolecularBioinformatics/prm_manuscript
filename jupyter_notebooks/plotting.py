import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def hist(pr, bins=10, save=False, filename=None,
         figsize=(10, 8), xlabel='Predicted change (log2)',
         ylabel='No. of metabolites', xlim=None, color=None, ylim=None):
    '''
    Plot histogram of predicted changes

    Args:

    pr (pandas.Series): predicted changes
    bins (int): number of bins
    save (bool): save figure
    filename (str): filename
    figsize (tuple): figure size
    xlabel (str): x-axis label
    ylabel (str): y-axis label
    xlim (tuple): x-axis limits
    color (str): color
    ylim (tuple): y-axis limits

    Returns:

    matplotlib.pyplot.figure
    '''
    fig, ax = plt.subplots(figsize=figsize)
    if color != None:
        sns.histplot(data=pr, bins=bins, ax=ax, color=color)
    else:
        sns.histplot(data=pr, bins=bins, ax=ax)
    ax.set_yscale('log')
    ax.axvline(0.0, linewidth=2.0, color='k', ls='--')
    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)
    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)
    plt.tight_layout()
    if save == True:
        return fig.savefig(filename, dpi=300)
    else:
        return plt.show()


def parity(data, xcolumn, ycolumn, figtitle=None, save=False,
           filename=None, lb=0.0, ub=0.5, nb=5, cc=False,
           show_names=False, show_percentage=False,
           ylim=(-3.0, 3.0), xlim=(-0.25, 0.25), figsize=(10, 8),
           l_p=[(0.05, 0.95), (0.55, 0.95), (0.05, 0.25), (0.55, 0.25)],
           l_n=[(0.05, 0.9), (0.55, 0.9), (0.05, 0.2), (0.55, 0.2)],
           n=[2, 2, 2, 2], fsize=15, cbar_label='probability score ($\Gamma$)',
           cbar_cmap='Reds', xlabel='Predicted change (log2)', alpha=0.5,
           ylabel='Measured change (log2)', color='C0', cc_column='control_coeff',
           edgecolor='k', **kwargs):
    '''
    Plot parity plot

    Args:

    data (pandas.DataFrame): data
    xcolumn (str): column name for x-axis
    ycolumn (str): column name for y-axis
    figtitle (str): figure title
    save (bool): save figure
    filename (str): filename
    lb (float): lower bound for colorbar
    ub (float): upper bound for colorbar
    nb (int): number of bins for colorbar
    cc (bool): colorbar
    show_names (bool): show metabolite names
    show_percentage (bool): show percentage
    ylim (tuple): y-axis limits
    xlim (tuple): x-axis limits
    figsize (tuple): figure size
    l_p (list): list of tuples for percentage locations
    l_n (list): list of tuples for metabolite names locations
    n (list): list of number of metabolites to show
    fsize (int): font size
    cbar_label (str): colorbar label
    cbar_cmap (str): colorbar colormap
    xlabel (str): x-axis label
    alpha (float): alpha
    ylabel (str): y-axis label
    color (str): color
    cc_column (str): column name for colorbar
    edgecolor (str): edgecolor

    Returns:

    matplotlib.pyplot.figure
    '''
    fig, ax = plt.subplots(figsize=figsize)
    if cc == True:
        sc = ax.scatter(x=np.log2(data[ycolumn]),
                        y=np.log2(data[xcolumn]),
                        c=data[cc_column],
                        cmap=cbar_cmap,
                        edgecolors=edgecolor,
                        # vmin=lb, vmax=ub,
                        alpha=alpha)

        plt.colorbar(sc, label=cbar_label, boundaries=np.linspace(lb, ub, nb))
    else:
        sc = ax.scatter(x=np.log2(data[ycolumn]),
                        y=np.log2(data[xcolumn]),
                        alpha=alpha, color=color, edgecolors=edgecolor)

    ax.axhline(0.0, ls="--", c="k")
    ax.axvline(0.0, ls="--", c="k")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    data1 = data.copy()
    if show_percentage == True:
        add_percentage(data1, colname=xcolumn, colname2=ycolumn, ax=ax,
                       show_sc_percentages=False,
                       l=l_p[0], l2=l_p[1], l3=l_p[2], l4=l_p[3], fsize=1.2*fsize)
    if show_names == True:
        add_metabolite_names1(data1, colname=xcolumn, colname2=ycolumn,
                              l=l_n[0], l2=l_n[1], l3=l_n[2], l4=l_n[3],
                              n=n[0], n2=n[1], n3=n[2], n4=n[3], ax=ax, fsize=fsize)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(figtitle)
    plt.tight_layout()
    if save == True:
        return fig.savefig(filename, dpi=300)
    else:
        return plt.show()


def calculate_percentage(df, colname, colname2):
    '''
    Calculate percentage of metabolites in each quadrant

    Args:

    df (pandas.DataFrame): data
    colname (str): column name for x-axis
    colname2 (str): column name for y-axis

    Returns:

    tuple: percentage of metabolites in each quadrant
    '''
    df = df.dropna(axis=0)
    UR = 100 * len(df.loc[(np.log2(df[colname]) > 0.0) &
                          (np.log2(df[colname2]) > 0.0)])/len(df)
    UL = 100 * len(df.loc[(np.log2(df[colname]) > 0.0) &
                          (np.log2(df[colname2]) < 0.0)])/len(df)
    LL = 100 * len(df.loc[(np.log2(df[colname]) < 0.0) &
                          (np.log2(df[colname2]) < 0.0)])/len(df)
    LR = 100 * len(df.loc[(np.log2(df[colname]) < 0.0) &
                          (np.log2(df[colname2]) > 0.0)])/len(df)
    return UL, UR, LL, LR


def calculate_sc_percentage(df, compartment, colname, colname2):
    '''
    Calculate percentage of metabolites in each quadrant for each compartment

    Args:

    df (pandas.DataFrame): data
    compartment (str): compartment
    colname (str): column name for x-axis
    colname2 (str): column name for y-axis

    Returns:

    tuple: percentage of metabolites in each quadrant for each compartment
    '''
    df = df.loc[df.compartment == compartment].dropna(axis=0)
    UR = 100 * len(df.loc[(np.log2(df[colname]) > 0.0) &
                          (np.log2(df[colname2]) > 0.0)])/len(df)
    UL = 100 * len(df.loc[(np.log2(df[colname]) > 0.0) &
                          (np.log2(df[colname2]) < 0.0)])/len(df)
    LL = 100 * len(df.loc[(np.log2(df[colname]) < 0.0) &
                          (np.log2(df[colname2]) < 0.0)])/len(df)
    LR = 100 * len(df.loc[(np.log2(df[colname]) < 0.0) &
                          (np.log2(df[colname2]) > 0.0)])/len(df)
    return UL, UR, LL, LR


def add_percentage(df, ax, colname, colname2,
                   l=(0.02, 0.8), l_1=(0.02, 0.75), l2=(0.7, 0.8), l_2=(0.7, 0.75),
                   l3=(0.02, 0.4), l_3=(0.02, 0.35), l4=(0.7, 0.4), l_4=(0.7, 0.35),
                   fsize=12, show_sc_percentages=True, **kwargs):
    '''
    Add percentage of metabolites in each quadrant

    Args:

    df (pandas.DataFrame): data
    ax (matplotlib.pyplot.axis): axis
    colname (str): column name for x-axis
    colname2 (str): column name for y-axis
    l (tuple): location of percentage for upper left quadrant
    l_1 (tuple): location of percentage for upper left quadrant for each compartment
    l2 (tuple): location of percentage for upper right quadrant
    l_2 (tuple): location of percentage for upper right quadrant for each compartment
    l3 (tuple): location of percentage for lower left quadrant
    l_3 (tuple): location of percentage for lower left quadrant for each compartment
    l4 (tuple): location of percentage for lower right quadrant
    l_4 (tuple): location of percentage for lower right quadrant for each compartment
    fsize (int): font size
    show_sc_percentages (bool): show percentage for each compartment
    kwargs (dict): keyword arguments

    Returns:

    matplotlib.pyplot.axis
    '''
    total_percentage = calculate_percentage(
        df=df, colname=colname, colname2=colname2)
    ax.text(l[0], l[1], "{:.2f}%".format(total_percentage[0]),
            transform=ax.transAxes, fontsize=fsize, verticalalignment='top',
            c=kwargs.get('c', 'black'), weight='bold')

    ax.text(l2[0], l2[1], "{:.2f}%".format(total_percentage[1]),
            transform=ax.transAxes, fontsize=fsize, verticalalignment='top',
            c=kwargs.get('c', 'black'), weight='bold')

    ax.text(l3[0], l3[1], "{:.2f}%".format(total_percentage[2]),
            transform=ax.transAxes, fontsize=fsize, verticalalignment='top',
            c=kwargs.get('c', 'black'), weight='bold')

    ax.text(l4[0], l4[1], "{:.2f}%".format(total_percentage[3]),
            transform=ax.transAxes, fontsize=fsize, verticalalignment='top',
            c=kwargs.get('c', 'black'), weight='bold')

    if show_sc_percentages == True:
        sc_percentage_cyto = calculate_sc_percentage(
            df=df, compartment='cytoplasm', colname=colname, colname2=colname2)
        sc_percentage_mito = calculate_sc_percentage(
            df=df, compartment='mitochondrial', colname=colname, colname2=colname2)
        sc_percentage_nuc = calculate_sc_percentage(
            df=df, compartment='nuclear', colname=colname, colname2=colname2)

        ax.text(kwargs.get('l_1', l)[0], kwargs.get('l_1', l)[1],
                "c:{:.2f}%,\nm:{:.2f}%,\nn:{:.2f}%".format(
                    sc_percentage_cyto[0], sc_percentage_mito[0], sc_percentage_nuc[0]),
                transform=ax.transAxes, fontsize=int(0.9*fsize), verticalalignment='top', c=kwargs.get('c', 'black'))
        ax.text(kwargs.get('l_2', l2)[0], kwargs.get('l_2', l2)[1],
                "c:{:.2f}%,\nm:{:.2f}%,\nn:{:.2f}%".format(
                    sc_percentage_cyto[1], sc_percentage_mito[1], sc_percentage_nuc[1]),
                transform=ax.transAxes, fontsize=int(0.9*fsize), verticalalignment='top', c=kwargs.get('c', 'black'))
        ax.text(kwargs.get('l_3', l3)[0], kwargs.get('l_3', l3)[1],
                "c:{:.2f}%,\nm:{:.2f}%,\nn:{:.2f}%".format(
                    sc_percentage_cyto[2], sc_percentage_mito[2], sc_percentage_nuc[2]),
                transform=ax.transAxes, fontsize=int(0.9*fsize), verticalalignment='top', c=kwargs.get('c', 'black'))
        ax.text(kwargs.get('l_4', l4)[0], kwargs.get('l_4', l4)[1],
                "c:{:.2f}%,\nm:{:.2f}%,\nn:{:.2f}%".format(
                    sc_percentage_cyto[3], sc_percentage_mito[3], sc_percentage_nuc[3]),
                transform=ax.transAxes, fontsize=int(0.9*fsize), verticalalignment='top', c=kwargs.get('c', 'black'))
    else:
        pass


def update_index(df):
    '''
    Update index of dataframe

    Args:

    df (pandas.DataFrame): dataframe

    Returns:

    pandas.DataFrame
    '''
    df.compartment = df.compartment.replace(
        {'mitochondrial': 'm', 'cytoplasm': 'c', 'nuclear': 'n'})
    new = {i: i+' ('+df.loc[(df.index == i)]
           ['compartment'][0]+')' for i in df.index}
    df.index = df.index.map(new)
    return df


def get_newlist(met_list, n):
    '''
    Get list of metabolites

    Args:

    met_list (list): list of metabolites
    n (int): number of metabolites to show

    Returns:

    list: list of metabolites
    '''
    newlist = []
    count = 0
    for i in range(0, len(met_list), n):
        newlist.append(', '.join(list(met_list[count:i+n])))
        count += n
    return newlist


def add_metabolite_names1(df, ax, colname, colname2,
                          l=(0.02, 0.8), l2=(0.7, 0.8), l3=(0.02, 0.4), l4=(0.7, 0.4),
                          n=1, c='black', fsize=12, compartments=False, **kwargs):
    '''
    Add metabolite names

    Args:

    df (pandas.DataFrame): data
    ax (matplotlib.pyplot.axis): axis
    colname (str): column name for x-axis
    colname2 (str): column name for y-axis
    l (tuple): location of metabolite names for upper left quadrant
    l2 (tuple): location of metabolite names for upper right quadrant
    l3 (tuple): location of metabolite names for lower left quadrant
    l4 (tuple): location of metabolite names for lower right quadrant
    n (int): number of metabolites to show
    c (str): color
    fsize (int): font size
    compartments (bool): show compartment
    kwargs (dict): keyword arguments

    Returns:

    matplotlib.pyplot.axis
    '''
    if compartments == True:
        UL = list(update_index(df.loc[(np.log2(df[colname]) > 0.0) & (
            np.log2(df[colname2]) < 0.0)]).index.unique())
        UR = list(update_index(df.loc[(np.log2(df[colname]) > 0.0) & (
            np.log2(df[colname2]) > 0.0)]).index.unique())
        LL = list(update_index(df.loc[(np.log2(df[colname]) < 0.0) & (
            np.log2(df[colname2]) < 0.0)]).index.unique())
        LR = list(update_index(df.loc[(np.log2(df[colname]) < 0.0) & (
            np.log2(df[colname2]) > 0.0)]).index.unique())
    else:
        UL = [i for i in list(df.loc[(np.log2(df[colname]) > 0.0) & (
            np.log2(df[colname2]) < 0.0)].index.unique())]
        UR = [i for i in list(df.loc[(np.log2(df[colname]) > 0.0) & (
            np.log2(df[colname2]) > 0.0)].index.unique())]
        LL = [i for i in list(df.loc[(np.log2(df[colname]) < 0.0) & (
            np.log2(df[colname2]) < 0.0)].index.unique())]
        LR = [i for i in list(df.loc[(np.log2(df[colname]) < 0.0) & (
            np.log2(df[colname2]) > 0.0)].index.unique())]
    textstr = '\n'.join(get_newlist(UL, n))
    textstr2 = '\n'.join(get_newlist(UR, kwargs.get('n2', n)))
    textstr3 = '\n'.join(get_newlist(LL, kwargs.get('n3', n)))
    textstr4 = '\n'.join(get_newlist(LR, kwargs.get('n4', n)))
    ax.text(l[0], l[1], textstr, transform=ax.transAxes,
            fontsize=fsize, verticalalignment='top', c=c)
    ax.text(l2[0], l2[1], textstr2, transform=ax.transAxes,
            fontsize=fsize, verticalalignment='top', c=c)
    ax.text(l3[0], l3[1], textstr3, transform=ax.transAxes,
            fontsize=fsize, verticalalignment='top', c=c)
    ax.text(l4[0], l4[1], textstr4, transform=ax.transAxes,
            fontsize=fsize, verticalalignment='top', c=c)


def scatter_with_errorbar(xdata, ydata, xerr, yerr, ax, colors=None, x='WT_Car',
                          y='KO_Car', figtitle_suffix=' (RNAseq)',
                          legend_loc='upper right', compartment=False,
                          xlabel='Measured metabolomics foldchange (log2)',
                          ylabel='Predicted foldchange (log2)', cmap=None,
                          fig_title=False, xlim=(-3.5, 3.5), ylim=(-3.5, 3.5)):
    '''
    Plot scatter plot with error bars

    Args:

    xdata (numpy.array): x-axis data
    ydata (numpy.array): y-axis data
    xerr (numpy.array): x-axis error
    yerr (numpy.array): y-axis error
    ax (matplotlib.pyplot.axis): axis
    colors (numpy.array): colors
    x (str): x-axis label
    y (str): y-axis label
    figtitle_suffix (str): figure title suffix
    legend_loc (str): legend location
    compartment (bool): compartment
    xlabel (str): x-axis label
    ylabel (str): y-axis label
    cmap (str): colormap
    fig_title (bool): figure title
    xlim (tuple): x-axis limits
    ylim (tuple): y-axis limits

    Returns:

    matplotlib.pyplot.axis
    '''

    if compartment == True:
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='cytoplasm',
                                  markerfacecolor='r', markersize=15),
                           Line2D([0], [0], marker='o', color='w', label='mitochondrial',
                                  markerfacecolor='g', markersize=15),
                           Line2D([0], [0], marker='o', color='w', label='nuclear',
                                  markerfacecolor='b', markersize=15)]

        ax.scatter(xdata, ydata, zorder=3, c=colors, cmap=cmap)
        ax.errorbar(xdata, ydata, yerr=yerr, xerr=xerr,
                    fmt="o", ecolor=colors, cmap=cmap)
        ax.legend(handles=legend_elements, loc=legend_loc)
    else:
        sc = ax.errorbar(xdata, ydata, yerr=yerr, xerr=xerr,
                         fmt="o", cmap=cmap, ecolor=colors)
        # plt.colorbar(sc)
    ax.axhline(0.0, ls="--", c="k")
    ax.axvline(0.0, ls="--", c="k")
    if fig_title == True:
        ax.set_title(y+' vs '+x+figtitle_suffix)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


def parity_with_errorbars(df, xcolumn, ycolumn, xerr_column, yerr_column,
                          filename=None, save=False, figsize=(10, 8),
                          xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), colors='C0',
                          xlabel='log$_2$(predicted change)',
                          ylabel='log$_2$(measured change)', fig_title=None,
                          legend_loc='upper right', cmap=None, figtitle_suffix=None,
                          show_percentage=False, show_sc_percentages=False, show_metabolite_names=False,
                          l_p=[(0.05, 0.95), (0.55, 0.95),
                               (0.05, 0.25), (0.55, 0.25)],
                          l_n=[(0.05, 0.9), (0.55, 0.9),
                               (0.05, 0.2), (0.55, 0.2)],
                          n_n=[1, 1, 2, 1], fsize=10):
    '''
    Plot parity plot with error bars

    Args:

    df (pandas.DataFrame): data
    xcolumn (str): column name for x-axis
    ycolumn (str): column name for y-axis
    xerr_column (str): column name for x-axis error
    yerr_column (str): column name for y-axis error
    filename (str): filename
    save (bool): save figure
    figsize (tuple): figure size
    xlim (tuple): x-axis limits
    ylim (tuple): y-axis limits
    colors (str): color
    xlabel (str): x-axis label
    ylabel (str): y-axis label
    fig_title (str): figure title
    legend_loc (str): legend location
    cmap (str): colormap
    figtitle_suffix (str): figure title suffix
    show_percentage (bool): show percentage
    show_sc_percentages (bool): show percentage for each compartment
    show_metabolite_names (bool): show metabolite names
    l_p (list): list of tuples for percentage locations
    l_n (list): list of tuples for metabolite names locations
    n_n (list): list of number of metabolites to show
    fsize (int): font size

    Returns:

    matplotlib.pyplot.figure
    matplotlib.pyplot.axis
    '''
    fig, ax = plt.subplots(figsize=figsize)
    scatter_with_errorbar(xdata=np.log2(df[xcolumn]), ydata=np.log2(df[ycolumn]),
                          xerr=df[xerr_column], yerr=df[yerr_column], ax=ax,
                          colors=colors, figtitle_suffix=figtitle_suffix, legend_loc=legend_loc,
                          compartment=False, xlabel=xlabel, ylabel=ylabel, cmap=cmap,
                          fig_title=fig_title, xlim=xlim, ylim=ylim)

    if show_percentage == True:
        add_percentage(df=df, ax=ax, colname2=xcolumn, colname=ycolumn,
                       show_sc_percentages=show_sc_percentages, fsize=1.2*fsize,
                       l=l_p[0], l2=l_p[1], l3=l_p[2], l4=l_p[3])
    if show_metabolite_names == True:
        add_metabolite_names1(df=df, ax=ax, colname2=xcolumn, colname=ycolumn,
                              fsize=fsize, l=l_n[0], l2=l_n[1], l3=l_n[2], l4=l_n[3],
                              n=n_n[0], n2=n_n[1], n3=n_n[2], n4=n_n[3])
    if save == True:
        return fig.savefig(filename, dpi=300, bbox_inches='tight')


def differential_exp(df, figsize, ylabel='genes', xlabel=None):
    '''
    Plot differential expression

    Args:

    df (pandas.DataFrame): data
    figsize (tuple): figure size
    ylabel (str): y-axis label
    xlabel (str): x-axis label

    Returns:

    matplotlib.pyplot.figure
    matplotlib.pyplot.axis
    '''
    fig, ax = plt.subplots(figsize=figsize)
    cm = sns.heatmap(data=df, cmap='seismic',
                     center=0, robust=True, xticklabels=False, yticklabels=False,
                     ax=ax, cbar_kws={
                         'label': '$\Delta\Phi$', 'location': 'top'},
                     )
    cm.set_ylabel(ylabel)
    cm.set_xlabel(xlabel)
    return fig, ax


def control_coeff(df, figsize, xlabel='genes', ylabel='metabolites'):
    '''
    Plot control coefficients

    Args:

    df (pandas.DataFrame): data
    figsize (tuple): figure size
    xlabel (str): x-axis label
    ylabel (str): y-axis label

    Returns:

    matplotlib.pyplot.figure
    matplotlib.pyplot.axis
    '''
    cm = sns.clustermap(data=df, cmap='seismic', center=0, robust=True,
                        figsize=figsize, yticklabels=False, dendrogram_ratio=(0.15, 0.0),
                        xticklabels=False, cbar_kws={'label': '$C_{\Phi}^{M}$'},
                        cbar_pos=(0.02, 0.8, 0.02, 0.2),
                        col_cluster=False, row_cluster=False,
                        )
    cm.ax_heatmap.set_xlabel(xlabel)
    cm.ax_heatmap.set_ylabel(ylabel)
    return cm


def response_coeff(df, figsize=(15, 8), xlabel='genes', ylabel='metabolites', fsize=20):
    '''
    Plot response coefficients

    Args:

    df (pandas.DataFrame): data
    figsize (tuple): figure size
    xlabel (str): x-axis label
    ylabel (str): y-axis label
    fsize (int): font size

    Returns:

    matplotlib.pyplot.figure
    '''
    cm = sns.clustermap(data=df, cmap='seismic', center=0, robust=True,
                        figsize=figsize, col_cluster=False, row_cluster=False,
                        yticklabels=True, dendrogram_ratio=(0.12, 0.0),
                        xticklabels=True, cbar_kws={'label': '$\Gamma^{\Phi}_{M}$'},
                        cbar_pos=(0.02, 0.8, 0.02, 0.18),
                        )
    cm.ax_heatmap.set_xlabel(xlabel, fontdict={'size': fsize})
    cm.ax_heatmap.set_ylabel(ylabel, fontdict={'size': fsize})
    return cm
