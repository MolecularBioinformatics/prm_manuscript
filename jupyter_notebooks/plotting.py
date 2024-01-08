import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def hist(pr, bins=10, save=False, filename=None,
         figsize=(10, 8), xlabel='Predicted change (log2)',
         ylabel='No. of metabolites', xlim=None, color=None, ylim=None):
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
    df.compartment = df.compartment.replace(
        {'mitochondrial': 'm', 'cytoplasm': 'c', 'nuclear': 'n'})
    new = {i: i+' ('+df.loc[(df.index == i)]
           ['compartment'][0]+')' for i in df.index}
    df.index = df.index.map(new)
    return df


def get_newlist(met_list, n):
    newlist = []
    count = 0
    for i in range(0, len(met_list), n):
        newlist.append(', '.join(list(met_list[count:i+n])))
        count += n
    return newlist


def add_metabolite_names1(df, ax, colname, colname2,
                          l=(0.02, 0.8), l2=(0.7, 0.8), l3=(0.02, 0.4), l4=(0.7, 0.4),
                          n=1, c='black', fsize=12, compartments=False, **kwargs):
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
    """Creates scatter plot with error bars

    Args:
        xdata (pandas.series): data on x-axis
        ydata (pandas.series): data on y-axis
        xerr (list): _description_
        yerr (list): _description_
        ax (_type_): _description_
        colors (_type_, optional): _description_. Defaults to None.
        x (str, optional): _description_. Defaults to 'WT_Car'.
        y (str, optional): _description_. Defaults to 'KO_Car'.
        figtitle_suffix (str, optional): _description_. Defaults to ' (RNAseq)'.
        legend_loc (str, optional): _description_. Defaults to 'upper right'.
        compartment (bool, optional): _description_. Defaults to False.
    """
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
                          n_n=[1, 1, 2, 1]):
    fig, ax = plt.subplots(figsize=figsize)
    scatter_with_errorbar(xdata=np.log2(df[xcolumn]), ydata=np.log2(df[ycolumn]),
                          xerr=df[xerr_column], yerr=df[yerr_column], ax=ax,
                          colors=colors, figtitle_suffix=figtitle_suffix, legend_loc=legend_loc,
                          compartment=False, xlabel=xlabel, ylabel=ylabel, cmap=cmap,
                          fig_title=fig_title, xlim=xlim, ylim=ylim)

    if show_percentage == True:
        add_percentage(df=df, ax=ax, colname2=xcolumn, colname=ycolumn,
                       show_sc_percentages=show_sc_percentages, fsize=15,
                       l=l_p[0], l2=l_p[1], l3=l_p[2], l4=l_p[3])
    if show_metabolite_names == True:
        add_metabolite_names1(df=df, ax=ax, colname2=xcolumn, colname=ycolumn,
                              fsize=10, l=l_n[0], l2=l_n[1], l3=l_n[2], l4=l_n[3],
                              n=n_n[0], n2=n_n[1], n3=n_n[2], n4=n_n[3])
    if save == True:
        return fig.savefig(filename, dpi=300, bbox_inches='tight')


def differential_exp(df, figsize, ylabel='genes', xlabel=None):
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
    cm = sns.clustermap(data=df, cmap='seismic', center=0, robust=True,
                        figsize=figsize, col_cluster=False, row_cluster=False,
                        yticklabels=True, dendrogram_ratio=(0.12, 0.0),
                        xticklabels=True, cbar_kws={'label': '$\Gamma^{\Phi}_{M}$'},
                        cbar_pos=(0.02, 0.8, 0.02, 0.18),
                        )
    cm.ax_heatmap.set_xlabel(xlabel, fontdict={'size': fsize})
    cm.ax_heatmap.set_ylabel(ylabel, fontdict={'size': fsize})
    return cm
