import pandas as pd
import numpy as np
import recon  # .recon as recon


class Utilities:
    def __init__(self) -> None:
        self.recon = recon.Recon()

    def divide_all_columns_by_each_column(self, df):
        """divide all columns by each column

        Args:
            df (pandas.DataFrame): dataframe to divide

        Returns:
            pandas.DataFrame: dataframe with all columns divided by each column
        """
        _df = pd.concat([df[df.columns.difference([col])].div(df[col], axis=0)
                        .add_suffix('_'+col) for col in df.columns], axis=1)
        return _df

    def get_consistent_mets(self, df):
        """get metabolites that are consistently produced or consumed

        Args:
            df (pandas.DataFrame): dataframe of metabolite changes

        Returns:
            pandas.DataFrame: dataframe of metabolite changes that are consistently produced or consumed
        """
        consistent_mets = set(df[(df < 1.0).all(axis=1)].index).union(
            set(df[(df > 1.0).all(axis=1)].index))
        return df[df.index.isin(consistent_mets)]

    def get_metabolites_from_pathway(self, pathwayName='Methionine'):
        """get metabolites from pathway

        Args:
            pathwayName (str, optional): name of pathway. Defaults to 'Methionine'.

        Returns:
            list: list of metabolites in pathway
        """
        mets = []
        for rxn in self.recon.reactions[self.recon.reactions.subsystem.str.contains(pathwayName)]['formula']:
            mets = mets + rxn.split(' ')

        mets = list(set([s for s in mets if s not in ['+', '', '<=>', '->']]))
        return mets

    def change_across_pathways(self, df):
        """_summary_

        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """
        _df = pd.DataFrame()
        for pathway in self.recon.reactions.subsystem.unique():
            _res = df[df.index.isin(
                self.get_metabolites_from_pathway(pathwayName=pathway))]
            _res = pd.DataFrame(_res)
            _res['pathway'] = pathway
            _df = pd.concat((_df, _res))
        return _df

    def mean_change_across_pathways(self, res):
        """calculates the mean change across pathways

        Args:
            res (pandas.DataFrame): DataFrame of metabolite changes

        Returns:
            pandas.DataFrame: DataFrame of mean metabolite changes across pathways
        """
        df = pd.DataFrame()
        for pathway in self.recon.reactions.subsystem.unique():
            _res = res[res.index.isin(
                self.get_metabolites_from_pathway(pathwayName=pathway))]
            df = pd.concat(
                (df, _res.groupby(_res.index).mean().mean()), axis=1)
            df = df.rename(columns={0: pathway})
        return df.T

    def calculate_cc(self, g_df, grouping=None):
        """calculate correlation coefficient

        Args:
            g_df (pandas.DataFrame): DataFrame of gene expression data
            grouping (str, optional): grouping of genes. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame of correlation coefficients
        """
        gs1 = self.reshape(self.recon.gs)
        if grouping in ['cytoplasm', 'mitochondrial', 'nuclear']:
            gs1 = gs1[gs1.compartment == grouping].drop('compartment', axis=1).groupby(
                'metabolites').mean()
        else:
            gs1 = gs1.drop('compartment', axis=1).groupby('metabolites').mean()
        cc = 1 - (gs1.abs().sum(axis=1) -
                  gs1[list(g_df.index)].abs().sum(axis=1)) / gs1.abs().sum(axis=1)
        return cc

    def fill_missing_values(self, df, value=0, replacewith=np.nan):
        """Fill missing values with a value and replace with the mean of the row

        Args:
            df (pandas.DataFrame): dataframe to fill missing values
            value (int, optional): value to fill missing values with. Defaults to 0.
            replacewith (int, optional): value to replace with. Defaults to np.nan.

        Returns:
            pandas.DataFrame: dataframe with missing values filled
        """
        df_ = df.replace(value, replacewith)
        df_ = df_.apply(lambda row: row.fillna(row.mean()), axis=1)
        return df_.dropna(axis=0)

    def make_data_comparable(self, df1, df2, fill_missing=True):
        """make data comparable by removing missing values and filling missing values

        Args:
            df1 (pandas.DataFrame): dataframe 1
            df2 (pandas.DataFrame): dataframe 2
            fill_missing (bool, optional): fill missing values. Defaults to True.

        Returns:
            pandas.DataFrame: dataframe 1 and dataframe 2 with missing values removed
        """
        df1 = df1[~(df1.mean(axis=1) == 0)]
        df2 = df2[~(df2.mean(axis=1) == 0)]
        common_index = list(set(df1.index) & set(df2.index))
        df1 = df1[df1.index.isin(common_index)]
        df2 = df2[df2.index.isin(common_index)]
        if fill_missing == True:
            return self.fill_missing_values(df1), self.fill_missing_values(df2)
        else:
            return df1, df2

    def count_zeros(self, df):
        """Count the number of entries with zero

        Args:
            df (pandas.DataFrame): dataframe to count zeros

        Returns:
            int: number of entries with zero
        """
        return print(f'Found {len(df[(df == 0.0).any(axis=1)])} entries with zero')

    def drop_constant_columns(self, df):
        """Drop constant columns

        Args:
            df (pandas.DataFrame): dataframe to drop constant columns

        Returns:
            pandas.DataFrame: dataframe with constant columns dropped
        """
        return df.loc[:, (df != df.iloc[0]).any()]

    def compounds_mapping(self):
        """mapping of compounds

        Returns:
            dict: dictionary of compounds
        """
        compounds = {'Acetyl CoA': 'Acetyl-CoA', 'Aconitate': 'cis-Aconitic acid', 'ADP': 'ADP',
                     'Alpha.Ketoglutarate': 'Oxoglutaric acid', 'ATP': 'ATP',
                     'Bisphosphoglycerate2.3': '2,3-Diphosphoglyceric acid',
                     'Citrate': 'Citric acid', 'Citrulline': 'Citrulline',
                     'Fructose.1.6.Bisphosphate': 'Fructose 1,6-bisphosphate',
                     'Fructose.6.Phosphate': 'Fructose 6-phosphate', 'Fumarate': 'Fumaric acid',
                     'Glucose': 'D-Glucose', 'Glucose.1.Phosphate': 'Glucose 1-phosphate',
                     'Glucose.6.Phosphate': 'Glucose 6-phosphate', 'Isocitrate': 'Isocitric acid',
                     'Lactate': 'L-Lactic acid', 'Malate': 'L-Malic acid', 'Malonyl CoA': 'Malonyl-CoA',
                     'Phosphoenolpyruvate': 'Phosphoenolpyruvic acid', 'Phosphogluconate6': '6-Phosphogluconic acid',
                     'Propionyl CoA': 'Propionyl-CoA', 'Pyruvate': 'Pyruvic acid',
                     'Sedoheptulose.7.Phosphate': 'D-Sedoheptulose 7-phosphate', 'Succinate': 'Succinic acid',
                     'Succinyl CoA': 'Succinyl-CoA', 'Alanine': 'L-Alanine', 'Arginine': 'L-Arginine',
                     'Aspartic.acid': 'L-Aspartic acid', 'Aspartic acid': 'L-Aspartic acid',
                     'Glutamic.acid': 'L-Glutamic acid', 'Glutamic acid': 'L-Glutamic acid',
                     'Glutamine': 'L-Glutamine', 'Glycine': 'Glycine',
                     'Histidine': 'L-Histidine', 'Isoleucine': 'L-Isoleucine', 'Leucine': 'L-Leucine',
                     'Lysine': 'L-Lysine', 'Methionine': 'L-Methionine', 'Phenylalanine': 'L-Phenylalanine',
                     'Proline': 'L-Proline', 'Serine': 'L-Serine', 'Threonine': 'L-Threonine',
                     'Tryptophan': 'L-Tryptophan', 'Tryptophane': 'L-Tryptophan', 'Tyrosine': 'L-Tyrosine',
                     'Tyrosin': 'L-Tyrosine',
                     'Valine': 'L-Valine', 'b-Alanine': 'beta-Alanine'}
        return compounds

    def map_gene(self, df, g_mapping, mapping_column='ensembl_gene'):
        """maps gene identifiers to gene numbers

        Args:
            df (pandas.DataFrame): DataFrame with gene expression data
            g_mapping (pandas.DataFrame): DataFrame with gene mapping
            mapping_column (str, optional): _description_. Defaults to 'ensembl_gene'.

        Returns:
            pandas.DataFrame: DataFrame with gene numbers
        """
        df = df[df.index.isin(g_mapping[mapping_column])]
        g_map = g_mapping.set_index(mapping_column).to_dict()['gene_number']
        df.index = df.index.map(g_map).astype('str')
        return df

    def get_fullnames(self, res):
        """get full names of metabolites

        Args:
            res (pandas.DataFrame): dataframe with metabolites

        Returns:
            dict: dictionary of metabolites
        """
        m_names = self.recon.metabolites
        fullnames = {}
        for x in res.index:
            try:
                fullnames[x] = m_names.loc[(
                    m_names.abbreviation == x[:-3]), 'fullName'].iloc[0]+x[-3:]
            except IndexError:
                pass
        return fullnames

    def reshape(self, res0, include=None):
        """reshape dataframe

        Args:
            res0 (pandas.DataFrame): dataframe to reshape
            include (list, optional): list of compartments to include. Defaults to ['cytoplasm', 'mitochondrial', 'nuclear'].

        Returns:
            pandas.DataFrame: reshaped dataframe
        """
        res0 = res0.T.rename(columns=self.get_fullnames(res0)).T
        res0['compartment'] = [i[-3:] for i in res0.index]
        res0['metabolites'] = [i[:-3] for i in res0.index]
        res0.compartment = res0.compartment.map({'[c]': 'cytoplasm', '[e]': 'extracellular', '[l]': 'luminal',
                                                '[m]': 'mitochondrial', '[n]': 'nuclear', '[x]': 'peroxisomal',
                                                 '[g]': 'golgi_aparatus', '[r]': 'endoplasic_reticular'})
        if include is not None:
            return res0.loc[res0.compartment.isin(include)]
        else:
            return res0  # .loc[~res0.compartment.isin(['extracellular', ])]

    def calculate_percentage(self, df, colname, colname2):
        """calculate percentage of metabolites in each quadrant

        Args:
            df (pandas.DataFrame): dataframe with metabolites
            colname (str): name of first column
            colname2 (str): name of second column

        Returns:
            tuple: percentages of metabolites in each quadrant
        """
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

    def calculate_sc_percentage(self, df, compartment, colname, colname2):
        """calculate percentage of metabolites in each quadrant

        Args:
            df (pandas.DataFrame): dataframe with metabolites
            compartment (str): name of compartment
            colname (str): name of first column
            colname2 (str): name of second column

        Returns:
            tuple: percentages of metabolites in each quadrant
        """
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

    def add_percentage(self, df, ax, colname='RNASeq', colname2='IFN+F vs IFN', l=(-1.5, 0.2), l2=(0.7, 0.2), l3=(-1.5, -0.45),
                       l4=(0.7, -0.45), fsize=12, show_sc_percentages=True, **kwargs):
        """add percentage of metabolites in each quadrant

        Args:
            df (pandas.DataFrame): dataframe with metabolites
            ax (matplotlib.axes.Axes): axes to plot on
            colname (str, optional): name of first column. Defaults to 'RNASeq'.
            colname2 (str, optional): name of second column. Defaults to 'IFN+F vs IFN'.
            l (tuple, optional): location of first percentage. Defaults to (-1.5, 0.2).
            l2 (tuple, optional): location of second percentage. Defaults to (0.7, 0.2).
            l3 (tuple, optional): location of third percentage. Defaults to (-1.5, -0.45).
            l4 (tuple, optional): location of fourth percentage. Defaults to (0.7, -0.45).
            fsize (int, optional): fontsize. Defaults to 12.
            show_sc_percentages (bool, optional): show percentages for each compartment. Defaults to True.

        Returns:
            None
        """
        total_percentage = self.calculate_percentage(
            df=df, colname=colname, colname2=colname2)
        ax.text(l[0], l[1], "{:.2f}%".format(total_percentage[0]),
                transform=ax.transAxes, fontsize=fsize, verticalalignment='top', c=kwargs.get('c', 'black'))

        ax.text(l2[0], l2[1], "{:.2f}%".format(total_percentage[1]),
                transform=ax.transAxes, fontsize=fsize, verticalalignment='top', c=kwargs.get('c', 'black'))

        ax.text(l3[0], l3[1], "{:.2f}%".format(total_percentage[2]),
                transform=ax.transAxes, fontsize=fsize, verticalalignment='top', c=kwargs.get('c', 'black'))

        ax.text(l4[0], l4[1], "{:.2f}%".format(total_percentage[3]),
                transform=ax.transAxes, fontsize=fsize, verticalalignment='top', c=kwargs.get('c', 'black'))

        if show_sc_percentages == True:
            sc_percentage_cyto = self.calculate_sc_percentage(
                df=df, compartment='cytoplasm', colname=colname, colname2=colname2)
            sc_percentage_mito = self.calculate_sc_percentage(
                df=df, compartment='mitochondrial', colname=colname, colname2=colname2)
            sc_percentage_nuc = self.calculate_sc_percentage(
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

    def add_metabolite_names(self, df, ax, colname='RNASeq', colname2='IFN+F vs IFN', l=(0.02, 0.8), l2=(0.9, 0.8),
                             l3=(0.02, 0.4), l4=(0.7, 0.3), c='black', n=1, fsize=12, compartments=False, **kwargs):
        """add metabolite names to each quadrant

        Args:
            df (pandas.DataFrame): dataframe with metabolites
            ax (matplotlib.axes.Axes): axes to plot on
            colname (str, optional): name of first column. Defaults to 'RNASeq'.
            colname2 (str, optional): name of second column. Defaults to 'IFN+F vs IFN'.
            l (tuple, optional): location of first metabolite name. Defaults to (0.02, 0.8).
            l2 (tuple, optional): location of second metabolite name. Defaults to (0.9, 0.8).
            l3 (tuple, optional): location of third metabolite name. Defaults to (0.02, 0.4).
            l4 (tuple, optional): location of fourth metabolite name. Defaults to (0.7, 0.3).
            c (str, optional): color of text. Defaults to 'black'.
            n (int, optional): number of metabolites to show. Defaults to 1.
            fsize (int, optional): fontsize. Defaults to 12.
            compartments (bool, optional): show compartments. Defaults to False.

        Returns:
            None
        """
        if compartments == True:
            UL = list(self.update_index(df.loc[(np.log2(df[colname]) > 0.0) & (
                np.log2(df[colname2]) < 0.0)]).index.unique())
            UR = list(self.update_index(df.loc[(np.log2(df[colname]) > 0.0) & (
                np.log2(df[colname2]) > 0.0)]).index.unique())
            LL = list(self.update_index(df.loc[(np.log2(df[colname]) < 0.0) & (
                np.log2(df[colname2]) < 0.0)]).index.unique())
            LR = list(self.update_index(df.loc[(np.log2(df[colname]) < 0.0) & (
                np.log2(df[colname2]) > 0.0)]).index.unique())
        else:
            UL = [i[0] for i in list(df.loc[(np.log2(df[colname]) > 0.0) & (
                np.log2(df[colname2]) < 0.0)].index.unique())]
            UR = [i[0] for i in list(df.loc[(np.log2(df[colname]) > 0.0) & (
                np.log2(df[colname2]) > 0.0)].index.unique())]
            LL = [i[0] for i in list(df.loc[(np.log2(df[colname]) < 0.0) & (
                np.log2(df[colname2]) < 0.0)].index.unique())]
            LR = [i[0] for i in list(df.loc[(np.log2(df[colname]) < 0.0) & (
                np.log2(df[colname2]) > 0.0)].index.unique())]
        textstr = '\n'.join(self.get_newlist(UL, n))
        textstr2 = '\n'.join(self.get_newlist(UR, kwargs.get('n2', n)))
        textstr3 = '\n'.join(self.get_newlist(LL, kwargs.get('n3', n)))
        textstr4 = '\n'.join(self.get_newlist(LR, kwargs.get('n4', n)))
        ax.text(l[0], l[1], textstr, transform=ax.transAxes,
                fontsize=fsize, verticalalignment='top', c=c)
        ax.text(l2[0], l2[1], textstr2, transform=ax.transAxes,
                fontsize=fsize, verticalalignment='top', c=c)
        ax.text(l3[0], l3[1], textstr3, transform=ax.transAxes,
                fontsize=fsize, verticalalignment='top', c=c)
        ax.text(l4[0], l4[1], textstr4, transform=ax.transAxes,
                fontsize=fsize, verticalalignment='top', c=c)

    def update_index(self, df):
        """update index of dataframe

        Args:
            df (pandas.DataFrame): DataFrame with metabolite changes        

        Returns:
            pandas.DataFrame: DataFrame with updated index
        """
        df.compartment = df.compartment.replace(
            {'mitochondrial': 'm', 'cytoplasm': 'c', 'nuclear': 'n'})
        new = {i: i+' ('+df.loc[(df.index == i)]
               ['compartment'][0]+')' for i in df.index}
        df.index = df.index.map(new)
        return df

    def get_newlist(self, met_list, n):
        """get list of metabolites

        Args:
            met_list (list): list of metabolites
            n (int): number of metabolites to show

        Returns:
            list: list of metabolites
        """
        newlist = []
        count = 0
        for i in range(0, len(met_list), n):
            newlist.append(', '.join(list(met_list[count:i+n])))
            count += n
        return newlist

    def get_pathways(self, list_of_mapped_compounds):
        """get pathways for list of mapped compounds

        Args:
            list_of_mapped_compounds (list): list of mapped compounds

        Returns:
            dict: dictionary of pathways
        """
        m_recon = self.recon.metabolites
        r_recon = self.recon.reactions
        pathways = {}
        for met in m_recon[m_recon.fullName.isin(list_of_mapped_compounds)]['abbreviation']:
            pathways[m_recon[m_recon.abbreviation == met]['fullName'].iloc[0]] = list(
                set(r_recon[r_recon.formula.str.contains(met)]['subsystem']))
        subsytems = []
        for v in pathways.values():
            subsytems = subsytems + v
        m_paths = {}
        for p in list(set(subsytems)):
            m_paths[p] = [k for k, v in pathways.items() if p in v]
        return m_paths

    def print_genes_mapping(self, df_recon):
        """print genes mapping summary

        Args:
            df_recon (pandas.DataFrame): DataFrame with genes mapping

        Returns:
            None
        """
        genes = []
        for r in self.recon.model.reactions:
            genes.append([g.id for g in r.genes])

        genes_dict = dict(
            zip([r.id for r in self.recon.model.reactions], genes))
        enzymes_dict = {k: v for k, v in genes_dict.items() if v}
        print(
            f'Recon3D model summary:\n\t{len(enzymes_dict)} of {len(genes_dict)} are enzyme catalysed reactions that form {len(self.recon.reactions.subsystem.unique())} subsystems.')
        rxns = []
        for i in enzymes_dict.keys():
            if len(set(df_recon.index) & set(enzymes_dict[i])) > 0:
                rxns.append(i)
            else:
                pass
        print(
            f'Mapping summary:\n\t{len(df_recon.index.unique())} genes catalyse {len(set(rxns))} of {len(enzymes_dict)} enzyme catalysed reactions in Recon3D.')
        subsystems = list(set(
            [self.recon.reactions[self.recon.reactions.abbreviation == r]['subsystem'].iloc[0] for r in rxns]))
        print(
            f'\tThese {len(set(rxns))} enzyme catalysed reactions form {len(subsystems)} subsystems in Recon3D model.')
        return subsystems


if __name__ == 'main':
    u = Utilities()
