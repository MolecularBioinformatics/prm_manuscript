import cobra
import pandas as pd
import importlib.resources


class Recon:
    def __init__(self) -> None:
        self.genes = pd.read_csv('./recon-store-genes.tsv', sep='\t')
        # with importlib.resources.path('recon', 'recon-store-genes.tsv') as path:
        #     self.genes = pd.read_csv(path, sep='\t')

        self.reactions = pd.read_csv('./recon-store-reactions.tsv', sep='\t')
        # with importlib.resources.path('recon', 'recon-store-reactions.tsv') as path:
        #     self.reactions = pd.read_csv(path, sep='\t')

        self.metabolites = pd.read_csv(
            './recon-store-metabolites.tsv', sep='\t')
        # with importlib.resources.path('recon', 'recon-store-metabolites.tsv') as path:
        #     self.metabolites = pd.read_csv(path, sep='\t')
        self.model = cobra.io.load_json_model('./Recon3D.json')
        # with importlib.resources.path('recon', 'Recon3D.json') as path:
        #     self.model = cobra.io.load_json_model(path)
        self.gs = pd.read_csv('./genes_sensitivity.tsv', sep='\t', index_col=0)
        # with importlib.resources.path('recon', 'genes_sensitivity.tsv') as path:
        #     self.gs = pd.read_csv(path, sep='\t', index_col=0)

        self.gpr = {rxn.id: rxn.gene_reaction_rule for rxn in self.model.reactions if len(
            rxn.gene_reaction_rule) != 0}

        self.catalysed_rxns = self.reactions[self.reactions.abbreviation.isin(
            self.gpr.keys())]

    def _get_nrxns(self, df, column_name):
        rxns = self.get_reactions(list(df.index))
        n_rxns = pd.DataFrame.from_dict(
            {i: len(rxns[i]) for i in rxns}, orient='index', columns=[column_name, ])
        return n_rxns

    def get_n_reactions(self, df):
        """get number of reactions catalysed by genes in df

        Args:
            df (pandas.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        gpr = {rxn.id: rxn.gene_reaction_rule for rxn in self.model.reactions if len(
            rxn.gene_reaction_rule) != 0}
        trxns = self.reactions[self.reactions.abbreviation.isin(
            gpr.keys())]
        tn_rxns = pd.DataFrame.from_dict({i: len(
            trxns[trxns.subsystem == i]) for i in trxns['subsystem'].unique()}, orient='index', columns=['total', ])
        if df.index.dtype != str:
            df.index = df.index.astype('str')
        n_rxns = self._get_nrxns(df, column_name='mapped')
        df_nrxns = pd.concat((tn_rxns, n_rxns), axis=1)
        return df_nrxns

    def print_genes_mapping(self, df_recon):
        genes = []
        for r in self.model.reactions:
            genes.append([g.id for g in r.genes])

        genes_dict = dict(zip([r.id for r in self.model.reactions], genes))
        enzymes_dict = {k: v for k, v in genes_dict.items() if v}
        print(
            f'Recon3D model summary:\n\t{len(enzymes_dict)} of {len(genes_dict)} are enzyme catalysed reactions that form {len(self.reactions.subsystem.unique())} subsystems.')
        rxns = []
        for i in enzymes_dict.keys():
            if len(set(df_recon.index) & set(enzymes_dict[i])) > 0:
                rxns.append(i)
            else:
                pass
        print(
            f'Mapping summary:\n\t{len(df_recon.index.unique())} genes catalyse {len(set(rxns))} of {len(enzymes_dict)} enzyme catalysed reactions in Recon3D.')
        subsystems = list(set(
            [self.reactions[self.reactions.abbreviation == r]['subsystem'].iloc[0] for r in rxns]))
        print(
            f'\tThese {len(set(rxns))} enzyme catalysed reactions form {len(subsystems)} subsystems in Recon3D model.')
        return subsystems

    def get_subsystem_from_metabolite(self, list_of_mapped_compounds):
        """get subsystems of reactions catalysed by metabolites in list_of_mapped_compounds

        Args:
            list_of_mapped_compounds (list): list of metabolites

        Returns:
            dict: dictionary of subsystems of reactions catalysed by metabolites in list_of_mapped_compounds
        """
        pathways = {}
        for met in self.metabolites[self.metabolites.fullName.isin(list_of_mapped_compounds)]['abbreviation']:
            pathways[met] = list(
                set(self.reactions[self.reactions.formula.str.contains(met)]['subsystem']))
        subsytems = []
        for v in pathways.values():
            subsytems = subsytems + v
        m_paths = {}
        for p in list(set(subsytems)):
            m_paths[p] = [k for k, v in pathways.items() if p in v]
        return m_paths

    def get_reactions(self, list_of_gene_number):
        """get reactions catalysed by genes in list_of_gene_number

        Args:
            list_of_gene_number (list): list of gene numbers

        Returns:
            dict: dictionary of reactions catalysed by genes in list_of_gene_number
        """
        m_paths = []
        for p in list_of_gene_number:
            m_paths = m_paths + [k for k, v in self.gpr.items() if p in v]
        rxns = self.reactions[self.reactions.abbreviation.isin(
            set(m_paths))]
        n_rxns = {s: list(rxns[rxns.subsystem == s]['abbreviation'])
                  for s in rxns['subsystem'].unique()}
        return n_rxns

    def get_subsystem_from_gene(self, list_of_gene_number):
        """get subsystems catalysed by genes in list_of_gene_number

        Args:
            list_of_gene_number (list): list of gene numbers

        Returns:    
            pandas.DataFrame: DataFrame of reactions catalysed by genes in list_of_gene_number
        """
        g_paths = []
        for p in list_of_gene_number:
            g_paths = g_paths + [(p, k) for k, v in self.gpr.items() if p in v]
        g_paths = pd.DataFrame(
            g_paths, columns=['gene_number', 'abbreviation'])

        for rxn in g_paths.abbreviation:
            g_paths.loc[g_paths['abbreviation'] == rxn,
                        'subsystem'] = self.reactions[self.reactions.abbreviation == rxn].subsystem.iloc[0]

        g_paths.gene_number = g_paths.gene_number.astype('float')
        g_paths = g_paths.merge(self.genes[['gene_number', 'symbol', 'uniprot_gname',
                                'description', 'ensembl_gene', 'ensembl_trans']], on='gene_number')
        return g_paths

    def get_reactions_from_metabolite(self, fullName):
        """get reactions catalysed by metabolite in fullName

        Args:   
            fullName (str): metabolite name

        Returns:
            pandas.DataFrame: DataFrame of reactions catalysed by metabolite in fullName
        """
        abb = self.metabolites.query(f'fullName == "{fullName}"')[
            'abbreviation'].iloc[0]
        rxns = self.reactions.query(f'formula.str.contains("{abb}")')
        return rxns[['abbreviation', 'description', 'formula', 'subsystem', 'ecnumber']]

    def add_gene_to_rxn(self, cobra_model, rxn_id, gene_number):
        """add gene to reaction

        Args:
            cobra_model (cobra.Model): cobra model
            rxn_id (str): reaction id
            gene_number (str): gene number

        Returns:
            cobra.Model: cobra model
        """
        cobra_model.reactions.get_by_id(
            rxn_id).gene_reaction_rule = gene_number
        return cobra_model


if __name__ == '__main__':
    recon = Recon()
