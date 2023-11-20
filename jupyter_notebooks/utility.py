import pandas as pd
import re
import scipy.stats as stats


def eval_gpr(gpr, model, weights):
    """
    Evaluate GPRs with weights
    :param model: cobra model
    :type model: cobra.Model
    :param weights: weights
    :type weights: pd.DataFrame
    :return: values
    :rtype: pd.Series
    """
    values = {}
    for i in gpr.keys():
        try:
            values[i] = eval(replace_and(gpr[i], weights))
        except SyntaxError:
            try:
                values[i] = eval(replace_and(gpr[i], weights))
            except SyntaxError:
                values[i] = 1.0

    for r in model.reactions.abbreviation:
        if r not in values.keys():
            values[r] = 1.0
    values = pd.Series(values)
    return values


def replace_and(gpr1, w1):
    re_float = '\d*\.\d*'
    re_and = re.compile(f'{re_float}(?: and {re_float})+')

    for gene in re.compile(re_float).findall(gpr1):
        if gene in w1.index:
            gpr1 = gpr1.replace(gene, str(w1[w1.index == gene].iloc[0]))
        else:
            gpr1 = gpr1.replace(gene, str(1.0))
    gpr = gpr1.replace('or', '+')
    hits = re_and.findall(gpr)
    for hit in hits:
        gids = hit.split('and')
        gids = [x.strip() for x in gids]
        gids = ', '.join(gids)
        new = f'stats.gmean([{gids}])'
        gpr = gpr.replace('('+hit+')', '('+new+')')
    return gpr


def get_met_id(met_name, model):
    met_id = model.metabolites.query(f'fullName == "{met_name}"')[
        'abbreviation'].iloc[0]
    return met_id


def get_rxns_with_weights(met_id, model, W):
    try:
        list_of_rxns = list(set([[r.id for r in m.reactions]
                            for m in model.model.metabolites if m.id.startswith(met_id)][0]))
    except IndexError:
        met_id = get_met_id(met_id, model)
        list_of_rxns = list(set([[r.id for r in m.reactions]
                            for m in model.model.metabolites if m.id.startswith(met_id)][0]))

    _W = W[W.index.isin(list_of_rxns)]
    rxn = model.model.reactions
    _df = pd.concat((pd.Series({i: rxn.get_by_id(i).reaction for i in _W.index}),
                     pd.Series(
                         {i: rxn.get_by_id(i).subsystem for i in _W.index}),
                     pd.Series({i: str(rxn.get_by_id(i).gpr)
                               for i in _W.index}),
                     _W.mean(axis=1)), axis=1)
    _df = _df.rename(columns={0: 'reaction', 2: 'gpr',
                     1: 'subsystem', 3: 'weight'})
    return _df


def get_rxns_from_subsystem(W, list_of_rxns, model):
    _W = W[W.index.isin(list_of_rxns)]
    rxn = model.model.reactions
    _df = pd.concat((pd.Series({i: rxn.get_by_id(i).reaction for i in _W.index}),
                     pd.Series(
                         {i: rxn.get_by_id(i).subsystem for i in _W.index}),
                     pd.Series({i: str(rxn.get_by_id(i).gpr)
                               for i in _W.index}),
                     _W.mean(axis=1)), axis=1)
    _df = _df.rename(columns={0: 'reaction', 2: 'gpr',
                     1: 'subsystem', 3: 'weight'})
    return _df


def get_genenumbers_from_rxn(gpr_dict):
    genes = []
    re_float = '\d*\.\d*'
    for key in gpr_dict.keys():
        genes = genes + re.compile(re_float).findall(gpr_dict[key])
    genes = list(set(genes))
    return genes


def get_genes_from_metabolite(model, met):
    _rxns = model.get_reactions_from_metabolite(met)
    _rxns_gpr = {k: v for k, v in model.gpr.items(
    ) if k in _rxns.abbreviation.values}
    _rxns_genes = get_genenumbers_from_rxn(_rxns_gpr)
    return _rxns_genes
