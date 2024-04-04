from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.tsa.vector_ar.vecm import VECMResults
from linearmodels.panel.results import PanelEffectsResults

from src.utils import get_stars


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _get_statsmodels_ols_summary(mod):
    df = pd.DataFrame(pd.concat([mod.params, mod.pvalues, mod.tvalues], axis=1))
    df.columns = ['coef', 'pval', 'stat']
    
    df_conf = mod.conf_int()
    df_conf.columns = ['conf_lower', 'conf_upper']
    
    endog_name = mod.model.endog_names
    df = df.join(df_conf)
    
    df_info = pd.DataFrame([], columns=df.columns)
    df_info.loc['R^2'] = list([mod.rsquared] * df.shape[1])
    df_info.loc['R^2 adj.'] = list([mod.rsquared_adj] * df.shape[1])
    df_info.loc['N'] = list([mod.nobs] * df.shape[1])
    
    df = pd.concat([df, df_info],)
        
    return df, endog_name, len(df_info)

def _get_linearmodels_pols_summary(mod):
    df = pd.DataFrame(pd.concat([mod.params, mod.pvalues, mod.tstats], axis=1))
    df.columns = ['coef', 'pval', 'stat']
    
    df_conf = mod.conf_int()
    df_conf.columns = ['conf_lower', 'conf_upper']
    
    endog_name = str(mod.model.dependent.dataframe.columns[0])
    df = df.join(df_conf)
    
    df_info = pd.DataFrame([], columns=df.columns)
    df_info.loc['R^2'] = list([mod.rsquared] * df.shape[1])
    df_info.loc['R^2 between'] = list([mod.rsquared_between] * df.shape[1])
    df_info.loc['N'] = list([mod.nobs] * df.shape[1])   
    
    df = pd.concat([df, df_info],)
    
    return df, endog_name, len(df_info)

def _get_statmodels_vecm_summary(mod, endog_index: 0):
    endog_name = mod.model.endog_names[endog_index]
    
    df = pd.DataFrame(mod.summary().tables[0].data).iloc[1:].set_index(0)
    df.index.name = ''
    df.columns = ['coef', 'stderr', 'stat', 'pval', 'conf_lower', 'conf_upper']
    df = df.astype(float)
    df = df[['coef', 'pval', 'stat', 'conf_lower', 'conf_upper']]
    
    df_info = pd.DataFrame([], columns=df.columns)
    df_info.loc['Coint. rank'] = list([mod.model.coint_rank] * df.shape[1])
    df_info.loc['N lags'] = list([mod.k_ar] * df.shape[1])
    df_info.loc['N'] = list([mod.nobs] * df.shape[1])

    
    df = pd.concat([df, df_info])
    
    return df , endog_name, len(df_info)
    

def get_statsmodels_summary(lst_mods, cols_out: str = 'print', vecm_endog_index: int = 0, seperator: str = "\n", 
                            tresh_sig: float = .05, is_filt_sig: bool = False):
    lst_dfs = []
    endog_name_save = ""
    for idx, mod in enumerate(lst_mods):
        
        if type(mod) == RegressionResultsWrapper:
            df, endog_name, n_info = _get_statsmodels_ols_summary(mod)
            
        elif type(mod) == VECMResults:
            df, endog_name, n_info = _get_statmodels_vecm_summary(mod, vecm_endog_index)
            
        elif type(mod) == PanelEffectsResults:
            df, endog_name, n_info = _get_linearmodels_pols_summary(mod)
            
        else:
            raise KeyError(f"{type(mod)} not specified")
            
        if endog_name == endog_name_save:
                endog_name += f"_{idx}"
        endog_name_save = endog_name
    
        
        # significance thresh
        df['is_significant'] = df['pval'] <= tresh_sig
        df.iloc[-n_info:, -1] = list([True] * n_info)
                
        # print output
        df['star'] = df['pval'].apply(lambda x: get_stars(x))
        df['print'] = df['coef'].round(3).astype(str)
        df.iloc[:-n_info, -1] = (
            df.coef.round(3).astype(str) + " " + df.star.astype(str) + seperator + "[" + df.stat.round(3).astype(str) + "]"
        ).iloc[:-n_info].values
        
        cols = [list(df.columns), list([endog_name] * df.shape[1])]
        df.columns = pd.MultiIndex.from_tuples(list(map(tuple, zip(*cols))))

        lst_dfs.append(df)

    out = pd.concat([df for df in lst_dfs], axis=1, join='outer').sort_index(axis=1)
    is_sig_filt = (out['is_significant'].sum(axis=1) > 0).values
    if is_filt_sig:
        out = out.loc[is_sig_filt]
        
    out = out[cols_out]
    # ensure correct order of additional info
    out = out.loc[
        [i for i in out.index if i not in ['N', 'R^2', 'Coint. rank', "R^2 between", "R^2 adj."]]
        + [i for i in out.index if i in ['N', 'R^2', 'Coint. rank', "R^2 between", "R^2 adj."]]
    ]
          

    return out


def get_fig_subplots(n_plots: int, n_cols: int = 2, figsize: tuple =(6,3.5), **kwargs):
    n_rows = int(np.ceil(n_plots/n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_rows, figsize[1] * n_cols), **kwargs)
    if n_plots == 1 and n_cols == 1:
        return fig, ax 
    else:
        ax = ax.ravel()[:n_plots]
    return fig, ax
    
def get_multiple_vecm_irfs(lst_vecms, idx_vecm: tuple = (0,1), dict_titles: dict = None, **kwargs):
    fig, axes = get_fig_subplots(len(lst_vecms), **kwargs)
    for idx, ax in enumerate(axes):
        irf = lst_vecms[idx].irf()
        
        ax.plot(irf.irfs[:, *idx_vecm], color='blue', label='irf')
        ax.fill_between(range(len(irf.irfs)), 
                        irf.irfs[:, idx_vecm[0], idx_vecm[1]] + 1.96 * irf.stderr()[:, idx_vecm[0], idx_vecm[1]],
                        irf.irfs[:, idx_vecm[0], idx_vecm[1]] - 1.96 * irf.stderr()[:, idx_vecm[0], idx_vecm[1]],
                        alpha=.3, color='grey', linestyle='dashed', label='90% conf.')

        ax.plot(list([0] * irf.irfs.shape[0]), color='black')
        
        n1, n2 = lst_vecms[idx].names[idx_vecm[1]], lst_vecms[idx].names[idx_vecm[0]]
        # print(v.names)
        if dict_titles is not None:
            try:
                n1 = dict_titles[n1]
                n2 = dict_titles[n2]
            except Exception as e:
                n1, n2 = lst_vecms[idx].names[idx_vecm[1]], lst_vecms[idx].names[idx_vecm[0]]
            
        ax.set_title(f"{n1} -> {n2}",)
        ax.legend()
    
    fig.tight_layout()
    return fig