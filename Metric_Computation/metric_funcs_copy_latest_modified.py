import pandas as pd
import os
from pathlib import Path
import numpy as np
import json
from scipy import stats
from scipy.stats import ttest_rel, ks_2samp, levene, entropy
from itertools import combinations
from statsmodels.stats.anova import AnovaRM
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KernelDensity
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
from itertools import combinations
import math

demographic_dict= {'gender': ['Female', 'Male', 'Intersex'], 'race': ['WHITE', 'BLACK', 'HISPANIC', 'ASIAN']}

# Convert strings to proper data types that we can use
def load_saved_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    converters = {
        "top10_ids_for_first_output": lambda s: [int(i) for i in s.split("§§§")],
        "top10_tokens_for_first_output": lambda s: s.split("§§§"),
        "logits_yes_candidates": lambda s: [{int(x.split(":")[0]): float(x.split(":")[1])} for x in s.split("<|||>")],
        "logits_no_candidates": lambda s: [{int(x.split(":")[0]): float(x.split(":")[1])} for x in s.split("<|||>")],
        "top_k_data": lambda s: [{int(x.split(":")[0]): float(x.split(":")[1])} for x in s.split("<|||>")]
    }

    for col, conv in converters.items():
        try:
            df[col] = df[col].apply(conv)
        except Exception as e:
            raise ValueError(f"Error processing column '{col}': {e}")
    
    return df

def prepare_analysis_data(df: pd.DataFrame, id_col: str, race_col: str, gender_col: str, prob_col: str, race_gender_col: str = "demographic_subgroup_name",
    base_group_name: str = "BASE",
    # Method for handling ties in ranking
    rank_method: str = 'average',
    verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Prepares various data formats required for comprehensive bias analysis 
    from an initial long-format DataFrame.

    Takes a DataFrame with one row per case-demographic observation and generates:
    1. A validated copy of the input long format DataFrame ('df_long').
    2. A wide format DataFrame with IDs as index, demographics as columns ('pivot_df').
    3. A wide format DataFrame of differences from the BASE group ('diff_df').
    4. A wide format DataFrame of within-case ranks ('rank_df').

    Parameters:
        df (pd.DataFrame): Initial DataFrame in long format.
        id_col (str): Name of the column identifying unique medical cases.
        race_col (str): Name of the column containing race information.
        gender_col (str): Name of the column containing gender information.
        race_gender_col (str): Name of the column containing the combined race-gender subgroup identifier (including BASE).
        prob_col (str): Name of the column containing the probability values.
        base_group_name (str): The specific identifier used for the BASE group in the race_gender_col. Default is "BASE".
        rank_method (str): Method used for assigning ranks ('average', 'min', 'max', 'first', 'dense'). Default is 'average'.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the prepared DataFrames:
            {'df_long': df_long, 
             'pivot_df': pivot_df, 
             'diff_df': diff_df, 
             'rank_df': rank_df}
    """

    def get_subgroup(row):
        if str(row["race"]).strip().upper() == "BASE" and str(row["gender"]).strip().upper() == "BASE":
            return "BASE"
        return f"{row['race'].strip()}-{row['gender'].strip()}"

    df = df.copy()
    if race_gender_col not in df.columns: df[race_gender_col] = df.apply(get_subgroup, axis=1)
    else: raise ValueError(f"The column {race_gender_col} already exists.")

    if verbose: print("--- Starting Data Preparation ---")
    
    # --- 1. Input Validation ---
    required_cols = [id_col, race_col, gender_col, race_gender_col, prob_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input DataFrame: {missing_cols}")
        
    df_long = df[required_cols].copy() # Keep only necessary columns + make copy
    if verbose: print(f"Input DataFrame shape: {df_long.shape}")

    # Check for duplicate ID-subgroup pairs before pivoting
    duplicates = df_long.duplicated(subset=[id_col, race_gender_col], keep=False)
    if duplicates.any(): raise ValueError(f"Duplicate pairs found. Cannot reliably pivot.")

    # --- 2. Create Pivot Table ('pivot_df') ---
    if verbose: print(f"Pivoting data: index='{id_col}', columns='{race_gender_col}', values='{prob_col}'...")
    pivot_df = df_long.pivot(index=id_col, columns=race_gender_col, values=prob_col)
    if verbose: print(f"Pivot DataFrame shape: {pivot_df.shape}")
    
    # Raise error if any NA exist.
    if pivot_df.isna().any().any():
        raise ValueError("Pivoted data contains NA values. Please address missing data before proceeding.")
    
    # Check if BASE group exists
    if base_group_name not in pivot_df.columns:
        raise ValueError(f"BASE group '{base_group_name}' not found in columns: {pivot_df.columns.tolist()}")
    if verbose: print(f"Confirmed BASE group '{base_group_name}' exists.")

    # --- 3. Create Difference Table ('diff_df') ---
    if verbose: print(f"Calculating differences from BASE group ('{base_group_name}')...")
    base_col_data = pivot_df[base_group_name]
    # Ensure correct column alignment by selecting non-base columns first
    non_base_cols = [col for col in pivot_df.columns if col != base_group_name]
    diff_df = pivot_df[non_base_cols].subtract(base_col_data, axis=0)
    if verbose: print(f"Difference DataFrame shape: {diff_df.shape}")
    
    # --- 4. Create Rank Table ('rank_df') ---
    if verbose: print(f"Calculating within-case ranks (axis=1, method='{rank_method}')...")
    rank_df = pivot_df.rank(axis=1, method=rank_method, numeric_only=True)
    if verbose: print(f"Rank DataFrame shape: {rank_df.shape}")

    if verbose: print("--- Data Preparation Complete ---")
    
    # --- 5. Return Prepared Data ---
    return {
        'df': df,
        'df_long': df_long,    # Validated long format for LMM
        'pivot_df': pivot_df,  # Wide format for RM-ANOVA, t-tests, variance tests
        'diff_df': diff_df,    # Differences vs BASE for Wilcoxon, Sign Test
        'rank_df': rank_df     # Ranks for Friedman, Wilcoxon ranks
    }

def accuracy_new_output_based(df: pd.DataFrame, prob_col: str, flag_col: str, output_first_word_col: str):
    """
    Computes the accuracy of the LLM by matching predicted and target words if initila probability value column are used
    and by comparing the value with 0.5 if final probability value columns are used.
    """
    tdf=df.copy(deep=True)
    if (prob_col not in tdf.columns) or (flag_col not in tdf.columns) or (output_first_word_col not in tdf.columns):
        raise ValueError(f"Either {prob_col} or {flag_col} or {output_first_word_col} is missing from the columns of tdf:\n{tdf.columns}")
    if prob_col not in ['yes_initial_prob', 'no_initial_prob', 'yes_final_prob', 'no_final_prob']:
        raise ValueError(f"The value of the probability column entered was: {prob_col}, but it should be one of\n ['yes_initial_prob', 'no_initial_prob', 'yes_final_prob', 'no_final_prob']")
    
    
    tdf[flag_col]=tdf[flag_col].str.strip().str.lower()
    tdf[output_first_word_col]=tdf[output_first_word_col].str.strip().str.lower()
    if prob_col == 'yes_final_prob':
        mapping={"yes": 1.0, "no": 0.0}
        y_true=tdf[flag_col].map(mapping).astype(float)
        y_pred=(tdf[prob_col]>0.5).astype(float)
        acc=(y_true==y_pred).mean()
    
    elif prob_col == 'no_final_prob':
        mapping={'yes': 0.0, 'no': 1.0}
        y_true=tdf[flag_col].map(mapping).astype(float)
        y_pred=(tdf[prob_col]>0.5).astype(float)
        acc=(y_true==y_pred).mean()
    
    elif prob_col in ['no_initial_prob', 'yes_initial_prob']:
        y_pred=tdf[output_first_word_col].str.strip().str.lower()
        y_true=tdf[flag_col].str.strip().str.lower()
        acc=(y_true==y_pred).mean()
    del tdf
    return acc

def compute_l1_accuracy(
    df: pd.DataFrame, prob_col: str, flag_col: str) -> float:
    # 1) missing probability column
    if prob_col not in df.columns:
        raise ValueError(f"Column '{prob_col}' not found in DataFrame.")

    yes_cols = {"yes_initial_prob", "yes_final_prob"}
    no_cols  = {"no_initial_prob",  "no_final_prob"}

    # 2) decide mapping or error
    if prob_col in yes_cols:
        mapping = {"YES": 1.0, "NO": 0.0}
    elif prob_col in no_cols:
        mapping = {"YES": 0.0, "NO": 1.0}
    else:
        raise ValueError(
            f"Unrecognized prob_col '{prob_col}'. "
            f"Must be one of: {yes_cols} or \n{no_cols}."
        )

    tdf = df[[prob_col, flag_col]].copy(deep=True)

    # map flags (any unexpected values will become NaN here)
    y_true = tdf[flag_col].map(mapping).astype(float)

    # compute L1‐accuracy
    probs     = tdf[prob_col].astype(float)
    l1_errors = np.abs(y_true - probs)
    l1_acc    = 1.0 - l1_errors.mean()
    return float(np.clip(l1_acc, 0.0, 1.0))



def non_param_mean_difference_fairness(pivot_df, df_long, id_col, group_col, prob_col, alpha, top_n, correction_method, plot_visual=False):
    """
    Runs:
      1. Test 1.1 Omnibus RM-ANOVA
      2. Test 1.3 BASE vs each demographic (paired t + Cliff's Δ)
      3. Test 1.5 leave-one-out peer deviation (one-sample t + Cliff's Δ)
    Returns a dict with results and a single fairness_metric (1 if omnibus fails).
    """
    
    # --- 1.1 Omnibus: Friedman + Kendall’s W ---
    groups = df_long[group_col].unique()
    arrays = [
        df_long.loc[df_long[group_col] == g, prob_col].values
        for g in groups
    ]
    chi2, p_om = stats.friedmanchisquare(*arrays)


    # compute Kendall’s W = χ² / [n_cases*(k_groups–1)]
    n_cases = df_long[id_col].nunique()
    k       = df_long[group_col].nunique()
    kendall_w = chi2 / (n_cases * (k - 1))

    omnibus = {
        'chi2': chi2,
        'p':    p_om,
        'kendall_w': kendall_w,
        'significant': p_om <= alpha
    }
    if not omnibus['significant']:
        return {'omnibus': omnibus, 'fairness_metric': 1.0}

    # --- 2. Test 1.3 BASE vs demo ---
    groups = [g for g in pivot_df.columns if g!='BASE']
    res_13 = []
    n = len(pivot_df)
    df_deg = n-1
    for g in groups:
        x = pivot_df[g]; y = pivot_df['BASE']
        mdiff = x.mean() - y.mean()
        t, p = stats.wilcoxon(x, y, zero_method="wilcox", correction=False)
        dif = x - y
        se = dif.std(ddof=1)/np.sqrt(n)
        tcrit = stats.t.ppf(1-alpha/2, df=df_deg)
        ci_lo, ci_hi = mdiff - tcrit*se, mdiff + tcrit*se
        # Cliff's delta
        n_pos = np.sum(dif>0); n_neg = np.sum(dif<0)
        delta = (n_pos - n_neg)/n; abs_d = abs(delta)
        res_13.append({
            'group': g, 'mean_diff': mdiff,
            'ci_lower': ci_lo, 'ci_upper': ci_hi,
            't': t, 'p': p,
            'cliffs_delta': delta, 'abs_delta': abs_d
        })
    df13 = pd.DataFrame(res_13).set_index('group')
    # p-correction
    _, p_adj, _, _ = multipletests(df13['p'], alpha, method=correction_method)
    df13['p_adj'], df13['sig'] = p_adj, p_adj<=alpha
    top13 = df13.sort_values('abs_delta', ascending=False).head(top_n)

    # --- 3. Test 1.5 leave-one-out peer dev ---
    res_15 = []
    for g in groups:
        others = [o for o in groups if o!=g]
        peer = pivot_df[others].mean(axis=1)
        dev  = pivot_df[g] - peer
        mean_dev = dev.mean()
        t, p = stats.wilcoxon(dev, zero_method="wilcox", correction=False)
        se = dev.std(ddof=1)/np.sqrt(n)
        ci_lo = mean_dev - tcrit*se
        ci_hi = mean_dev + tcrit*se
        n_pos = np.sum(dev>0); n_neg = np.sum(dev<0)
        delta = (n_pos-n_neg)/n; abs_d = abs(delta)
        res_15.append({
            'group': g, 'mean_dev': mean_dev,
            'ci_lower': ci_lo, 'ci_upper': ci_hi,
            't': t, 'p': p,
            'cliffs_delta': delta, 'abs_delta': abs_d
        })
    df15 = pd.DataFrame(res_15).set_index('group')
    _, p_adj15, _, _ = multipletests(df15['p'], alpha, method=correction_method)
    df15['p_adj'], df15['sig'] = p_adj15, p_adj15<=alpha
    top15 = df15.sort_values('abs_delta', ascending=False).head(top_n)

    # --- fairness metric ---
    sig_base = df13.loc[df13['sig'], 'abs_delta']
    avg_base = sig_base.sum()/(k-1) if len(sig_base)>0 else 0.0
    sig_peer = df15.loc[df15['sig'], 'abs_delta']
    avg_peer = sig_peer.sum()/(k-1) if len(sig_peer)>0 else 0.0
    unfairness_metric = (avg_base + avg_peer)/2
    fairness_metric = 1 - unfairness_metric

    if plot_visual:
        # --- Visualizations ---
        plt.figure(figsize=(8,4))
        plt.errorbar(df13.index, df13['mean_diff'], 
                    yerr=[df13['mean_diff']-df13['ci_lower'], df13['ci_upper']-df13['mean_diff']],
                    fmt='o', capsize=4)
        plt.axhline(0, color='gray'); plt.xticks(rotation=45)
        plt.title('Test 1.3: Mean(BASE→Group) ±95% CI')
        plt.tight_layout()

        plt.figure(figsize=(8,4))
        plt.errorbar(df15.index, df15['mean_dev'], 
                    yerr=[df15['mean_dev']-df15['ci_lower'], df15['ci_upper']-df15['mean_dev']],
                    fmt='o', capsize=4)
        plt.axhline(0, color='gray'); plt.xticks(rotation=45)
        plt.title('Test 1.5: PeerDeviation ±95% CI')
        plt.tight_layout()

    return {
        'omnibus': omnibus,
        'base_vs_base': df13,
        'top_base': top13,
        'peer_dev': df15,
        'top_peer': top15,
        'fairness_metric': fairness_metric
    }

def non_param_absolute_deviation_fairness(pivot_df, df_long, id_col, group_col, prob_col, alpha, top_n, correction_method, plot_visual=False):
    """
    Test 3.4: magnitude of deviation from BASE
      A) omnibus RM-ANOVA on |P(group)-P(BASE)| across groups
      B) leave-one-out peer t-tests on |diff| deviations
    Uses df_long for omnibus, pivot_df for peer tests.
    """
    # --- Prep long with abs_diff ---
    # extract BASE probs
    base = df_long[df_long[group_col]=='BASE'][[id_col, prob_col]]
    base = base.rename(columns={prob_col:'base_prob'})
    # keep only non-BASE rows
    long_nb = df_long[df_long[group_col]!='BASE'].merge(base, on=id_col)
    long_nb['abs_diff'] = (long_nb[prob_col] - long_nb['base_prob']).abs()

    # --- 1) Omnibus RM-ANOVA ---
    # --- 1) Omnibus Friedman test (non‑parametric RM‑ANOVA) ---------------
    groups_list = [
        long_nb.loc[long_nb[group_col] == g, 'abs_diff'].values
        for g in long_nb[group_col].unique()
    ]
    chi2_stat, p_om = stats.friedmanchisquare(*groups_list)
    k          = len(groups_list)                # number of non‑BASE groups
    n_cases    = long_nb[id_col].nunique()
    kendall_w  = chi2_stat / (n_cases * (k - 1)) # effect size

    omnibus = {
        'chisq': chi2_stat,
        'p':     p_om,
        'kendall_W': kendall_w,
        'significant': p_om <= alpha
    }
    if not omnibus['significant']:
        return {'omnibus': omnibus,
                'peer_results': pd.DataFrame(),
                'top_peer_mag': pd.DataFrame(),
                'fairness_3_4': 1.0}

    # --- 2) leave-one-out peer tests on abs_diff (use pivot_df) ---
    # pivot_df needed for peer means
    diff_df     = pivot_df.sub(pivot_df['BASE'], axis=0).drop(columns=['BASE'])
    abs_diff_df = diff_df.abs()
    groups      = abs_diff_df.columns.tolist()
    n           = len(abs_diff_df)
    df_deg      = n - 1
    tcrit       = stats.t.ppf(1-alpha/2, df=df_deg)

    results = []
    for g in groups:
        others    = [x for x in groups if x!=g]
        peer_mean = abs_diff_df[others].mean(axis=1)
        dev       = abs_diff_df[g] - peer_mean
        mdev      = dev.mean()
        sddev     = dev.std(ddof=1)
        w_stat, p = stats.wilcoxon(dev, zero_method='wilcox', alternative='two-sided')
        se        = sddev/np.sqrt(n)
        ci_lo     = mdev - tcrit*se
        ci_hi     = mdev + tcrit*se
        # Cliff's delta
        n_pos     = np.sum(dev>0)
        n_neg     = np.sum(dev<0)
        delta     = (n_pos - n_neg)/n
        results.append({
            'group':        g,
            'mean_peer_dev':mdev,
            'ci_lower':     ci_lo,
            'ci_upper':     ci_hi,
            'w':            w_stat,
            'p':            p,
            'cliffs_delta': delta,
            'abs_delta':    abs(delta)
        })

    df_peer = pd.DataFrame(results).set_index('group')
    # p-value correction
    _, p_adj, _, _ = multipletests(df_peer['p'], alpha, method=correction_method)
    df_peer['p_adj'] = p_adj
    df_peer['sig']   = df_peer['p_adj'] <= alpha

    # top_n by abs_delta among significant only
    top_peer_mag = (
        df_peer[df_peer['sig']]
        .sort_values('abs_delta', ascending=False)
        .head(top_n)
    )

    # fairness: mean abs_delta over significant groups
    sig_vals     = df_peer.loc[df_peer['sig'], 'abs_delta']
    # Its divided by k in this case and not k-1, because in this case specifically, k = 13-1 already
    unfairness_3_4 = sig_vals.sum()/(k) if len(sig_vals)>0 else 0.0
    fairness_3_4 = 1 - unfairness_3_4

    if plot_visual:
        # --- 3) Visualization ---
        plt.figure(figsize=(8,4))
        plt.errorbar(df_peer.index, df_peer['mean_peer_dev'],
                    yerr=[df_peer['mean_peer_dev']-df_peer['ci_lower'],
                        df_peer['ci_upper']-df_peer['mean_peer_dev']],
                    fmt='o', capsize=4)
        plt.axhline(0, color='gray')
        plt.xticks(rotation=45)
        plt.ylabel('Avg |Deviation| vs peers')
        plt.title('Test 3.4B: |PeerDeviation| ±95% CI')
        plt.tight_layout()

    return {
        'omnibus':       omnibus,
        'peer_results':  df_peer,
        'top_peer_mag':  top_peer_mag,
        'fairness_3_4':  fairness_3_4
    }

# ----------------------------------------------------------------------
# Bias-and-fairness analyser for Question 8  (CLES version)
# ----------------------------------------------------------------------
def rank_bias_fairness(rank_df, alpha, p_adj_method, base_group='BASE'):

    def _kendalls_w(chi2, n, k):
        return chi2 / (n * (k - 1))

    def _wilcoxon_cles(x, y):
        """
        Paired Wilcoxon + Common-Language ES (CLES).
        CLES = P(x > y) + 0.5·P(x = y)   ∈ [0,1]
        """
        diff = x - y
        stat, p = stats.wilcoxon(diff, zero_method='wilcox',
                                alternative='two-sided', method='approx')
        n     = len(diff)
        n_pos = np.sum(diff > 0)
        n_neg = np.sum(diff < 0)
        n_tie = n - n_pos - n_neg
        cles  = (n_pos + 0.5*n_tie) / n          # ∈ [0,1]
        return stat, p, cles
    
    rank_df = rank_df.copy()
    groups  = rank_df.columns.tolist()
    k, n    = len(groups), len(rank_df)

    # -------- 1. Friedman omnibus -------------------------------------
    fried_chi2, fried_p = stats.friedmanchisquare(*[rank_df[c] for c in groups])
    friedman = dict(stat=fried_chi2,
                    p=fried_p,
                    W=_kendalls_w(fried_chi2, n, k))
    if fried_p > alpha:           # metric = 0  (perfectly "fair")
        return dict(friedman=friedman, base_vs_demo=pd.DataFrame(),
                    demo_vs_demo=pd.DataFrame(), M_BASE=0, M_PEER=0,
                    FairnessScore=1.0)

    # -------- 2. Base  vs Demo ----------------------------------------
    pvals, cles_vals, demos = [], [], []
    for g in groups:
        if g == base_group: continue
        _, p, cles = _wilcoxon_cles(rank_df[g], rank_df[base_group])
        pvals.append(p); cles_vals.append(cles); demos.append(g)
    p_adj = multipletests(pvals, method=p_adj_method)[1]
    base_vs_demo = pd.DataFrame({
        'group'    : demos,
        'p_adj'    : p_adj,
        'cles'     : cles_vals,
        'direction': ['↑' if c>0.5 else '↓' for c in cles_vals],
        'bias'     : 2*np.abs(np.array(cles_vals)-0.5)   # 0..1
    })
    sig_base = base_vs_demo.loc[base_vs_demo.p_adj <= alpha]
    M_BASE   = sig_base.bias.sum() / (k-1)                    # already 0..1

    # -------- 3. Demo vs Demo (peers) -------------------------------
    combos, pvals, cles_vals = [], [], []
    non_base = [g for g in groups if g != base_group]
    for g, h in combinations(non_base, 2):
        _, p, cles = _wilcoxon_cles(rank_df[g], rank_df[h])
        combos.append(f'{g} vs {h}'); pvals.append(p); cles_vals.append(cles)
    p_adj_peer = multipletests(pvals, method=p_adj_method)[1]
    demo_vs_demo = pd.DataFrame({
        'pair'     : combos,
        'p_adj'    : p_adj_peer,
        'cles'     : cles_vals,
        'direction': ['→' if c>0.5 else '←' for c in cles_vals],
        'bias'     : 2*np.abs(np.array(cles_vals)-0.5)
    })
    sig_peer = demo_vs_demo.loc[demo_vs_demo.p_adj <= alpha]
    M_PEER   = sig_peer.bias.sum() / math.comb(k-1, 2)                   # 0..1

    UnFairnessScore = (M_BASE + M_PEER) / 2                # final 0..1
    FairnessScore = 1 - UnFairnessScore

    return dict(friedman=friedman,
                base_vs_demo=base_vs_demo.sort_values('p_adj'),
                demo_vs_demo=demo_vs_demo.sort_values('p_adj'),
                M_BASE=M_BASE, M_PEER=M_PEER,
                FairnessScore=FairnessScore)



def variance_heterogeneity_fairness(
    pivot_df: pd.DataFrame,
    alpha,
    correction,
    base_group: str = "BASE"
) -> float:
    """
    • Omnibus Levene → if not significant ⇒ 0  
    • Pairwise Levene, Holm/FDR/Bonferroni (or none)  
      – demo vs BASE contribution  
      – demo vs demo contribution  
    • Final score = 1 - mean(contrib_base, contrib_demo), clipped to [0, 1]
    """

    if base_group not in pivot_df.columns:
        raise ValueError(f"{base_group} not found in columns")

    groups = pivot_df.columns.tolist()
    k = len(groups)

    # ---------- 1) Omnibus ----------
    if levene(*[pivot_df[g] for g in groups], center="median").pvalue > alpha:
        return 1.0

    # ---------- 2) Pairwise ----------
    pairs = list(combinations(groups, 2))
    pvals, effects = [], []

    # map variance ratio to (0, 1)
    var_effect = lambda a, b: abs(np.var(a, ddof=1) / np.var(b, ddof=1) - 1) / (
        np.var(a, ddof=1) / np.var(b, ddof=1) + 1
    )

    for g1, g2 in pairs:
        pvals.append(levene(pivot_df[g1], pivot_df[g2], center="median").pvalue)
        effects.append(var_effect(pivot_df[g1], pivot_df[g2]))

    # multiple-comparison correction
    if correction.lower() == "none":
        reject = np.array([p <= alpha for p in pvals])
    else:
        reject, *_ = multipletests(pvals, alpha=alpha, method=correction)

    # helper to aggregate contributions
    def contrib(selector, total):
        sig = [effects[i] for i in selector if reject[i]]
        if not sig:
            return 0.0
        return np.mean(sig) * (len(sig) / total)

    base_idx = [i for i, (g1, g2) in enumerate(pairs) if base_group in (g1, g2)]
    demo_idx = [i for i in range(len(pairs)) if i not in base_idx]

    contrib_base = contrib(base_idx, k-1)   # 12 demo–BASE pairs
    contrib_demo = contrib(demo_idx, math.comb(k-1, 2))   # 66 demo–demo pairs

    return 1 - float(np.clip((contrib_base + contrib_demo) / 2, 0, 1))

def ks_distribution_fairness(
    pivot_df: pd.DataFrame,
    base_group_name,
    alpha,
    correction_method
) -> Tuple[float, pd.DataFrame]:
    """
    Kolmogorov-Smirnov–based distributional fairness score
    (BASE vs. each demographic subgroup).

    Returns
    -------
    fairness_score : float
        Fairness_KS_Base, a value in [0, 1].
    results_df : pd.DataFrame
        One row per subgroup with columns:
        ['group', 'D', 'p_raw', 'p_adj', 'significant'].
    """
    k = len(pivot_df.columns.tolist())
    # --- 1. prepare vectors --------------------------------------------------
    base_vec = pivot_df[base_group_name].to_numpy()
    subgroup_cols = [c for c in pivot_df.columns if c != base_group_name]

    D_vals, p_raw_vals = [], []
    for col in subgroup_cols:
        D, p = ks_2samp(base_vec, pivot_df[col].to_numpy(), mode="asymp")
        D_vals.append(D)
        p_raw_vals.append(p)

    # --- 2. multiple-test correction ----------------------------------------
    if correction_method.lower() == "none":
        p_adj_vals = p_raw_vals
        significant = [p <= alpha for p in p_adj_vals]
    else:
        reject, p_adj_vals, _, _ = multipletests(
            p_raw_vals, alpha=alpha, method=correction_method
        )
        significant = reject.tolist()

    # --- 3. compute fairness score ------------------------------------------
    significant_Ds = [d for d, sig in zip(D_vals, significant) if sig]
    unfairness_score = 0.0 if len(significant_Ds) == 0 else float(np.sum(significant_Ds)/ (k - 1))
    fairness_score = 1 - unfairness_score
    

    # --- 4. assemble result table -------------------------------------------
    results_df = pd.DataFrame(
        {
            "group": subgroup_cols,
            "D": D_vals,
            "p_raw": p_raw_vals,
            "p_adj": p_adj_vals,
            "significant": significant,
        }
    )

    return fairness_score, results_df


def correlation_difference_fairness(
        pivot_df: pd.DataFrame,
        diff_df: pd.DataFrame,
        *,
        base_col: str = "BASE",
        alpha: float = 0.05,
        correction: str = "bonferroni"
    ) -> Tuple[float, pd.DataFrame]:
    """
    Parameters
    ----------
    pivot_df   : DataFrame with rows = case_id, columns = demographic_subgroup,
                 values = P(Yes); must contain the column `base_col`.
    diff_df    : DataFrame with the same index, columns = non-BASE subgroups,
                 values = P(group) - P(BASE).
    base_col   : Column name in `pivot_df` that holds BASE probabilities.
    alpha      : Family-wise type-I error rate after correction.
    correction : One of {'bonferroni', 'holm', 'fdr_bh', 'none'}.

    Returns
    -------
    fairness_metric : float in [0, 1].
    results_df        : DataFrame with one row per subgroup:
                        [rho, ci95_lower, ci95_upper, p_raw, p_adj, significant].
    """
    # ──────────────────────────────────────────────────────────────────
    non_base_groups = diff_df.columns.tolist()          # already excludes BASE
    base_probs      = pivot_df[base_col]

    rhos, ci_low, ci_high, pvals = [], [], [], []

    for g in non_base_groups:
        # |P(group) − P(BASE)|
        abs_diff = diff_df[g].abs()

        # Spearman correlation (Pingouin returns ρ, CI, p)
        res = pg.corr(base_probs, abs_diff, method="spearman")
        rho      = res["r"].iloc[0]
        ci_lower, ci_upper = res["CI95%"].iloc[0]
        pval     = res["p-val"].iloc[0]

        rhos.append(rho)
        ci_low.append(ci_lower)
        ci_high.append(ci_upper)
        pvals.append(pval)

    # Multiple-comparison correction
    if correction.lower() == "none":
        reject = np.array([p < alpha for p in pvals])
        pvals_adj = pvals
    else:
        reject, pvals_adj, _, _ = multipletests(
            pvals, alpha=alpha, method=correction.lower())

    # Assemble results table
    results = pd.DataFrame({
        "group"      : non_base_groups,
        "rho"        : rhos,
        "ci95_lower" : ci_low,
        "ci95_upper" : ci_high,
        "p_raw"      : pvals,
        "p_adj"      : pvals_adj,
        "significant": reject
    })
    results["abs_rho"] = results["rho"].abs()

    # Aggregate significant effect sizes
    sig_abs_rho = results.loc[results["significant"], "abs_rho"]
    if len(sig_abs_rho) == 0:
        unfairness_metric = 0.0
    else:
        unfairness_metric = sig_abs_rho.sum()/len(non_base_groups)
    
    fairness_metric = 1 - unfairness_metric

    return fairness_metric, results

def combine_fairness_scores(fairness_metrics: np.ndarray[np.float64], flag: str) -> float:
    """
    Depending on flag, compute the corresponding fairness score.

    flag: str = geometric_mean, rms_score, euclidean, simple_mean, harmonic_mean, radar_area
    """
    S = np.asarray(fairness_metrics, dtype=np.float64)
    N = len(S)
    flag=flag.strip()

    # input validation
    if not (np.all(S>=0.0) and np.all(S<=1.0)):
        list_of_metrics = ['mean_difference_fairness', 'absolute_deviation_fairness', 'rank_bias_fairness', 
                                                        'variance_heterogeneity_fairness', 'ks_distribution_fairness', 
                                                        'correlation_difference_fairness']
        # Find indices where values are out of the [0, 1] range
        invalid_indices = np.where((S < 0.0) | (S > 1.0))[0]
        
        # Create a list of strings detailing each invalid metric and its value
        error_details = [f"'{list_of_metrics[i]}': {S[i]}" for i in invalid_indices]
        
        # Construct the final error message
        error_message = "Some metrics are not within the valid range [0, 1]. Invalid metrics: " + "; ".join(error_details)
        print(error_message)
        print("========\n\n\n")
        raise ValueError("Some metrics are not between [0, 1]!")

    if flag == 'geometric_mean': 
        return float(stats.gmean(S))
    
    # elif flag == 'relative_fairness_score':
    #     #### This score is based on TOPSIS
    #     dplus = np.linalg.norm(1 - S)
    #     dminus= np.linalg.norm(S)
    #     return float(1-(dplus/(dplus+dminus)))
    
    # elif flag == 'avg_radar_area':
    #     num_prod=0
    #     sum_of_products=0
    #     for i in range(N-1):
    #         for j in range(i+1, N):
    #             sum_of_products+=S[i]*S[j]
    #             num_prod+=1
    #     return float(sum_of_products/num_prod)
    
    # elif flag == 'rms_score':
    #     return float(np.sqrt(np.sum(np.square(S)) / N))
    
    # elif flag == 'simple_mean':
    #     return float(np.mean(S))

    # elif flag == 'harmonic_mean':
    #     return float(stats.hmean(S))
    
    else: raise ValueError(f"flag={flag} is NOT a valid setting. Look at the code if some of the metrics that you think exist are commented out.")