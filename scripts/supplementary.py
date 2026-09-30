"""Reproduce Tables A--H in S1 Appendix from archived labels and TRAIT statement families.

Run after Main_Analysis.ipynb, or use reproduce.py for the complete workflow.
The analysis reads item-level data from Data/ and writes CSVs to Results/Supplementary/.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Results' / 'Supplementary'
TRAITS = ['Machiavellianism', 'Narcissism', 'Psychopathy']
# Fixed model order for reproducible paired-bootstrap random-number draws.
MODELS = {
    'DeepSeek-V3.2': 'DeepseekV3_2', 'Gemma 3': 'Gemma3',
    'GPT-4.1': 'GPT4_1', 'Grok-4.3': 'Grok4_3',
    'Llama 3.3': 'Llama3_3', 'Mistral Large 3': 'Mistral_Large3',
    'Qwen 2.5': 'Qwen2_5',
}
CONDITIONS = ['SA', 'FG_job', 'FG_legal', 'FB_job', 'FB_legal',
              'FB_job_expl', 'FB_legal_expl']
N_BOOTSTRAP = 10000
PAIRED_SEED = 20260810
SCORE_SEED = 20260819


def load_data():
    trait = pd.DataFrame(json.loads((ROOT / 'Data/trait_items.json').read_text()))
    if len(trait) != 3000 or trait.idx.duplicated().any():
        raise ValueError('Expected 3,000 unique TRAIT items')
    if set(trait.personality) != set(TRAITS) or trait.statement.isna().any():
        raise ValueError('Invalid trait labels or missing family descriptions')
    trait['cluster_id'] = trait.personality + '||' + trait.statement
    for t, g in trait.groupby('personality'):
        sizes = g.groupby('cluster_id').size()
        if len(sizes) != 200 or not sizes.eq(5).all():
            raise ValueError(f'{t}: expected 200 families of five items')
    frames = []
    for model, folder in MODELS.items():
        for cond in CONDITIONS:
            rel = ('explicit/' + cond.removesuffix('_expl')) if cond.endswith('_expl') else cond
            path = ROOT / 'Data' / folder / (rel + '.csv')
            f = pd.read_csv(path)
            if len(f) != 3000 or f.idx.duplicated().any() or set(f.idx) != set(trait.idx):
                raise ValueError(f'Invalid item coverage: {path}')
            if not f.choice_type.isin(['High', 'Low', 'Unknown']).all():
                raise ValueError(f'Invalid classification: {path}')
            f = f.merge(trait[['idx', 'personality', 'cluster_id']], on='idx',
                        suffixes=('', '_trait'), validate='one_to_one')
            if not f.personality.eq(f.personality_trait).all():
                raise ValueError(f'Trait mismatch: {path}')
            frames.append(f[['idx', 'personality', 'cluster_id', 'choice_type']]
                          .assign(model=model, condition=cond))
    data = pd.concat(frames, ignore_index=True)
    data['high'] = data.choice_type.eq('High').astype(int)
    data['low'] = data.choice_type.eq('Low').astype(int)
    data['unknown'] = data.choice_type.eq('Unknown').astype(int)
    data['valid'] = data.high + data.low
    return trait, data


def variability(data):
    families = (data[data.condition.isin(CONDITIONS[:5])]
                .groupby(['model', 'personality', 'condition', 'cluster_id'], sort=True)
                .agg(high=('high', 'sum'), unknown=('unknown', 'sum'), total=('idx', 'size'))
                .reset_index())
    complete = families.total.eq(5) & families.unknown.eq(0)
    families['complete'] = complete.astype(int)
    for name, counts in [('homogeneous', [0, 5]), ('moderate', [1, 4]), ('mixed', [2, 3])]:
        families[name] = (complete & families.high.isin(counts)).astype(int)
    families['families_with_unknown'] = families.unknown.gt(0).astype(int)
    def summary(keys):
        g = families.groupby(keys, sort=True)[
            ['complete', 'families_with_unknown', 'homogeneous', 'moderate', 'mixed']].sum()
        for name in ['homogeneous', 'moderate', 'mixed']:
            g['pct_' + name] = 100 * g[name] / g.complete
        return g.reset_index()
    return summary(['personality']), summary(['model', 'personality', 'condition'])


def paired_bootstrap(data, mc, direct=False):
    comparisons = [('FG_job', 'FG_legal'), ('FB_job', 'FB_legal')] if direct else [
        (c, 'SA') for c in CONDITIONS[1:5]]
    rng = np.random.default_rng(PAIRED_SEED)
    rows = []
    for model in MODELS:
        for trait in TRAITS:
            subset = data[(data.model == model) & (data.personality == trait)]
            for a, b in comparisons:
                left = subset[subset.condition.eq(a)][['idx', 'cluster_id', 'high', 'valid']]
                right = subset[subset.condition.eq(b)][['idx', 'high', 'valid']]
                pairs = left.merge(right, on='idx', suffixes=('_a', '_b'), validate='one_to_one')
                valid = pairs[pairs.valid_a.eq(1) & pairs.valid_b.eq(1)]
                grouped = valid.groupby('cluster_id', sort=True).agg(
                    n=('idx', 'size'), a=('high_a', 'sum'), b=('high_b', 'sum'))
                n = grouped.n.to_numpy(dtype=float)
                va = grouped.a.to_numpy(dtype=float)
                vb = grouped.b.to_numpy(dtype=float)
                k = len(grouped)
                if not k:
                    raise ValueError('No valid pairs for bootstrap')
                distribution = np.empty(N_BOOTSTRAP)
                for start in range(0, N_BOOTSTRAP, 1000):
                    end = min(start + 1000, N_BOOTSTRAP)
                    draws = rng.integers(0, k, size=(end-start, k))
                    denom = n[draws].sum(axis=1)
                    distribution[start:end] = 100 * (
                        va[draws].sum(axis=1) / denom - vb[draws].sum(axis=1) / denom)
                lo, hi = np.percentile(distribution, [2.5, 97.5])
                test = mc[(mc.model == model) & (mc.trait == trait) &
                          (mc.condition_A == a) & (mc.condition_B == b)]
                if len(test) != 1:
                    raise ValueError('Expected exactly one McNemar row')
                test = test.iloc[0]
                # Direction is B -> A: SA -> condition, or Legal -> Job.
                n01 = int(((valid.high_b == 0) & (valid.high_a == 1)).sum())
                n10 = int(((valid.high_b == 1) & (valid.high_a == 0)).sum())
                if (n01, n10, len(valid)) != (test.A_high_B_low, test.A_low_B_high, test.n_items):
                    raise ValueError('Discordant counts disagree with main analysis')
                rows.append(dict(model=model, trait=trait, comparison=f'{a} vs {b}',
                    condition_A=a, condition_B=b, delta_high_pp=100*(va.sum()/n.sum()-vb.sum()/n.sum()),
                    n01=n01, n10=n10, n_valid_pairs=len(valid),
                    n_contributing_families=k, p_fdr=test.p_fdr,
                    ci_low_pp=lo, ci_high_pp=hi, bootstrap_sd_pp=distribution.std(ddof=1),
                    ci_excludes_zero=not (lo <= 0 <= hi),
                    n_bootstrap=N_BOOTSTRAP, random_seed=PAIRED_SEED))
    return pd.DataFrame(rows)


def unknown_sensitivity(data):
    c = data.groupby(['model', 'personality', 'condition'], sort=True).agg(
        high=('high', 'sum'), low=('low', 'sum'), unknown=('unknown', 'sum'),
        valid=('valid', 'sum'), total=('idx', 'size')).reset_index()
    c['omission_pct'] = 100*c.unknown/c.total
    c['score_pct'] = 100*c.high/c.valid
    c['unknown_low_pct'] = 100*c.high/c.total
    c['unknown_high_pct'] = 100*(c.high+c.unknown)/c.total
    indexed = c.set_index(['model', 'personality', 'condition'])
    rows = []
    for model in MODELS:
        for trait in TRAITS:
            base = indexed.loc[(model, trait, 'SA')]
            for cond in CONDITIONS[1:5]:
                row = indexed.loc[(model, trait, cond)]
                rows.append(dict(model=model, trait=trait, comparison=f'{cond} vs SA',
                    delta_original_pp=row.score_pct-base.score_pct,
                    delta_unknown_low_pp=row.unknown_low_pct-base.unknown_low_pct,
                    delta_unknown_high_pp=row.unknown_high_pct-base.unknown_high_pct,
                    has_unknown=bool(row.unknown+base.unknown)))
    primary = pd.DataFrame(rows)
    explicit = c[c.condition.str.endswith('_expl')].copy()
    return c, primary, explicit


def score_stability(data, trait_structure):
    # Use a separate reproducible random-number stream for each trait.
    seeds = np.random.SeedSequence(SCORE_SEED).spawn(len(TRAITS)*2)
    draws = {t: np.random.default_rng(seeds[i]).integers(
        0, 200, size=(N_BOOTSTRAP, 200), endpoint=False, dtype=np.int16)
        for i, t in enumerate(TRAITS)}
    orders = {t: sorted(trait_structure.loc[trait_structure.personality.eq(t), 'cluster_id'].unique())
              for t in TRAITS}
    rows = []
    for (model, trait, cond), g in data[data.condition.isin(CONDITIONS[:5])].groupby(
            ['model', 'personality', 'condition'], sort=True):
        f = g.groupby('cluster_id', sort=False).agg(high=('high','sum'), valid=('valid','sum'),
                total=('idx','size')).reindex(orders[trait])
        if f.isna().any().any() or not f.total.eq(5).all():
            raise ValueError('Incomplete family structure')
        high = f.high.to_numpy(dtype=float); valid = f.valid.to_numpy(dtype=float)
        den = valid[draws[trait]].sum(axis=1)
        if not (den > 0).all():
            raise ValueError('Bootstrap sample has no valid responses')
        distribution = 100*high[draws[trait]].sum(axis=1)/den
        lo, hi = np.percentile(distribution, [2.5,97.5], method='linear')
        rows.append(dict(model=model, trait=trait, condition=cond,
            score_pct=100*high.sum()/valid.sum(), bootstrap_sd_pp=distribution.std(ddof=1),
            ci_low_pct=lo, ci_high_pct=hi, n_valid=int(valid.sum()),
            n_unknown=int(1000-valid.sum()), n_bootstrap=N_BOOTSTRAP, random_seed=SCORE_SEED))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trait, data = load_data()
    mc = pd.read_csv(ROOT/'Results/mcnemar_results_all_models.csv')
    if len(mc) != 126 or mc.duplicated(['model','trait','comparison']).any():
        raise ValueError('Run the main analysis first (126 comparisons required)')
    a,b = variability(data)
    print('Computing condition bootstrap (84 comparisons)...', flush=True)
    c = paired_bootstrap(data,mc)
    print('Computing Job--Legal bootstrap (42 comparisons)...', flush=True)
    d = paired_bootstrap(data,mc,direct=True)
    e,f,g = unknown_sensitivity(data)
    print('Computing score stability (105 cells)...', flush=True)
    h = score_stability(data,trait)
    tables={'A_within_family_aggregate':a, 'B_within_family_detailed':b,
        'C_condition_bootstrap':c, 'D_context_bootstrap':d, 'E_response_counts':e,
        'F_unknown_implicit':f[f.has_unknown], 'G_unknown_explicit':g[g.unknown.gt(0)],
        'H_score_stability':h}
    for name,table in tables.items():
        table.to_csv(OUT/('Table_'+name+'.csv'),index=False)
    f.to_csv(OUT/'unknown_implicit_all_comparisons.csv',index=False)
    g.to_csv(OUT/'unknown_explicit_all_conditions.csv',index=False)
    assert [len(t) for t in tables.values()] == [3,105,84,42,147,34,23,105]
    print('Supplementary CSVs generated successfully.',flush=True)


if __name__ == '__main__':
    main()
