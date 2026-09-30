"""Check regenerated results against numerical reference results.

Run: python -m unittest discover -s tests -v
Reference CSVs are test fixtures, never inputs to the analysis.
"""
from pathlib import Path
import re
import unittest

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'tests/reference'
OUT = ROOT / 'Results'
SUP = OUT / 'Supplementary'


def key(value):
    value = re.sub(r'[^a-z0-9]', '', str(value).lower())
    value = value.replace('assessment', 'sa').replace('legal', 'leg')
    return {'expfbjob': 'fbjobexpl', 'expfbleg': 'fblegexpl'}.get(value, value)


def aligned(frame, keys):
    frame = frame.copy()
    frame.index = pd.MultiIndex.from_arrays([frame[k].map(key) for k in keys])
    if frame.index.duplicated().any():
        raise AssertionError('Duplicate comparison keys')
    return frame.sort_index()


class ReproductionTests(unittest.TestCase):
    def compare(self, new, old, new_keys, old_keys, columns):
        new, old = aligned(new, new_keys), aligned(old, old_keys)
        self.assertEqual(new.index.tolist(), old.index.tolist())
        for a, b in columns.items():
            np.testing.assert_allclose(new[a], old[b], rtol=1e-11, atol=1e-11,
                                       err_msg=f'{a} / {b}')

    def test_main_scores(self):
        new = pd.read_csv(OUT / 'all_models_high_percentages.csv')
        old = pd.read_csv(REF / 'all_models_high_percentages.csv')
        self.assertEqual(len(new), 147)
        cols = ['High', 'Low', 'Unknown', 'Total', 'Valid_Total', 'high_percent']
        self.compare(new, old, ['model', 'trait', 'condition'], ['model', 'trait', 'condition'],
                     {c: c for c in cols})

    def test_exact_mcnemar_and_joint_fdr(self):
        new = pd.read_csv(OUT / 'mcnemar_results_all_models.csv')
        old = pd.read_csv(REF / 'recalculated_mcnemar.csv')
        self.assertEqual(len(new), 126)
        cols = ['n_items', 'A_0_B_0', 'A_0_B_1', 'A_1_B_0', 'A_1_B_1',
                'A_high_B_low', 'A_low_B_high', 'discordant']
        self.compare(new, old, ['model', 'trait', 'comparison'], ['model', 'trait', 'comparison'],
                     {c: c for c in cols})
        p = np.array([binomtest(int(r.A_high_B_low), int(r.discordant)).pvalue
                      if r.discordant else 1.0 for r in new.itertuples()])
        np.testing.assert_allclose(new.p_value, p, rtol=1e-12, atol=0)
        np.testing.assert_allclose(new.p_fdr, multipletests(p, method='fdr_bh')[1],
                                   rtol=1e-12, atol=0)
        self.assertTrue(new.p_fdr.gt(0).all())
        odds = (new.A_high_B_low + .5) / (new.A_low_B_high + .5)
        np.testing.assert_allclose(new.matched_or_A_over_B, odds, rtol=1e-12)
        np.testing.assert_allclose(new.yules_q_A_over_B, (odds-1)/(odds+1), atol=1e-14)
        self.assertEqual(int(new.p_fdr.lt(.05).sum()), 78)

    def test_deltas_and_abstract_counts(self):
        scores = pd.read_csv(OUT / 'all_models_high_percentages.csv')
        deltas = pd.read_csv(OUT / 'all_models_deltas.csv')
        indexed = scores.set_index(['model', 'trait', 'condition']).high_percent
        for row in deltas.to_dict('records'):
            base = indexed.loc[(row['Model'], row['Trait'], 'SA')]
            for cond in ['FG_job', 'FG_legal', 'FB_job', 'FB_legal', 'FB_job_expl', 'FB_legal_expl']:
                expected = indexed.loc[(row['Model'], row['Trait'], cond)] - base
                self.assertAlmostEqual(row['Δ' + cond], expected, places=11)
        self.assertEqual(int((deltas[['ΔFG_job', 'ΔFG_legal']] < 0).sum().sum()), 39)
        self.assertEqual(int((deltas[['ΔFB_job', 'ΔFB_legal']] > 0).sum().sum()), 23)

    def test_family_variability(self):
        new = pd.read_csv(SUP / 'Table_B_within_family_detailed.csv')
        old = pd.read_csv(REF / 'recalculated_variability.csv')
        self.compare(new, old, ['model', 'personality', 'condition'], ['model', 'trait', 'condition'],
                     {'complete': 'n_clusters_complete', 'families_with_unknown': 'n_clusters_with_unknown',
                      **{c: c for c in ['pct_homogeneous', 'pct_moderate', 'pct_mixed']}})
        a = pd.read_csv(SUP / 'Table_A_within_family_aggregate.csv')
        for row in a.itertuples():
            subset = new[new.personality.eq(row.personality)]
            for name in ['complete', 'homogeneous', 'moderate', 'mixed']:
                self.assertEqual(getattr(row, name), subset[name].sum())
        np.testing.assert_allclose(a[['pct_homogeneous', 'pct_moderate', 'pct_mixed']].round(1),
                                   [[48.6, 33.0, 18.4], [78.1, 16.9, 5.0], [91.0, 6.8, 2.2]])

    def test_paired_bootstraps(self):
        for table, reference, size, significant in [
                ('Table_C_condition_bootstrap.csv', 'recalculated_bootstrap84.csv', 84, 51),
                ('Table_D_context_bootstrap.csv', 'recalculated_bootstrap42.csv', 42, 27)]:
            new, old = pd.read_csv(SUP / table), pd.read_csv(REF / reference)
            self.assertEqual(len(new), size)
            self.compare(new, old, ['model', 'trait', 'comparison'], ['model', 'trait', 'comparison'],
                         {'delta_high_pp': 'point_delta_pp', 'ci_low_pp': 'ci_low_pp',
                          'ci_high_pp': 'ci_high_pp', 'bootstrap_sd_pp': 'bootstrap_sd_pp'})
            np.testing.assert_allclose(new.delta_high_pp, 100*(new.n01-new.n10)/new.n_valid_pairs,
                                       atol=1e-12)
            sig = new.p_fdr.lt(.05)
            self.assertEqual(int(sig.sum()), significant)
            self.assertTrue(new.loc[sig, 'ci_excludes_zero'].all())

    def test_unknown_counts_and_sensitivity(self):
        new = pd.read_csv(SUP / 'Table_E_response_counts.csv')
        old = pd.read_csv(REF / 'response_counts.csv')
        self.compare(new, old, ['model', 'personality', 'condition'], ['Model', 'Trait', 'Condition'],
                     {'high': 'High_n', 'low': 'Low_n', 'unknown': 'Unknown_n',
                      'valid': 'Valid_n', 'total': 'Total_n', 'omission_pct': 'Omission_pct'})
        new = pd.read_csv(SUP / 'unknown_implicit_all_comparisons.csv')
        old = pd.read_csv(REF / 'unknown_implicit_sensitivity.csv')
        self.compare(new, old, ['model', 'trait', 'comparison'], ['Model', 'Trait', 'Comparison'],
                     {'delta_original_pp': 'Delta_original_pp', 'delta_unknown_low_pp': 'Delta_unknown_low_pp',
                      'delta_unknown_high_pp': 'Delta_unknown_high_pp'})
        new = pd.read_csv(SUP / 'unknown_explicit_all_conditions.csv')
        old = pd.read_csv(REF / 'unknown_explicit_sensitivity.csv')
        self.compare(new, old, ['model', 'personality', 'condition'], ['Model', 'Trait', 'Condition'],
                     {'score_pct': 'Score_original_pct', 'unknown_low_pct': 'Score_unknown_low_pct',
                      'unknown_high_pct': 'Score_unknown_high_pct'})
        self.assertEqual(len(pd.read_csv(SUP / 'Table_F_unknown_implicit.csv')), 34)
        self.assertEqual(len(pd.read_csv(SUP / 'Table_G_unknown_explicit.csv')), 23)

    def test_score_stability(self):
        new = pd.read_csv(SUP / 'Table_H_score_stability.csv')
        old = pd.read_csv(REF / 'score_stability.csv')
        old = old[old.Condition.isin(['SA', 'FG-Job', 'FG-Legal', 'FB-Job', 'FB-Legal'])]
        self.assertEqual(len(new), 105)
        self.compare(new, old, ['model', 'trait', 'condition'], ['Model', 'Trait', 'Condition'],
                     {'score_pct': 'Observed_High_score_pct', 'bootstrap_sd_pp': 'Family_bootstrap_SD_pp',
                      'ci_low_pct': 'Bootstrap_95pct_low', 'ci_high_pct': 'Bootstrap_95pct_high',
                      'n_valid': 'Valid_n', 'n_unknown': 'Unknown_n'})
        self.assertAlmostEqual(new.bootstrap_sd_pp.median(), 1.11, places=2)
        widths = new.ci_high_pct-new.ci_low_pct
        self.assertAlmostEqual(widths.median(), 4.30, places=2)
        self.assertAlmostEqual(widths.max(), 6.58, places=2)

    def test_figures_exist(self):
        names = ['self_assessment_distribution', 'self_assessment_by_model',
                 'condition_shifts_by_context', 'mcnemar_effect_size_heatmaps',
                 'implicit_profiles_by_model', 'mcnemar_job_legal_heatmaps',
                 'explicit_fake_bad_shifts', 'explicit_fake_bad_profiles_by_model']
        for name in names:
            for extension in ['pdf', 'eps']:
                self.assertGreater((ROOT / 'Plots' / f'{name}.{extension}').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
