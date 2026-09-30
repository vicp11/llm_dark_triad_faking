# Faking Good and Faking Bad in LLMs

This repository contains the public data, analysis notebook, aggregate results, and figures associated with the paper:

**Faking Good and Faking Bad in LLMs: Response Distortion Across Dark Triad Personality Traits**

**Publication status:** The manuscript is currently under review at PLOS ONE.

The study examines response distortion in large language models across three Dark Triad traits: Machiavellianism, narcissism, and psychopathy. Models were evaluated under self-assessment, fake-good, fake-bad, and explicit fake-bad prompt conditions, including employment and legal contexts.

The manuscript and S1 Appendix describe the prompt templates and context manipulations. The analysis reproduces the results from the archived responses and does not require model access or API calls.

## Repository Contents

```text
.
├── Main_Analysis.ipynb           # Main reproducible analysis notebook
├── reproduce.py                  # Run all analyses and generate all figures
├── scripts/supplementary.py      # S1 Appendix numerical analyses
├── Data/                         # Curated item-level labeled model outputs
│   ├── DeepseekV3_2/
│   ├── GPT4_1/
│   ├── Gemma3/
│   ├── Grok4_3/
│   ├── Llama3_3/
│   ├── Mistral_Large3/
│   ├── Qwen2_5/
│   ├── skipped_responses.csv
│   └── trait_items.json          # Item metadata and statement-family mapping
├── Results/                      # Main CSVs and Supplementary/ Tables A–H
├── Plots/                        # Generated analysis figures
├── tests/                        # Numerical regression checks and references
└── requirements.txt              # Python dependencies
```

## Data

Each model folder in `Data/` contains the labeled item-level CSV files used by the analyses:

- `SA.csv`: self-assessment baseline.
- `FG_job.csv` and `FG_legal.csv`: fake-good conditions in job and legal contexts.
- `FB_job.csv` and `FB_legal.csv`: fake-bad conditions in job and legal contexts.
- `explicit/FB_job.csv` and `explicit/FB_legal.csv`: explicit fake-bad prompt variants.

The item-level response CSV files contain:

| Column | Description |
|---|---|
| `idx` | TRAIT item identifier. |
| `personality` | Target trait: Machiavellianism, Narcissism, or Psychopathy. |
| `query` | Scenario or question presented to the model. |
| `choice_type` | TRAIT label for the selected option: `High`, `Low`, or `Unknown`. |

Rows with `choice_type = Unknown` remain in the main model CSV files so omission/non-classifiable rates can be reproduced. The same rows are also collected in `Data/skipped_responses.csv`, which adds the raw model response for auditability.

`Data/skipped_responses.csv` contains:

| Column | Description |
|---|---|
| `model_folder` | Model folder containing the source response file. |
| `condition` | Analysis condition: `SA`, `FG_job`, `FG_legal`, `FB_job`, `FB_legal`, `FB_job_expl`, or `FB_legal_expl`. |
| `source_file` | Path to the corresponding item-level CSV file. |
| `idx` | TRAIT item identifier. |
| `personality` | Target trait. |
| `query` | Scenario or question presented to the model. |
| `raw_response` | Original model response or API/filtering error for the non-classifiable row. |

The `_expl` suffix marks explicit fake-bad prompt variants; these correspond to the files under each model folder's `explicit/` subfolder.

There are 49 response files: seven models × seven conditions × 3,000 items
(147,000 responses).
`trait_items.json` contains the 3,000 administered TRAIT items and randomized
answer options. Match it to responses using `idx`. Within each trait, the
`statement` field identifies 200 families of five items. The item content comes
from TRAIT (Lee et al., 2025, *Do LLMs Have Distinct and Consistent Personality?
TRAIT: Personality Testset designed for LLMs with Psychometrics*,
doi:10.18653/v1/2025.findings-naacl.469).

## Results

`Results/` contains the derived aggregate outputs:

- `all_models_high_percentages.csv`: counts and valid-response high-trait percentages for each model, condition, and trait.
- `all_models_deltas.csv`: condition-level scores and changes relative to self-assessment (percentage points).
- `mcnemar_results_all_models.csv`: all 126 paired comparisons, contingency counts, exact McNemar p-values, jointly Benjamini–Hochberg-adjusted p-values, matched odds ratios, and Yule's Q. These comprise 84 implicit-condition versus self-assessment comparisons and 42 direct Job–Legal comparisons.

Percentages are computed over valid classified responses:

```text
100 × High / (High + Low)
```

`Unknown` responses are reported separately as omission/non-classifiable rates.

CSV numerical values retain calculation precision. Rounding is applied only
for presentation; in particular, very small p-values are not rounded to zero.
Main score differences use the valid responses in each condition separately.
Paired comparisons instead retain only items classified in **both** conditions,
so their percentage-point differences can differ slightly from the main score
differences when classifications are missing.

### S1 Appendix

`Results/Supplementary/` provides the numerical contents of Tables A–H, in
machine-readable form rather than LaTeX. Column names are descriptive; `pp`
denotes percentage points and `pct` denotes a percentage.
Here, primary conditions are SA and the four implicit FG/FB conditions.

| S1 table | CSV | Contents |
|---|---|---|
| A | `Table_A_within_family_aggregate.csv` | Within-family response patterns, pooled by trait |
| B | `Table_B_within_family_detailed.csv` | Same patterns by model, trait, and primary condition (105 rows) |
| C | `Table_C_condition_bootstrap.csv` | Paired condition–SA differences and cluster-bootstrap intervals (84 rows) |
| D | `Table_D_context_bootstrap.csv` | Paired Job–Legal differences and cluster-bootstrap intervals (42 rows) |
| E | `Table_E_response_counts.csv` | High, Low, Unknown counts and omission rates for all 147 cells |
| F | `Table_F_unknown_implicit.csv` | Unknown-response sensitivity for affected implicit comparisons (34 rows) |
| G | `Table_G_unknown_explicit.csv` | Unknown-response sensitivity for affected explicit conditions (23 rows) |
| H | `Table_H_score_stability.csv` | Score stability for the 105 primary-condition cells |

Tables A–B use families with five classified responses. Patterns are homogeneous
(5/0), moderately mixed (4/1), or mixed (3/2). Table A pools family counts before
computing percentages; it is not an unweighted average of the percentages in B.

Tables C–D report pointwise percentile 95% intervals from 10,000 family-bootstrap
samples. Both analyses start a separate random stream with seed `20260810`.
Families containing at least one valid pair contribute to these paired analyses.
For a comparison A versus B, `n01` counts B-Low → A-High and `n10` counts
B-High → A-Low; the difference is `100 × (n01 − n10) / n_valid_pairs`.
Thus the direction is SA → condition in C and Legal → Job in D.
The FDR column uses the joint correction over all 126 McNemar tests.

Tables F–G recode all Unknown responses as Low or all as High. Their complete
outputs, including unaffected comparisons/conditions, are also saved as
`unknown_implicit_all_comparisons.csv` (84 rows) and
`unknown_explicit_all_conditions.csv` (42 rows).

Table H uses 10,000 draws of 200 statement families, with replacement, for each
trait. Seed `20260819` initializes the trait-specific streams; the same family
draws are used across models and conditions within a trait. Scores exclude
Unknown responses.

Tables I–J in S1 document prompts and generation settings rather than calculated
results; consult the manuscript/S1 Appendix for that experimental documentation.

## Figures

`Plots/` contains the generated figures used in the analysis. Each figure is provided in both PDF and EPS format:

| Paper figure | Filename stem |
|---|---|
| 1 | `self_assessment_distribution` |
| 2 | `self_assessment_by_model` |
| 3 | `condition_shifts_by_context` |
| 4 | `mcnemar_effect_size_heatmaps` |
| 5 | `implicit_profiles_by_model` |
| 6 | `mcnemar_job_legal_heatmaps` |
| 7 | `explicit_fake_bad_shifts` |
| 8 | `explicit_fake_bad_profiles_by_model` |

## Reproducing the Analysis

Tested with Python 3.13. From the repository root, create a virtual environment,
install the pinned dependencies, run the analyses, and check the outputs:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
python -m unittest discover -s tests -v
```

The complete run regenerates all result CSVs and eight figures without using
precomputed reference results. It also works when invoked via an absolute path
from another working directory. Tests compare the results with numerical
reference results, including all bootstrap intervals and the exact
McNemar/FDR calculations. Numerical differences at floating-point tolerance
are accepted; missing rows or changed results fail the checks.

For interactive use, launch JupyterLab from the activated environment:

```bash
python -m jupyter lab
```

Run `Main_Analysis.ipynb` from top to bottom to regenerate the main CSVs and
figures, then run `python scripts/supplementary.py` for Tables A–H.

If you prefer to use an existing JupyterLab installation outside this virtual environment, register the environment as a selectable kernel:

```bash
python -m ipykernel install --user --name dark-triad-llms --display-name "Python (Dark Triad LLMs)"
```

Then open JupyterLab as usual and select `Python (Dark Triad LLMs)` as the notebook kernel.

All eight figures use the same environment specified in `requirements.txt`.
Font availability and graphics-library versions can affect appearance on other
systems; PDF/EPS timestamps can also differ between runs.
