# Faking Good and Faking Bad in LLMs

This repository contains the public data, analysis notebook, aggregate results, and figures associated with the paper:

**Faking Good and Faking Bad in LLMs: Response Distortion Across Dark Triad Personality Traits**

The study examines response distortion in large language models across three Dark Triad traits: Machiavellianism, narcissism, and psychopathy. Models were evaluated under self-assessment, fake-good, fake-bad, and explicit fake-bad prompt conditions, including employment and legal contexts.

The manuscript describes the prompt templates and context manipulations used for the self-assessment, job, and legal conditions.

## Repository Contents

```text
.
├── Main_Analysis.ipynb           # Main reproducible analysis notebook
├── Data/                         # Curated item-level labeled model outputs
│   ├── DeepseekV3_2/
│   ├── GPT4_1/
│   ├── Gemma3/
│   ├── Grok4_3/
│   ├── Llama3_3/
│   ├── Mistral_Large3/
│   ├── Qwen2_5/
│   └── skipped_responses.csv
├── Results/                      # Aggregate result CSVs
├── Plots/                        # Generated analysis figures
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

## Results

`Results/` contains the derived aggregate outputs:

- `all_models_high_percentages.csv`: counts and valid-response high-trait percentages for each model, condition, and trait.
- `all_models_deltas.csv`: condition-level changes relative to self-assessment.
- `mcnemar_results_all_models.csv`: paired McNemar test outputs, FDR-corrected p-values, significance labels, and directional effect sizes.

Percentages are computed over valid classified responses:

```text
High / (High + Low)
```

`Unknown` responses are reported separately as omission/non-classifiable rates.

## Figures

`Plots/` contains the generated figures used in the analysis. Each figure is provided in both PDF and EPS format:

- `self_assessment_distribution`
- `self_assessment_by_model`
- `condition_shifts_by_context`
- `implicit_profiles_by_model`
- `explicit_fake_bad_profiles_by_model`
- `explicit_fake_bad_shifts`
- `mcnemar_effect_size_heatmaps`

## Reproducing the Analysis

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then launch JupyterLab from the activated environment:

```bash
python -m jupyter lab
```

Open `Main_Analysis.ipynb` and run it from top to bottom to regenerate the aggregate result CSVs and figures.

If you prefer to use an existing JupyterLab installation outside this virtual environment, register the environment as a selectable kernel:

```bash
python -m ipykernel install --user --name dark-triad-llms --display-name "Python (Dark Triad LLMs)"
```

Then open JupyterLab as usual and select `Python (Dark Triad LLMs)` as the notebook kernel.
