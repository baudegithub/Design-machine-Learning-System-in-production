# News Article Topic Classifier — ML Project Lifecycle Case Study

## Problem

A news aggregation start-up needed to automatically classify incoming articles into topics, using only the article **title** as input. Their previous developer had shipped a lightweight TensorFlow prototype, but it was underperforming in production and the original evaluation metrics had been lost. This project walks through the full **ML Project Lifecycle** — Scoping → Data → Modeling → Deployment — to troubleshoot the prototype and ship an improved beta.

Manual sorting couldn't keep up with the volume of incoming articles, and simple rule-based classification wasn't accurate enough — making this a good fit for a supervised ML approach.

## Approach

**Data.** Titles are labeled into 8 topics (`ENTERTAINMENT`, `HEALTH`, `TECHNOLOGY`, `WORLD`, `BUSINESS`, `SPORTS`, `NATION`, `SCIENCE`), sourced from a revised version of the [Topic Labeled News Dataset](https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset). Since no baseline metrics existed, the first step was re-evaluating the existing prototype on held-out test data to establish one.

**Data-centric iteration.** Rather than jumping straight to model tuning, the project follows a data-centric AI approach — holding the model fixed and improving data quality first:
- Rebuilt the train/dev/test split to fix a severe class imbalance (`SCIENCE` made up under 1% of the original data).
- Ran error analysis per topic to find where the model was failing, which surfaced a labeling bug: every `BUSINESS`-labeled row in one split shared the exact same title, meaning the model had only memorized one pattern instead of learning the category.
- Investigated label ambiguity (e.g., titles that plausibly belong to both `HEALTH` and `NATION`) using top-2 accuracy as a diagnostic, which showed the model's second-best guess was often correct — a sign of genuine class overlap rather than model failure.
- Monitored for out-of-vocabulary token spikes on new data, which surfaced non-English article titles slipping into the pipeline undetected.

**Model.** A TensorFlow/Keras model: a `TextVectorization` layer (10,000-word vocabulary, max sequence length 20) feeding an `Embedding` → `Dense` → `Flatten` → softmax classification head over the 8 topics.

**Deployment.** Served with **TensorFlow Serving**, including:
- Blue-green and canary deployment patterns for safely rolling out new model versions.
- A confidence-threshold pattern for partial automation: predictions above 60% confidence are auto-routed, everything else is flagged for manual review rather than forced into a possibly-wrong category.
- Production monitoring for out-of-vocabulary rate as an early-warning signal for data/model drift.

## Results

| Stage | Sparse categorical accuracy |
|---|---|
| Baseline (original prototype, as re-evaluated) | 77.8% |
| After rebuilding a balanced train/dev/test split | 80.8% |

Error analysis also surfaced a data quality bug (duplicated `BUSINESS` titles) that was silently inflating that category's apparent accuracy to 100% — a reminder that a suspiciously perfect score is itself worth investigating.

## Tech stack

`TensorFlow` / `Keras` · `pandas` · `scikit-learn` · `SQLite` · `TensorFlow Serving`

## Repo structure

```
project-name/
├── README.md
├── E1/                  # Original prototype: data, model, vocab
├── E2/                  # Retrained model after data-centric fixes
├── E3/                  # Vocabulary rebuilt after the BUSINESS label fix
├── assets/              # Diagrams (ML lifecycle, deployment architecture)
└── lab_utils.py         # Shared helpers: data loading, tf.data conversion, metrics
```

## What I'd improve next

- Complete the remaining error-analysis exercises (label-ambiguity resolution, additional data augmentation for `SCIENCE`) to push accuracy further.
- Add automated data-quality checks (e.g., duplicate-title detection) to the pipeline so issues like the `BUSINESS` mislabeling are caught before training rather than during error analysis.
- Track HLP (human-level performance) more rigorously to know how much further model improvement is realistically possible given label ambiguity.

## Acknowledgement

Case study and starter code from DeepLearning.AI's *Machine Learning Engineering for Production (MLOps)* specialization. Dataset adapted from the [Topic Labeled News Dataset](https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset) on Kaggle.

