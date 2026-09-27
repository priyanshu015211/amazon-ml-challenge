# amazon-ml-challenge
````markdown
# Entity Resolution & Record Linkage

A high-precision machine learning pipeline for matching records representing the same real-world entity across multiple data sources.

The system is designed for noisy, duplicated, and inconsistently formatted entity data, with particular emphasis on **high precision and minimizing false merges**.

---

## Overview

Real-world datasets rarely represent the same entity in exactly the same way.

For example:

```text
Source 1
Acme Robotics Inc.
500 Market St, San Jose

Source 2
Acme Robotics Incorporated
500 Market Street, San Jose CA
````

Although the records are formatted differently, they may represent the same entity.

This project addresses that problem using a multi-stage **entity-resolution pipeline**:

```text
                    Raw Data
                       │
                       ▼
              ┌─────────────────┐
              │ Normalization   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    Blocking     │
              └────────┬────────┘
                       │
                       ▼
               Candidate Pairs
                       │
                       ▼
              ┌─────────────────┐
              │ Feature Engine  │
              └────────┬────────┘
                       │
                       ▼
             ┌───────────────────┐
             │   Random Forest   │
             │   Classifier      │
             └─────────┬─────────┘
                       │
                       ▼
             Match Probability
                       │
                       ▼
             Threshold Optimization
                       │
                       ▼
              Match / No Match
                       │
                       ▼
              Final TSV Output
```

---

## Key Objectives

* Match records representing the same entity across different sources.
* Handle variations in names and addresses.
* Reduce the number of comparisons using high-recall blocking.
* Generate rich pairwise similarity features.
* Train a supervised Random Forest classifier.
* Handle severe class imbalance.
* Include hard negative examples.
* Optimize the decision threshold for **F0.5**.
* Explicitly support **no-match/singleton** cases.
* Provide an auditable candidate-matching process.

---

## Features

### Name Normalization

The normalization layer handles:

* Lowercase conversion
* Unicode normalization
* Punctuation removal
* Whitespace normalization
* Company suffix normalization
* Tokenization
* Core-name extraction

Example:

```text
Acme Robotics, Incorporated
             ↓
acme robotics inc
             ↓
acme robotics
```

### Address Normalization

Handles common address variations such as:

```text
Street     → st
Road       → rd
Avenue     → ave
Boulevard  → blvd
Drive      → dr
Lane       → ln
```

Example:

```text
500 Market Street
             ↓
500 market st
```

The original data is retained rather than aggressively destroying information.

---

## Blocking

Comparing every Source 1 record against every Source 2/3 record can be computationally expensive.

Blocking reduces the search space:

```text
1,000,000 records
       ↓
Blocking
       ↓
50,000 candidate pairs
       ↓
Detailed similarity calculations
```

The project uses multiple blocking strategies:

* Name prefix
* First name token
* House number + name
* House number + first token
* Address tokens

Multiple blocking keys are intentionally combined because:

> Blocking determines the maximum possible recall of the matching system.

If a true pair is never generated as a candidate, the classifier cannot recover it.

---

## Feature Engineering

For every candidate pair, the system generates numerical features.

### Name Features

* Normalized name similarity
* Partial similarity
* Token-sort similarity
* Token-set similarity
* Core-name similarity
* Jaccard similarity
* Token overlap
* Exact match
* First-token match
* Length difference
* Length ratio

### Address Features

* Address similarity
* Partial similarity
* Token-sort similarity
* Token-set similarity
* Jaccard similarity
* Token overlap
* Exact address match
* Address length difference
* Address length ratio
* House-number match
* Numeric-token overlap

Example:

```text
name_core_ratio       = 0.98
name_token_set        = 1.00
address_ratio         = 0.93
address_token_set     = 0.96
house_number_match    = 1
```

These features are then passed to the machine-learning model.

---

## Machine Learning Model

The primary classifier is a **Random Forest**.

```text
Candidate Pair
      │
      ▼
Feature Vector
      │
      ▼
Random Forest
      │
      ▼
P(match)
      │
      ├───────────────┐
      ▼               ▼
High probability   Low probability
      │               │
      ▼               ▼
   MATCH           NO MATCH
```

The model supports:

* Random Forest hyperparameter optimization
* `balanced_subsample` class weighting
* Group-aware validation
* Multiple estimators
* Depth and leaf-size tuning
* Feature-subset tuning
* Bootstrap sampling
* Model persistence

---

## Why Random Forest?

Random Forest is useful here because the feature matrix contains different types of signals:

```text
Similarity scores
Exact-match flags
Token overlap
Length ratios
Address features
Numeric matches
```

It can model nonlinear relationships between these features without requiring feature scaling.

For example, the combination:

```text
High name similarity
+
High address similarity
+
Matching house number
```

can provide much stronger evidence than any single feature independently.

---

## Hard Negative Mining

A major challenge in entity resolution is distinguishing:

```text
Similar records that ARE the same
```

from:

```text
Similar records that ARE NOT the same
```

For example:

```text
Acme Robotics Inc.
Acme Robotics Ltd.
```

may look highly similar while potentially representing different entities.

The project therefore supports **hard-negative sampling**.

Hard negatives are non-matching pairs that nevertheless have high similarity.

These examples are particularly valuable for teaching the classifier to avoid false merges.

---

## Validation Strategy

Random pair-level splitting can introduce leakage.

For example:

```text
S1-001 → S2-100
S1-001 → S2-201
S1-001 → S2-999
```

should not be randomly distributed between training and validation sets.

Instead, the project uses **group-aware validation**, grouping candidate pairs by the Source 1 entity.

This helps ensure that the model is evaluated on previously unseen reference entities.

---

## Evaluation

The primary metric is:

$$
F_{0.5}
=
(1+0.5^2)
\frac{Precision \times Recall}
{0.5^2 Precision + Recall}
$$

or:

$$
F_{0.5}
=
1.25
\frac{PR}{0.25P+R}
$$

F0.5 gives greater importance to precision than recall.

The system evaluates:

* Precision
* Recall
* F0.5
* False positives
* False negatives
* Confusion matrix
* Match coverage
* Feature importance

---

## Threshold Optimization

The classifier produces a probability:

```text
P(match)
```

We don't automatically use:

```text
P >= 0.5
```

Instead, the system searches across possible thresholds:

```text
0.01
0.02
0.03
...
0.98
0.99
```

and selects a threshold based on validation performance.

Example:

```text
Threshold    Precision    Recall    F0.5
------------------------------------------
0.50          0.91        0.94      ...
0.70          0.96        0.86      ...
0.80          0.98        0.78      ...
```

The actual values are learned from the dataset.

---

## No-Match Handling

Not every Source 1 entity necessarily has a corresponding record in every source.

Therefore:

```text
S1-001 → S2-100
S1-001 → S3-201

S1-002 → S2-101
S1-002 → NO MATCH in Source 3

S1-003 → NO MATCH
```

The system does **not** force a low-confidence candidate to become a match.

This is important for reducing false merges.

---

## Candidate Ranking

For each Source 1 entity, candidates are ranked by their predicted probability:

```text
S1-001

Candidate       Probability
---------------------------
S2-100             0.97
S2-201             0.72
S2-399             0.18
```

The strongest candidate can then be selected if it satisfies the configured decision criteria.

An optional probability-margin rule is also supported:

```text
Best candidate       = 0.96
Second candidate      = 0.71

Margin = 0.25
```

This can prevent ambiguous matches from being accepted.

---

## Project Structure

```text
entity-resolution/
│
├── data/
│   ├── source1.tsv
│   ├── source2.tsv
│   ├── source3.tsv
│   └── train_ground_truth.tsv
│
├── src/
│   ├── normalization.py
│   ├── blocking.py
│   ├── features.py
│   ├── model.py
│   └── matching.py
│
├── models/
│   └── entity_matcher.joblib
│
├── output/
│   ├── candidate_pairs.tsv
│   ├── matching_results.tsv
│   ├── matching_audit.tsv
│   └── feature_importance.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── train.py
├── requirements.txt
└── README.md
```

---

## Module Responsibilities

| File               | Responsibility                           |
| ------------------ | ---------------------------------------- |
| `normalization.py` | Clean and standardize names/addresses    |
| `blocking.py`      | Generate high-recall candidate pairs     |
| `features.py`      | Generate pairwise similarity features    |
| `model.py`         | Train and optimize Random Forest         |
| `matching.py`      | Convert probabilities into final matches |
| `train.py`         | Execute model training                   |
| `analysis.ipynb`   | Dataset exploration and experiments      |

---

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd entity-resolution
```

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Create `requirements.txt`:

```text
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
rapidfuzz>=3.9
joblib>=1.4
```

---

## Data Preparation

Place the source files inside:

```text
data/
```

Expected conceptual structure:

```text
data/
├── source1.tsv
├── source2.tsv
├── source3.tsv
└── train_ground_truth.tsv
```

The exact column names should be adjusted according to the official challenge dataset specification.

---

## Training

Once the training feature matrix has been generated:

```bash
python train.py
```

The training process:

```text
Training data
     ↓
Group-aware split
     ↓
Random Forest hyperparameter search
     ↓
Best model
     ↓
Validation predictions
     ↓
F0.5 threshold optimization
     ↓
Final Random Forest
     ↓
models/entity_matcher.joblib
```

---

## Output

The final matching file is:

```text
output/matching_results.tsv
```

Example:

```text
source1_id    source2_id    source3_id
S1-001        S2-183        S3-928
S1-002        S2-201
S1-003                      S3-774
S1-004
S1-005        S2-441        S3-812
```

An empty field represents a **no-match** decision.

Additional files:

```text
output/
├── matching_results.tsv
├── matching_audit.tsv
├── candidate_pairs.tsv
└── feature_importance.csv
```

---

## Auditability

The system preserves candidate-level information so that individual decisions can be investigated.

For example:

```text
source1_id    candidate_id    probability    selected
S1-001        S2-183          0.982           True
S1-001        S2-212          0.731           False
S1-001        S2-441          0.102           False
```

This makes it possible to investigate:

* Why a record was matched
* Why another candidate was rejected
* Which features influenced the model
* Where false merges occur
* Where blocking failed

---

## Research Direction

The current system provides a strong supervised entity-resolution baseline.

Potential research extensions include:

```text
Random Forest
     ↓
Gradient Boosting
     ↓
Calibrated probabilities
     ↓
Graph-based entity resolution
     ↓
Probabilistic record linkage
     ↓
Transformer-based semantic similarity
     ↓
Hybrid rule + ML + semantic matching
```

Other possible extensions:

* Character-level embeddings
* Sentence-transformer embeddings
* Cross-encoder pair scoring
* Graph clustering
* Active learning
* Uncertainty-based human review
* Region-specific normalization
* Learned blocking
* Probability calibration
* Cost-sensitive classification
* Global one-to-one assignment

---

## Limitations

The exact effectiveness of the system depends on:

* Quality of the training labels
* Quality of blocking
* Dataset-specific name patterns
* Dataset-specific address formats
* Distribution of positive and negative pairs
* Presence of hard negatives
* Correct handling of singleton entities

The blocking stage is particularly important because a true match that is never generated as a candidate cannot be recovered by the classifier.

---

## License

MIT License

---

## Project Status

**Current pipeline:**

```text
[✓] Data normalization
[✓] Name normalization
[✓] Address normalization
[✓] Multi-strategy blocking
[✓] Candidate generation
[✓] Pairwise feature engineering
[✓] Random Forest classifier
[✓] Class imbalance handling
[✓] Hard-negative sampling
[✓] Group-aware validation
[✓] F0.5 threshold optimization
[✓] No-match handling
[✓] Candidate ranking
[✓] Matching output
[✓] Audit output
[✓] Model persistence

[ ] Dataset-specific tuning
[ ] Official evaluation
[ ] Error analysis
[ ] Advanced semantic matching
[ ] Final benchmark
```

```
```
