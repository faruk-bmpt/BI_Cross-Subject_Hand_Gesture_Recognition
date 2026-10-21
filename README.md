# Cross-Subject Bioimpedance Gesture Recognition Using Few-Shot Calibration and Ensemble Learning

## Project Overview

This project investigates subject-independent hand-gesture recognition
from bioimpedance measurements, with an emphasis on reducing the amount
of user-specific calibration required for reliable predictions. The
central experimental design is a Leave-One-Subject-Out (LOSO) evaluation
with few-shot calibration.

The project plan also includes subject-dependent benchmarking, feature
ablation, statistical validation, feature-importance analysis, and
computational-efficiency assessment. These complementary analyses are
intended to characterize generalization, personalization,
interpretability, and potential wearable-system applicability.

> **Status note:** This repository README describes the planned research
> workflow and reporting structure. Report numerical results or mark
> analyses as completed only after checking the corresponding experiment
> outputs.

## Research Focus

The main research questions are:

-   How well can bioimpedance features generalize to a previously unseen
    participant?
-   How does a limited number of calibration examples affect
    performance?
-   Which impedance feature groups contribute most to gesture
    discrimination?
-   How does the proposed ensemble compare with individual baseline
    classifiers?
-   What are the subject-to-subject variability and computational costs
    of the approach?

## Experimental Workstreams

  -----------------------------------------------------------------------
  Workstream                          Purpose
  ----------------------------------- -----------------------------------
  Dataset analysis:                    Inspect gesture and subject
                                      balance, recording completeness,
                                      and signal characteristics

  Subject-independent evaluation:      Measure cross-subject
                                      generalization using LOSO

  Few-shot calibration:                Evaluate 0-, 1-, 3-, 5-, 7-, and
                                      10-shot adaptation

  Subject-dependent evaluation:        Establish a
                                      personalized-performance benchmark

  Feature ablation:                    Assess the contribution of
                                      different feature groups

  Statistical analysis:                Evaluate differences across
                                      calibration settings and subjects

  Explainability:                      Rank discriminative features and
                                      interpret their contributions

  Runtime analysis:                    Measure training/inference time and
                                      computational requirements

  Visualization:                       Produce figures and tables for
                                      analysis and manuscript reporting
  -----------------------------------------------------------------------

## Dataset Analysis

The initial dataset audit should document:

-   Number of subjects, gestures, channels, and valid samples/recordings
-   Sample counts by gesture and subject
-   Subject--gesture completeness and balance
-   Representative raw magnitude and phase sweeps
-   Raw-versus-filtered signal examples
-   Gesture definitions and acquisition details

Recommended outputs:

-   Samples per gesture
-   Samples per subject
-   Subject-by-gesture heatmap
-   Representative raw magnitude and phase plots
-   Raw-versus-smoothed signal plots
-   Dataset summary table and gesture-definition table

Populate dataset counts and acquisition specifications from the verified
dataset metadata. Do not infer missing values.

## Bioimpedance Feature Representation

The proposed feature representation is built from impedance magnitude
and phase responses and may include:

-   Magnitude descriptors
-   Phase descriptors
-   Impedance-gradient descriptors
-   Inter-channel ratio descriptors
-   Cumulative impedance-profile descriptors

The planned preprocessing steps include Savitzky--Golay smoothing, phase
unwrapping, and impedance stabilization where appropriate for the
recorded signals.

A feature-summary table should report each feature group, its feature
count, and a concise definition. The final feature dimensions must be
taken from the implemented feature-extraction code and saved outputs.

## Subject-Independent Evaluation: LOSO

The primary evaluation uses Leave-One-Subject-Out cross-validation:

1.  Hold out one participant as the target test subject.
2.  Train the source model using the remaining participants.
3.  For the 0-shot setting, evaluate the target subject without using
    target-subject calibration examples.
4.  For few-shot settings, use the designated number of target-subject
    examples for calibration.
5.  Keep calibration examples separate from target examples used for
    evaluation.
6.  Repeat across subjects and summarize subject-level performance.

### Calibration settings

`k ∈ {0, 1, 3, 5, 7, 10}` shots.

The exact definition of a "shot" (for example, one labelled example per
class or one labelled sample under another sampling rule) must be
specified consistently in the code and manuscript.

### Leakage prevention

The implementation and report should verify that:

-   Source training subjects do not overlap with the held-out target
    subject.
-   Calibration examples are excluded from the target evaluation set.
-   Preprocessing and feature scaling are fitted using source training
    data only, unless a clearly specified adaptation protocol explicitly
    permits otherwise.
-   Target-subject labels are used only for the designated calibration
    examples.
-   Calibration sampling is reproducible and recorded.
-   Model selection and hyperparameter tuning do not use the held-out
    evaluation examples.

Document any deviations from these rules rather than describing the
evaluation as leakage-free without verification.

## Proposed Ensemble and Baselines

The proposed ensemble combines:

-   XGBoost
-   Random Forest

The planned ensemble prediction averages the class-probability outputs
of the two classifiers. Confirm that probability averaging and class
ordering are implemented consistently before reporting results.

Baseline comparisons should include Random Forest, XGBoost, and the
proposed ensemble under comparable data splits, calibration samples, and
metrics. Hyperparameters should be documented from the actual scripts or
configuration files rather than described only in general terms.

## Subject-Dependent Benchmark

A separate subject-dependent experiment is planned to estimate
personalized performance. Keep this analysis separate from the primary
LOSO experiment because it answers a different question.

Suggested outputs:

-   Per-subject accuracy
-   A representative or aggregated confusion matrix, with its
    aggregation method stated
-   Comparison of subject-dependent and LOSO protocols
-   A results table containing accuracy, F1-score, and MCC where
    available

Do not directly interpret subject-dependent results as evidence of
cross-subject generalization.

## Feature Ablation and Explainability

The planned ablation study compares feature groups such as:

-   Magnitude
-   Phase
-   Gradient
-   Ratio
-   Cumulative descriptors

Use consistent data splits and model settings when comparing feature
subsets. Report the feature-selection or removal procedure and the
resulting feature dimensions.

Explainability analyses may include:

-   Averaged XGBoost feature-importance rankings across LOSO folds
-   A ranked table of the most important features
-   SHAP summaries, if implemented and validated
-   Interpretation of feature groups in relation to measured impedance
    responses

Feature importance indicates model reliance, not by itself causal
physiological significance. Physiological interpretations should remain
proportional to the available evidence.

## Metrics and Statistical Analysis

The proposed evaluation metrics are:

-   Accuracy
-   Balanced accuracy
-   Weighted F1-score
-   Matthews Correlation Coefficient (MCC)

Planned statistical analyses include:

-   Wilcoxon signed-rank comparisons between 0-shot and each nonzero
    calibration setting
-   Friedman test across all calibration settings
-   95% confidence intervals
-   Effect-size estimates where appropriate

Statistical tests should use paired subject-level results when the same
subjects are evaluated across conditions. State the unit of analysis,
test assumptions, correction for multiple comparisons, and effect-size
method. Do not treat correlated windows or repeated samples as
independent participants.

## Recommended Figures

  -----------------------------------------------------------------------
  ID                                  Figure
  ----------------------------------- -----------------------------------
  D1                                  Sample count by gesture

  D2                                  Sample count by subject

  D3                                  Subject-by-gesture completeness
                                      heatmap

  D4                                  Representative raw magnitude and
                                      phase responses

  D5                                  Raw versus smoothed signal

  SI1                                 Performance across calibration-shot
                                      settings, with confidence intervals

  SI2                                 LOSO performance distribution
                                      across calibration settings

  SI3                                 Normalized confusion matrix for
                                      0-shot

  SI4                                 Normalized confusion matrix for
                                      10-shot

  SI5                                 Calibration gain relative to 0-shot

  SI6                                 Per-subject performance at selected
                                      calibration levels

  SI7                                 Performance versus number of
                                      calibration shots

  SI8                                 Gesture-wise recall comparison

  SI9                                 Feature-importance ranking

  SI10                                Random Forest, XGBoost, and
                                      ensemble comparison

  SI11                                Training and inference runtime
                                      comparison

  P1                                  Runtime comparison

  P2                                  End-to-end pipeline diagram
  -----------------------------------------------------------------------

Figures should be generated from saved experiment outputs and include
clear labels, units, sample sizes, and uncertainty estimates where
applicable.

## Recommended Tables

  ID    Table
  ----- ------------------------------------------
  D1    Dataset summary
  D2    Gesture definitions
  SI1   Feature-group summary
  SI2   Main LOSO calibration results
  SI3   Per-subject performance
  SI4   Per-gesture recall
  SI5   Statistical comparisons
  SI6   Top discriminative features
  SI7   Baseline model comparison
  SI8   Computational complexity
  SD1   Subject-dependent results
  SD2   Subject-dependent versus LOSO comparison
  A1    Feature-ablation results

Populate all tables from verified outputs. Leave values clearly marked
as pending until they have been computed; do not insert estimated or
illustrative values as if they were experimental findings.

## Computational and Deployment Analysis

The planned computational analysis records:

-   Training time
-   Inference time
-   Feature dimensionality
-   Memory requirements, where measured

These measurements can inform discussion of potential wearable
deployment. Claims about real-time operation, edge deployment, or
clinical utility require appropriate measurements and validation; they
should not be inferred solely from using tree-based models.

## Limitations to Discuss

Potential limitations to assess and discuss include:

-   Dataset size and participant diversity
-   Sensitivity to sensor placement and electrode contact
-   Physiological variation between participants
-   Remaining calibration requirements
-   Generalization beyond the recorded acquisition setup
-   Whether runtime measurements reflect the intended deployment
    hardware

Include only limitations relevant to the actual dataset and experiments,
and distinguish observed limitations from anticipated future challenges.

## Suggested Repository Structure

``` text
.
├── README.md
├── 0_base_model_SI_SD_pipeline.py
├── 01_dataset_analysis.py
├── 02_subject_independent.py
├── 03_subject_dependent.py
├── 04_ablation_study.py
├── 05_feature_explainability.py
├── 06_statistical_analysis.py
├── 07_runtime_analysis.py
├── 08_visualization_tools.py
├── Figures/
├── Tables/
├── Results/
└── Models/
```

This is a suggested organization based on the project roadmap. Adjust
filenames and folders to match the actual repository; do not list files
that are not present.

## Reproducibility Checklist

Before publishing the repository, document:

-   Python and package versions
-   Dataset access conditions and data provenance
-   Preprocessing parameters
-   Feature definitions and final feature count
-   LOSO split and calibration-sampling rules
-   Random seeds
-   Model configurations and hyperparameters
-   Metric definitions and statistical procedures
-   Output filenames and how figures/tables are regenerated

Do not upload sensitive participant information or restricted raw data.
Confirm that the dataset's consent, ethics, and sharing conditions
permit any files included in the repository.

## Manuscript Positioning

The main scientific narrative should emphasize **cross-subject
generalization and few-shot bioimpedance calibration**, supported by
subject-level analysis, transparent leakage prevention, feature
interpretation, and appropriate statistical validation---not raw
accuracy alone.

## Keywords

Bioimpedance; hand-gesture recognition; subject-independent learning;
Leave-One-Subject-Out; few-shot calibration; domain adaptation; ensemble
learning; Random Forest; XGBoost; feature importance; wearable sensing;
physiological signal processing.
