# CASI benchmark audit

Reproduction materials for **Auditing Exact Duplicate Exposure and Candidate Coverage in Clinical Abbreviation Disambiguation Benchmarks**, by Tianze Yang, Qiqing Li, Jialun Wu and Xinyu Qiu.

## Recalculate the revision statistics

Use Python 3.9 to 3.12 and install `requirements.txt` in a virtual environment. From this directory run:

```sh
python -m pip install -r requirements.txt
python scripts/run_revision.py
python scripts/context_sensitivity.py
MPLCONFIGDIR=tmp/matplotlib python scripts/make_figures.py
```

This route uses archived predictions and reconstructs the gate reference rates, support AUROC, clustered intervals, paired macro F1 differences, subgroup calibration, named-label sensitivity and PNG figures. Bootstrap analyses use 10,000 draws and seed 20261030. `results/revision_summary.json` contains the reported values.

## Refit selectors with an authorized CASI copy

Obtain CASI from its original distributor: https://hdl.handle.net/11299/137703 . The University of Minnesota resource page is https://healthinformatics.umn.edu/natural-language-processing/resources . Place the source file under the ignored `private/source_data/` directory and run:

```sh
python scripts/run_revision.py --source-casi private/source_data/casi.txt
python scripts/rerun_selectors.py --source-casi private/source_data/casi.txt
```

The first command regenerates the word selector, validation and audit probabilities, exact-duplicate arms and split manifest. It verifies agreement with the archived word predictions. The second fits all three TF-IDF representations and writes newly measured outputs to `private/refit_results/`. Resource measurements from the manuscript are preserved in `inputs/reference/reference_model_summary.json`; newly measured timings characterize the rerun environment.

To rerun the frozen encoder comparator, install `requirements-encoder.txt` and obtain a local snapshot of `emilyalsentzer/Bio_ClinicalBERT`, revision `d5892b39a4adaed74b92212a44081509db72f87b`. Then add `--encoder-snapshot PATH_TO_SNAPSHOT` to the selector command. The encoder uses local weights, dynamic padding, 128 tokens, batch size 32, attention-mask mean pooling and balanced logistic regression heads. No hosted inference service is used.

## Contents and interpretation

* `results/split_manifest.json`: deterministic hashed item IDs and train, validation, core test or audit membership. Authorized source holders can regenerate IDs from the documented normalization rule.
* `results/*predictions.json`: item labels, candidate probabilities where available, confidence and paired ablation outcomes. `inputs/reference/` preserves the original comparison predictions in the same core test order.
* `results/calibration_by_*.csv`: calibration for every abbreviation and candidate-pool stratum.
* `results/human_review_summary.json` and `human_review_ratings_without_context.csv`: the first author's completed random 50-item inspection, with E 0, R 1, D 47 and U 2. The sampling seed is 20261030. The R case is a label-span near miss involving SA arrest. U denotes an indeterminate relation.
* `results/contextual_meaning_labels.csv`: explicit named contextual meanings used in the sensitivity calculation. This is a named-label sensitivity of the original candidate inventories.
* `legacy/`: portable copies of the preprocessing and selector implementation. The source parser preserves the original UTF-8 replacement behavior.
* `scripts/make_figures.py`: generates manuscript Figures 1 to 4 and supplementary Figure S1 as PNG files in `figures/`. Run the reproduction commands above to create the figures.

The core metrics cover 207 abbreviation-qualified classes. The audit contains 141 deliberately withheld low-support classes and evaluates confidence when the correct label is absent. Macro F1 intervals condition on the primary split and its observed class support. Duplicate comparisons use exposed minus purged training, and the cluster sensitivity resamples exact normalized context groups. Support AUROC uses supported core items as positive and resamples abbreviation clusters. A single validation-fitted temperature and coverage gate are applied unchanged to the primary core and audit partitions.

Source clinical contexts, source annotations, UMLS content and model weights are obtained separately from their original distributors. The included hashes are reproducibility join keys for authorized source holders. The public package supports reconstruction from archived predictions and source-based reruns by the authors or other authorized users.

## Code license and data terms

Original analysis code in this repository is released under the MIT License. CASI-derived labels and prediction records retain the applicable source-data terms. The code license does not relicense CASI, UMLS or encoder weights.
