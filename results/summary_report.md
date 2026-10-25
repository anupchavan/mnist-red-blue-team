# Consolidated MNIST Red/Blue-Team Results

> Generated automatically from the JSON files in `results/` by `python analyze_results.py`. This is a summary of recorded results, not an independent rerun of the experiments.

## Baseline and evasion robustness

| Scenario | Accuracy | Macro F1 | Samples |
|---|---:|---:|---:|
| clean | 99.02% | 99.01% | 10000 |
| fgsm_0.1 | 86.90% | 86.90% | 10000 |
| fgsm_0.2 | 40.48% | 40.23% | 10000 |
| fgsm_0.3 | 14.59% | 11.96% | 10000 |
| pgd_0.1 | 65.26% | 65.62% | 10000 |
| pgd_0.2 | 0.08% | 0.04% | 10000 |
| pgd_0.3 | 0.00% | 0.00% | 10000 |

## Adversarial-training evaluation

| Scenario | Accuracy | Macro F1 | Samples |
|---|---:|---:|---:|
| clean | 98.83% | 98.83% | 10000 |
| whitebox_fgsm_0.1 | 97.79% | 97.77% | 10000 |
| whitebox_fgsm_0.2 | 96.78% | 96.75% | 10000 |
| whitebox_fgsm_0.3 | 95.85% | 95.81% | 10000 |
| whitebox_pgd_0.1 | 97.64% | 97.62% | 10000 |
| whitebox_pgd_0.2 | 95.95% | 95.91% | 10000 |
| whitebox_pgd_0.3 | 92.77% | 92.68% | 10000 |
| whitebox_pgd100x5_0.3_subset | 90.90% | 90.97% | 1000 |

## Backdoor poisoning and sanitization

| Experiment | Scenario | Accuracy | Attack success rate | Detection precision | Detection recall |
|---|---|---:|---:|---:|---:|
| backdoor_poisoning | poison_rate_0.005_clean | 99.05% | — | — | — |
| backdoor_poisoning | poison_rate_0.005_triggered | 0.20% | 99.80% | — | — |
| backdoor_poisoning | poison_rate_0.01_clean | 99.15% | — | — | — |
| backdoor_poisoning | poison_rate_0.01_triggered | 0.01% | 99.99% | — | — |
| backdoor_poisoning | poison_rate_0.05_clean | 98.97% | — | — | — |
| backdoor_poisoning | poison_rate_0.05_triggered | 0.00% | 100.00% | — | — |
| backdoor_sanitization_ICA | clean | 98.55% | — | 0.00% | 0.00% |
| backdoor_sanitization_ICA | triggered | 0.01% | 99.99% | 0.00% | 0.00% |
| backdoor_sanitization_PCA | clean | 98.92% | — | 100.00% | 100.00% |
| backdoor_sanitization_PCA | triggered | 98.88% | 0.07% | 100.00% | 100.00% |

## Data consistency checks

- All available confusion-matrix totals and derived accuracies are consistent with the saved metrics.

## Interpretation notes

- Accuracy on clean inputs alone does not characterize robustness to adversarial inputs or backdoor triggers.
- Attack success rate is reported as stored by the experiment scripts; confirm its target-label definition when describing it in the final report.
- The PCA and ICA sanitization results differ substantially; report both rather than selecting only the stronger result.
- Some saved evaluations use subsets while others use the full test set. State sample counts when comparing results.

## Robustness figure

Generated chart: `robustness_comparison.svg`
