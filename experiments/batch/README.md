# Batch evaluation

| Cohort | Design | Reader files |
|---|---|---|
| Positive | 200 registered xBD assets, 25 fixed-seed attack chains per asset | [Judgements](positive/judgements.csv), [metrics](positive/metrics.csv), [selected assets](positive/final_samples.csv), [attack settings](positive/attack_config.json) |
| Negative | 500 separately selected registered assets, six wrong-image inputs per asset | [Judgements](negative/judgements.csv), [metrics](negative/metrics.csv), [pairings](negative/negative_pairings_materialized.csv), [targets](negative/negative_targets.csv) |
| Configuration check | The same 500 positive-control tasks assessed under three model configurations | [Subset and per-model decisions](model_comparison/) |

Within each cohort, use the task ID to look up the panel and response. Join the positive metric table to the judgements on asset ID and attack name. Join negative records on negative ID. A model match is a true positive in the positive cohort and a false alarm in the negative cohort.

Each panel compares the registered reference on the left with the extracted watermark on the right. The JSON response files retain the inference settings, scientific response and timing. Service request IDs, endpoints, credential-variable names and workstation run directories have been removed from the publication copies. See the [prompt](positive/prompt.txt) and [main positive-run settings](positive/request_profile.json).
