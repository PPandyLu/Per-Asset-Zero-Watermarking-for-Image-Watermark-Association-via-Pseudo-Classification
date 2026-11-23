# Figures and source data

| Paper figure | Plot | Source data |
|---|---|---|
| 22 | [Objective-metric ROC](plots/fig22_objective_metric_roc.png) | [6,344 held-out scores](roc/held_out_scores.csv), [curve coordinates](roc/roc_points.csv), [AUCs](roc/auc_summary.csv) |
| 23 | [Batch validation by condition](plots/fig23_batch_validation_stratification.png) | [Positive results by chain length](stratified_statistics/positive_by_chain_length.csv), [negative results by relationship and condition](stratified_statistics/negative_by_relationship_condition.csv) |
| 24 | [Attack-family responses](plots/fig24_attack_family_forest_plot.png) | [Positive results by attack-family presence](stratified_statistics/positive_by_attack_family_presence.csv) |
| 25 | [Crop retention](plots/fig25_crop_response.png) | [Match rates by retained area](../experiments/controlled/crop_spatial/results/crop_response_by_retain_percent.csv) |
| 26 | [Direct reconstruction metrics](plots/fig26_direct_metric_response.png) | [Plotted means](publication_figures/Fig04_direct_metric_response_data.csv), [all 45 settings](../experiments/controlled/direct_metric_summary/direct_metric_response_by_level.csv) |
| 27 | [AI and collective human judgements](plots/fig27_ai_collective_blind_review_agreement.png) | [Representative-set judgements](../experiments/blind_review/representative_200/ratings.csv) |
| 28 | [Example decision cases](plots/fig28_representative_decision_cases.png) | [Representative-set case IDs and labels](../experiments/blind_review/representative_200/ratings.csv), [panels](../experiments/blind_review/representative_200/panels.zip) |

The ROC coordinates refer to the primary held-out split. The manuscript also reports the median across ten image-disjoint splits. In Figure 24 a compound attack can contribute to several family strata.
