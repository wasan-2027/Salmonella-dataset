# ML Prediction of Antibiotic-Class Resistance in Poultry-Associated *Salmonella enterica*

Code and processed data accompanying the manuscript:
**"Machine Learning-Based Prediction of Antibiotic Class-Level Resistance from Genomic Marker Profiles in Poultry-Associated Salmonella enterica"**

## Contents
- `data/Raw_Data.csv` — merged AMR-gene presence/absence matrix (299 isolates x 186 genes) and drug-class labels, sourced from the public Mendeley dataset (see Data Source below), used as input to the pipeline.
- `data/multilabel_results.csv` — per-antibiotic-class, per-model performance metrics (accuracy, precision, recall, F1, AUC) from the held-out test set.
- `data/multilabel_model_summary.csv` — metrics averaged across the 11 antibiotic classes, by algorithm.
- `data/global_top_features.csv` — mean Random Forest feature importances across the 11 per-class models.
- `code/analysis_pipeline.py` — full, reproducible analysis script (preprocessing, model training, evaluation, figure generation).
- `figures/` — publication figures (F1 heatmap, mean AUC comparison, global feature importance).

## Data Source
Raw genomic AMR-gene annotations originate from:
> Salmonella enterica Multi-Drug Resistance in Chicken Genomes: A Dataset for Machine Learning Applications. Mendeley Data, V1, 2024. https://doi.org/10.17632/523gzcwhzb.1

## Reproducing the analysis
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
python code/analysis_pipeline.py
```

## Citation
If you use this code or the processed outputs, please cite the manuscript (details to be added upon publication) and the original dataset above.

## License
Code released under the MIT License (see LICENSE).
