import pandas as pd, numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

np.random.seed(42)
sns.set_style("whitegrid")

df = pd.read_csv("/mnt/user-data/uploads/Raw_Data.csv")
gene_cols_all = df.columns[1:-2].tolist()
gene_cols = [c for c in gene_cols_all if df[c].sum() > 0]
X_full = df[gene_cols].values

classes_all = ['Amoxicillin-Clavulanic Acid','Ampicillin','Ceftriaxone','Chloramphenicol','Ciprofloxacin',
               'Gentamicin','Nalidixic Acid','Streptomycin','Sulfamethoxazole-Sulfisoxazole',
               'Tetracycline','Trimethoprim-Sulfamethoxazole']
dc = df['drug classes'].fillna('NA')
label_matrix = pd.DataFrame({c: dc.apply(lambda s, c=c: int(c in s)) for c in classes_all})

models_def = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "SVM (RBF)": lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
rf_importances = pd.DataFrame(index=gene_cols)

for drug in classes_all:
    y = label_matrix[drug].values
    if y.sum() < 10:  # skip too-rare classes for stable eval
        continue
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, stratify=y, random_state=42)
    for mname, mfun in models_def.items():
        model = mfun()
        try:
            cvacc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
        except Exception:
            cvacc = np.nan
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        rows.append(dict(
            drug_class=drug, model=mname, n_positive=int(y.sum()), n_total=len(y),
            cv_accuracy=cvacc,
            test_accuracy=accuracy_score(y_test,y_pred),
            precision=precision_score(y_test,y_pred, zero_division=0),
            recall=recall_score(y_test,y_pred, zero_division=0),
            f1=f1_score(y_test,y_pred, zero_division=0),
            auc=roc_auc_score(y_test,y_prob) if len(set(y_test))>1 else np.nan
        ))
        if mname=="Random Forest":
            rf_importances[drug] = model.feature_importances_

res = pd.DataFrame(rows)
res.to_csv("/home/claude/amr_project/multilabel_results.csv", index=False)
print(res.round(3).to_string())

# Aggregate by model
agg = res.groupby("model")[["test_accuracy","precision","recall","f1","auc"]].mean().round(3)
agg.to_csv("/home/claude/amr_project/multilabel_model_summary.csv")
print("\nMODEL AVERAGES ACROSS 10 DRUG CLASSES:\n", agg)

# Heatmap of F1 per drug per model
pivot_f1 = res.pivot(index="drug_class", columns="model", values="f1")
plt.figure(figsize=(7,6))
sns.heatmap(pivot_f1, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
plt.title("F1-score by Drug Class and Model")
plt.tight_layout()
plt.savefig("/home/claude/amr_project/figures/f1_heatmap.png", dpi=300)
plt.close()

# Global gene importance (mean across drug-specific RF models)
mean_imp = rf_importances.mean(axis=1).sort_values(ascending=False).head(15)
plt.figure(figsize=(7,5.5))
sns.barplot(x=mean_imp.values, y=mean_imp.index, color="#55A868")
plt.xlabel("Mean Random Forest Importance (across 10 drug-class models)")
plt.title("Top 15 AMR Genes — Overall Predictive Importance")
plt.tight_layout()
plt.savefig("/home/claude/amr_project/figures/global_feature_importance.png", dpi=300)
plt.close()
mean_imp.to_csv("/home/claude/amr_project/global_top_features.csv")

# Bar chart of mean AUC per model
plt.figure(figsize=(5.5,4.5))
sns.barplot(x=agg.index, y=agg["auc"], palette="Set2")
plt.ylabel("Mean AUC (10 drug classes)")
plt.title("Overall Model Comparison — Mean AUC")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("/home/claude/amr_project/figures/mean_auc_comparison.png", dpi=300)
plt.close()

print("\nGenes retained:", len(gene_cols), "| Drug classes modeled:", res['drug_class'].nunique())
