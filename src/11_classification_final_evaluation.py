
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "labeled_documents_split.tsv"
)

OUTPUT_DIR = BASE_DIR / "results" / "classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0


# ============================================================
# 2. CARGAR CORPUS OFICIAL
# ============================================================

print("=" * 70)
print("EVALUACIÓN FINAL - CLASIFICACIÓN SUPERVISADA")
print("=" * 70)

print("\nCargando corpus oficial...")

df = pd.read_csv(INPUT_FILE, sep="\t")

df["text"] = df["text"].fillna("").astype(str)

required_columns = {
    "document_id",
    "text",
    "category",
    "split"
}

if not required_columns.issubset(df.columns):
    raise ValueError("Faltan columnas obligatorias.")

if df["category"].isna().any():
    raise ValueError("Existen documentos sin categoría.")

if df["document_id"].duplicated().any():
    raise ValueError("Existen document_id duplicados.")

train_df = df[df["split"] == "train"].copy()
dev_df = df[df["split"] == "dev"].copy()
test_df = df[df["split"] == "test"].copy()

if train_df.empty or dev_df.empty or test_df.empty:
    raise ValueError("Falta alguna partición oficial.")

print(f"Corpus total: {len(df)}")
print(f"TRAIN: {len(train_df)}")
print(f"DEV:   {len(dev_df)}")
print(f"TEST:  {len(test_df)}")


# ============================================================
# 3. PREPARAR DATOS
# ============================================================

print("\nUniendo TRAIN y DEV...")

train_dev_df = pd.concat(
    [train_df, dev_df],
    ignore_index=True
)

X_train_dev_text = train_dev_df["text"]
y_train_dev = train_dev_df["category"]

X_test_text = test_df["text"]
y_test = test_df["category"]

classes = sorted(y_train_dev.unique())

unknown_classes = set(y_test.unique()) - set(classes)

if unknown_classes:
    raise ValueError(
        f"Hay categorías en TEST que no aparecen en TRAIN + DEV: "
        f"{unknown_classes}"
    )

print(f"Documentos para entrenamiento final: {len(train_dev_df)}")
print(f"Documentos para evaluación: {len(test_df)}")

print("\nDistribución TRAIN + DEV:")
print(y_train_dev.value_counts())

print("\nDistribución TEST:")
print(y_test.value_counts())


# ============================================================
# 4. CONFIGURAR TF-IDF
# ============================================================

print("\nConfigurando TF-IDF...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words=None,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85,
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9]+\b"
)

print("Ajustando vocabulario con TRAIN + DEV...")

X_train_dev = vectorizer.fit_transform(X_train_dev_text)

print("Transformando TEST...")

X_test = vectorizer.transform(X_test_text)

print(f"Vocabulario final: {len(vectorizer.vocabulary_)}")
print(f"Dimensiones TRAIN + DEV: {X_train_dev.shape}")
print(f"Dimensiones TEST:        {X_test.shape}")

empty_train_dev = int(
    np.sum(np.asarray(X_train_dev.getnnz(axis=1)) == 0)
)

empty_test = int(
    np.sum(np.asarray(X_test.getnnz(axis=1)) == 0)
)

print(f"Documentos vacíos TRAIN + DEV: {empty_train_dev}")
print(f"Documentos vacíos TEST:        {empty_test}")


# ============================================================
# 5. ENTRENAR MODELO FINAL
# ============================================================

print("\nEntrenando Logistic Regression...")

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

model.fit(X_train_dev, y_train_dev)

print("Entrenamiento finalizado.")


# ============================================================
# 6. EVALUACIÓN SOBRE TEST
# ============================================================

print("\nEvaluando sobre TEST...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision_macro = precision_score(
    y_test,
    y_pred,
    labels=classes,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test,
    y_pred,
    labels=classes,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    y_pred,
    labels=classes,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    y_test,
    y_pred,
    labels=classes,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_test,
    y_pred,
    labels=classes,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    labels=classes,
    average="weighted",
    zero_division=0
)


# ============================================================
# 7. GUARDAR MÉTRICAS GENERALES
# ============================================================

metrics_df = pd.DataFrame([{
    "experiment_id": "FINAL_LR_BALANCED",
    "model": "LogisticRegression",
    "class_weight": "balanced",
    "accuracy": accuracy,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    "precision_weighted": precision_weighted,
    "recall_weighted": recall_weighted,
    "f1_weighted": f1_weighted,
    "train_dev_documents": len(train_dev_df),
    "test_documents": len(test_df),
    "vocabulary_size": len(vectorizer.vocabulary_),
    "empty_train_dev": empty_train_dev,
    "empty_test": empty_test
}])

metrics_df.to_csv(
    OUTPUT_DIR / "classification_final_metrics.tsv",
    sep="\t",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 8. REPORTE POR CATEGORÍA
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    labels=classes,
    output_dict=True,
    zero_division=0
)

report_results = []

for category in classes:

    report_results.append({
        "category": category,
        "precision": report[category]["precision"],
        "recall": report[category]["recall"],
        "f1_score": report[category]["f1-score"],
        "support": report[category]["support"]
    })

report_df = pd.DataFrame(report_results)

report_df.to_csv(
    OUTPUT_DIR / "classification_final_report.tsv",
    sep="\t",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 9. GUARDAR PREDICCIONES
# ============================================================

predictions_df = test_df[
    ["document_id", "category", "split"]
].copy()

predictions_df = predictions_df.rename(
    columns={"category": "true_category"}
)

predictions_df["predicted_category"] = y_pred

predictions_df["correct"] = (
    predictions_df["true_category"] ==
    predictions_df["predicted_category"]
)

predictions_df.to_csv(
    OUTPUT_DIR / "classification_final_predictions.tsv",
    sep="\t",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 10. MATRIZ DE CONFUSIÓN
# ============================================================

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

confusion_results = []

for i, true_category in enumerate(classes):

    for j, predicted_category in enumerate(classes):

        confusion_results.append({
            "true_category": true_category,
            "predicted_category": predicted_category,
            "documents": int(matrix[i, j])
        })

confusion_df = pd.DataFrame(confusion_results)

confusion_df.to_csv(
    OUTPUT_DIR / "classification_final_confusion_matrix.tsv",
    sep="\t",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 11. RESUMEN FINAL
# ============================================================

print("\n" + "=" * 70)
print("RESULTADOS FINALES SOBRE TEST")
print("=" * 70)

print(f"Accuracy:           {accuracy:.4f}")
print(f"Precision Macro:    {precision_macro:.4f}")
print(f"Recall Macro:       {recall_macro:.4f}")
print(f"F1 Macro:           {f1_macro:.4f}")
print(f"Precision Weighted: {precision_weighted:.4f}")
print(f"Recall Weighted:    {recall_weighted:.4f}")
print(f"F1 Weighted:        {f1_weighted:.4f}")

print("\nReporte por categoría:")
print(
    report_df.to_string(index=False)
)

print("\nArchivos generados:")

print("- classification_final_metrics.tsv")
print("- classification_final_report.tsv")
print("- classification_final_predictions.tsv")
print("- classification_final_confusion_matrix.tsv")

print("\nEvaluación final completada.")
