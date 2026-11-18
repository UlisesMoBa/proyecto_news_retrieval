
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

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
# 2. CARGAR CORPUS
# ============================================================

print("=" * 65)
print("EXPERIMENTOS DE CLASIFICACIÓN SUPERVISADA")
print("=" * 65)

print("\nCargando corpus oficial...")

df = pd.read_csv(INPUT_FILE, sep="\t")

required_columns = {
    "document_id",
    "text",
    "category",
    "split"
}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Faltan columnas obligatorias: "
        f"{required_columns - set(df.columns)}"
    )

df["text"] = df["text"].fillna("").astype(str)

if df["category"].isna().any():
    raise ValueError("Existen documentos sin categoría.")

if df["document_id"].duplicated().any():
    raise ValueError("Existen document_id duplicados.")

train_df = df[df["split"] == "train"].copy()
dev_df = df[df["split"] == "dev"].copy()

if train_df.empty or dev_df.empty:
    raise ValueError("No se encontraron las particiones train o dev.")

print(f"Total de documentos: {len(df)}")
print(f"Train: {len(train_df)}")
print(f"Dev:   {len(dev_df)}")

print("\nDistribución de categorías en TRAIN:")
print(train_df["category"].value_counts())

print("\nDistribución de categorías en DEV:")
print(dev_df["category"].value_counts())


# ============================================================
# 3. PREPARAR VARIABLES
# ============================================================

X_train_text = train_df["text"]
y_train = train_df["category"]

X_dev_text = dev_df["text"]
y_dev = dev_df["category"]

classes = sorted(y_train.unique())

unknown_classes = set(y_dev.unique()) - set(classes)

if unknown_classes:
    raise ValueError(
        f"Hay categorías en DEV que no aparecen en TRAIN: "
        f"{unknown_classes}"
    )


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

print("Ajustando vocabulario exclusivamente con TRAIN...")

X_train = vectorizer.fit_transform(X_train_text)

print("Transformando DEV...")

X_dev = vectorizer.transform(X_dev_text)

print(f"Vocabulario: {len(vectorizer.vocabulary_)}")
print(f"Dimensiones TRAIN: {X_train.shape}")
print(f"Dimensiones DEV:   {X_dev.shape}")

empty_train = int(np.sum(np.asarray(X_train.getnnz(axis=1)) == 0))
empty_dev = int(np.sum(np.asarray(X_dev.getnnz(axis=1)) == 0))

print(f"Documentos sin términos TRAIN: {empty_train}")
print(f"Documentos sin términos DEV:   {empty_dev}")


# ============================================================
# 5. DEFINIR MODELOS
# ============================================================

models = {
    "NB_Multinomial": MultinomialNB(
        alpha=1.0
    ),

    "LR_Standard": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "LR_Balanced": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "SVM_Standard": LinearSVC(
        random_state=RANDOM_STATE
    ),

    "SVM_Balanced": LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
}


# ============================================================
# 6. ENTRENAMIENTO Y EVALUACIÓN EN DEV
# ============================================================

print("\nIniciando experimentos...")

metrics_results = []
prediction_results = []
report_results = []
confusion_results = []

for experiment_id, model in models.items():

    print("\n" + "-" * 65)
    print(f"Experimento: {experiment_id}")
    print("-" * 65)

    # --------------------------------------------------------
    # Entrenamiento exclusivamente con TRAIN
    # --------------------------------------------------------

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predicciones en DEV
    # --------------------------------------------------------

    y_pred = model.predict(X_dev)

    # --------------------------------------------------------
    # Métricas generales
    # --------------------------------------------------------

    accuracy = accuracy_score(y_dev, y_pred)

    precision_macro = precision_score(
        y_dev,
        y_pred,
        labels=classes,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_dev,
        y_pred,
        labels=classes,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_dev,
        y_pred,
        labels=classes,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        y_dev,
        y_pred,
        labels=classes,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_dev,
        y_pred,
        labels=classes,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_dev,
        y_pred,
        labels=classes,
        average="weighted",
        zero_division=0
    )

    metrics_results.append({
        "experiment_id": experiment_id,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "train_documents": len(train_df),
        "dev_documents": len(dev_df),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "empty_train": empty_train,
        "empty_dev": empty_dev
    })

    # --------------------------------------------------------
    # Guardar predicciones
    # --------------------------------------------------------

    for i, (_, row) in enumerate(dev_df.iterrows()):

        prediction_results.append({
            "experiment_id": experiment_id,
            "document_id": row["document_id"],
            "true_category": y_dev.iloc[i],
            "predicted_category": y_pred[i],
            "correct": y_dev.iloc[i] == y_pred[i]
        })

    # --------------------------------------------------------
    # Métricas individuales por categoría
    # --------------------------------------------------------

    report = classification_report(
        y_dev,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0
    )

    for category in classes:

        report_results.append({
            "experiment_id": experiment_id,
            "category": category,
            "precision": report[category]["precision"],
            "recall": report[category]["recall"],
            "f1_score": report[category]["f1-score"],
            "support": report[category]["support"]
        })

    # --------------------------------------------------------
    # Matriz de confusión
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_dev,
        y_pred,
        labels=classes
    )

    for i, true_category in enumerate(classes):

        for j, predicted_category in enumerate(classes):

            confusion_results.append({
                "experiment_id": experiment_id,
                "true_category": true_category,
                "predicted_category": predicted_category,
                "documents": int(matrix[i, j])
            })

    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision Macro:   {precision_macro:.4f}")
    print(f"Recall Macro:      {recall_macro:.4f}")
    print(f"F1 Macro:          {f1_macro:.4f}")
    print(f"Precision Weighted:{precision_weighted:.4f}")
    print(f"Recall Weighted:   {recall_weighted:.4f}")
    print(f"F1 Weighted:       {f1_weighted:.4f}")


# ============================================================
# 7. GUARDAR RESULTADOS
# ============================================================

print("\nGuardando resultados...")

metrics_df = pd.DataFrame(metrics_results)

predictions_df = pd.DataFrame(prediction_results)

report_df = pd.DataFrame(report_results)

confusion_df = pd.DataFrame(confusion_results)

# Ordenar experimentos por Macro F1
metrics_df = metrics_df.sort_values(
    by="f1_macro",
    ascending=False
)

metrics_df.to_csv(
    OUTPUT_DIR / "classification_dev_metrics.tsv",
    sep="\t",
    index=False,
    encoding="utf-8"
)

predictions_df.to_csv(
    OUTPUT_DIR / "classification_dev_predictions.tsv",
    sep="\t",
    index=False,
    encoding="utf-8"
)

report_df.to_csv(
    OUTPUT_DIR / "classification_dev_report.tsv",
    sep="\t",
    index=False,
    encoding="utf-8"
)

confusion_df.to_csv(
    OUTPUT_DIR / "classification_dev_confusion_matrix.tsv",
    sep="\t",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 8. RESUMEN FINAL
# ============================================================

print("\n" + "=" * 65)
print("RESUMEN DE EXPERIMENTOS")
print("=" * 65)

print(
    metrics_df[
        [
            "experiment_id",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted"
        ]
    ].to_string(index=False)
)

best_model = metrics_df.iloc[0]

print("\nMejor experimento según F1 Macro en DEV:")
print(f"Modelo: {best_model['experiment_id']}")
print(f"F1 Macro: {best_model['f1_macro']:.4f}")

print("\nArchivos generados:")
print("- classification_dev_metrics.tsv")
print("- classification_dev_predictions.tsv")
print("- classification_dev_report.tsv")
print("- classification_dev_confusion_matrix.tsv")

print("\nEvaluación completada.")
print("IMPORTANTE: TEST todavía no ha sido utilizado.")
