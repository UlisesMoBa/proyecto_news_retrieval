
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "labeled_documents_split.tsv"
)

OUTPUT_DIR = BASE_DIR / "results" / "kmeans"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_ID = "KFINAL_K5"

OUTPUT_METRICS = (
    OUTPUT_DIR / "final_kmeans_metrics.tsv"
)

OUTPUT_TERMS = (
    OUTPUT_DIR / "final_kmeans_cluster_terms.tsv"
)

OUTPUT_REPRESENTATIVES = (
    OUTPUT_DIR / "final_kmeans_representative_documents.tsv"
)

OUTPUT_ASSIGNMENTS = (
    OUTPUT_DIR / "final_kmeans_assignments.tsv"
)

OUTPUT_CATEGORY_DISTRIBUTION = (
    OUTPUT_DIR / "final_kmeans_category_distribution.tsv"
)

OUTPUT_CLUSTER_DISTRIBUTION = (
    OUTPUT_DIR / "final_kmeans_cluster_distribution.tsv"
)

OUTPUT_EMPTY = (
    OUTPUT_DIR / "final_kmeans_empty_documents.tsv"
)

EXPERIMENT_LOG = BASE_DIR / "experiments.tsv"

RANDOM_STATE = 0
K = 5

TOP_TERMS = 15
TOP_DOCUMENTS = 5

EXPECTED_SPLIT_COUNTS = {
    "train": 281,
    "dev": 60,
    "test": 61
}


# ============================================================
# 2. STOPWORDS EN ESPAÑOL
# ============================================================

# Se conserva exactamente la lista utilizada en K02.

SPANISH_STOPWORDS = [
    "a", "al", "algo", "algunas", "algunos", "ante", "antes",
    "como", "con", "contra", "cual", "cuando", "de", "del",
    "desde", "donde", "durante", "e", "el", "ella", "ellas",
    "ello", "ellos", "en", "entre", "era", "erais", "eran",
    "eras", "eres", "es", "esa", "esas", "ese", "eso",
    "esos", "esta", "estaba", "estaban", "estado", "estamos",
    "estan", "estar", "estas", "este", "esto", "estos",
    "fue", "fueron", "ha", "habia", "han", "hasta", "hay",
    "la", "las", "le", "les", "lo", "los", "mas", "me",
    "mi", "mis", "mientras", "muy", "nada", "ni", "no",
    "nos", "nosotros", "nuestra", "nuestras", "nuestro",
    "nuestros", "o", "os", "otra", "otras", "otro", "otros",
    "para", "pero", "poco", "por", "porque", "que", "quien",
    "quienes", "se", "sea", "ser", "si", "sin", "sobre",
    "son", "su", "sus", "tambien", "te", "tiene", "tienen",
    "todo", "todos", "tu", "tus", "un", "una", "uno",
    "unos", "unas", "y", "ya",

    "ano", "anos", "hoy", "ayer", "manana",
    "mexico", "mexicana", "mexicano", "mexicanos",
    "pais", "paises", "ciudad", "estados",
    "ciento", "mil", "millones", "pesos",
    "dijo", "dice", "senalo", "segun",
    "ademas", "asi", "tras", "cada",
    "puede", "pueden", "podria", "seria",
    "parte", "caso", "nuevo", "nueva",
    "nuevos", "nuevas", "tambien"
]


# ============================================================
# 3. FUNCIONES AUXILIARES
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def calculate_silhouette(X, labels):
    """
    Calcula silhouette para un conjunto de documentos.

    Solo se utiliza como métrica descriptiva.
    No se utiliza para modificar el modelo final.
    """

    if X.shape[0] < 2:
        return np.nan

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return np.nan

    if len(unique_labels) >= X.shape[0]:
        return np.nan

    return float(
        silhouette_score(
            X,
            labels,
            metric="euclidean"
        )
    )


def get_top_terms(model, feature_names, top_n=15):

    results = []

    for cluster_id, center in enumerate(
        model.cluster_centers_
    ):

        top_indices = center.argsort()[::-1][:top_n]

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            results.append({
                "experiment_id": EXPERIMENT_ID,
                "cluster_id": cluster_id,
                "rank": rank,
                "term": feature_names[index],
                "weight": float(center[index])
            })

    return results


def append_experiment_log(new_result):
    """
    Agrega el experimento final al registro general.

    Solo reemplaza el identificador KFINAL_K5 si ya existe.
    Conserva los demás experimentos.
    """

    new_df = pd.DataFrame([new_result])

    if EXPERIMENT_LOG.exists():

        old_df = pd.read_csv(
            EXPERIMENT_LOG,
            sep="\t"
        )

        if "experiment_id" in old_df.columns:

            old_df = old_df[
                old_df["experiment_id"] != EXPERIMENT_ID
            ]

        combined = pd.concat(
            [old_df, new_df],
            ignore_index=True,
            sort=False
        )

    else:
        combined = new_df

    combined.to_csv(
        EXPERIMENT_LOG,
        sep="\t",
        index=False,
        encoding="utf-8"
    )


# ============================================================
# 4. CARGAR Y VALIDAR CORPUS OFICIAL
# ============================================================

print("=" * 70)
print("EVALUACIÓN FINAL - K-MEANS K=5")
print("=" * 70)

print("\nCargando corpus oficial...")

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"No se encontró el archivo:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    encoding="utf-8"
)

required_columns = [
    "document_id",
    "split",
    "category"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Falta la columna obligatoria: {column}"
        )

if "normalized_text" in df.columns:

    TEXT_COLUMN = "normalized_text"

elif "text" in df.columns:

    TEXT_COLUMN = "text"

else:

    raise ValueError(
        "No se encontró normalized_text ni text."
    )


df["document_id"] = df["document_id"].astype(str)

df["split"] = (
    df["split"]
    .astype(str)
    .str.lower()
)

df["category"] = (
    df["category"]
    .fillna("Sin categoría")
)

df["input_text"] = (
    df[TEXT_COLUMN]
    .apply(clean_text)
)


# Validar identificadores únicos.

if df["document_id"].duplicated().any():

    raise ValueError(
        "Se encontraron document_id duplicados."
    )


# Validar particiones.

if set(df["split"].unique()) != {
    "train", "dev", "test"
}:

    raise ValueError(
        "Las particiones deben ser train, dev y test."
    )


split_counts = (
    df["split"]
    .value_counts()
    .to_dict()
)

print(f"\nTotal de documentos: {len(df)}")

print("\nDistribución oficial:")

for split_name in ["train", "dev", "test"]:

    actual = split_counts.get(split_name, 0)
    expected = EXPECTED_SPLIT_COUNTS[split_name]

    print(
        f"{split_name.upper():5s}: {actual}"
    )

    if actual != expected:

        raise ValueError(
            f"Cantidad inesperada en {split_name}: "
            f"{actual}. Se esperaban {expected}."
        )


train_df = (
    df[df["split"] == "train"]
    .copy()
    .reset_index(drop=True)
)

dev_df = (
    df[df["split"] == "dev"]
    .copy()
    .reset_index(drop=True)
)

test_df = (
    df[df["split"] == "test"]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# 5. VECTORIZACIÓN TF-IDF
# ============================================================

print("\nConfigurando TF-IDF...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words=SPANISH_STOPWORDS,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85,
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9]+\b"
)


# IMPORTANTE:
# El vocabulario se ajusta exclusivamente con TRAIN.

print("Ajustando TF-IDF exclusivamente con TRAIN...")

X_train_all = vectorizer.fit_transform(
    train_df["input_text"]
)


# DEV y TEST únicamente se transforman.
# No se ajusta nuevamente el vocabulario.

X_dev_all = vectorizer.transform(
    dev_df["input_text"]
)

X_test_all = vectorizer.transform(
    test_df["input_text"]
)

feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulario: {len(feature_names)}")


# ============================================================
# 6. IDENTIFICAR DOCUMENTOS VACÍOS
# ============================================================

print("\nIdentificando documentos sin términos...")

split_data = {
    "train": (train_df, X_train_all),
    "dev": (dev_df, X_dev_all),
    "test": (test_df, X_test_all)
}

valid_data = {}
empty_rows = []

for split_name, (split_df, X_all) in split_data.items():

    valid_mask = np.asarray(
        X_all.getnnz(axis=1)
    ).ravel() > 0

    valid_df = (
        split_df.loc[valid_mask]
        .copy()
        .reset_index(drop=True)
    )

    X_valid = X_all[valid_mask]

    valid_data[split_name] = {
        "df": valid_df,
        "X": X_valid,
        "mask": valid_mask
    }

    invalid_df = split_df.loc[~valid_mask]

    for _, row in invalid_df.iterrows():

        empty_rows.append({
            "document_id": row["document_id"],
            "split": split_name,
            "category": row["category"],
            "reason": "Sin términos después de TF-IDF"
        })

    print(
        f"{split_name.upper():5s}: "
        f"{X_valid.shape[0]} válidos / "
        f"{len(split_df)} documentos"
    )


X_train = valid_data["train"]["X"]

if X_train.shape[0] < K:

    raise ValueError(
        "No hay suficientes documentos válidos en TRAIN."
    )


# ============================================================
# 7. ENTRENAMIENTO DEL MODELO FINAL
# ============================================================

print("\nEntrenando modelo final...")

model = KMeans(
    n_clusters=K,
    init="k-means++",
    n_init=20,
    max_iter=300,
    random_state=RANDOM_STATE
)


# El modelo se ajusta exclusivamente con TRAIN.

train_labels = model.fit_predict(X_train)

print("\nModelo entrenado.")
print(f"Inercia TRAIN: {model.inertia_:.6f}")


# ============================================================
# 8. ASIGNACIÓN DE TRAIN, DEV Y TEST
# ============================================================

print("\nAsignando documentos a los clústeres...")

labels_by_split = {
    "train": train_labels
}

for split_name in ["dev", "test"]:

    X_valid = valid_data[split_name]["X"]

    if X_valid.shape[0] > 0:

        labels_by_split[split_name] = (
            model.predict(X_valid)
        )

    else:

        labels_by_split[split_name] = np.array(
            [],
            dtype=int
        )


# ============================================================
# 9. MÉTRICAS POR PARTICIÓN
# ============================================================

print("\nCalculando métricas...")

metrics_results = []

for split_name in ["train", "dev", "test"]:

    X_valid = valid_data[split_name]["X"]
    labels = labels_by_split[split_name]

    silhouette = calculate_silhouette(
        X_valid,
        labels
    )

    metrics_results.append({
        "experiment_id": EXPERIMENT_ID,
        "algorithm": "KMeans",
        "k": K,
        "split": split_name,
        "representation": "TF-IDF",
        "ngram_range": "1,2",
        "vocabulary_size": len(feature_names),
        "total_documents": len(split_data[split_name][0]),
        "valid_documents": X_valid.shape[0],
        "empty_documents": (
            len(split_data[split_name][0])
            - X_valid.shape[0]
        ),
        "observed_clusters": (
            len(np.unique(labels))
            if len(labels) > 0
            else 0
        ),
        "inertia": (
            float(model.inertia_)
            if split_name == "train"
            else np.nan
        ),
        "silhouette": silhouette,
        "random_state": RANDOM_STATE
    })

    if np.isnan(silhouette):

        print(
            f"{split_name.upper()}: "
            "Silhouette no disponible."
        )

    else:

        print(
            f"{split_name.upper()}: "
            f"Silhouette = {silhouette:.6f}"
        )


# ============================================================
# 10. TÉRMINOS REPRESENTATIVOS
# ============================================================

terms_results = get_top_terms(
    model,
    feature_names,
    TOP_TERMS
)


# ============================================================
# 11. ASIGNACIONES DE TODOS LOS DOCUMENTOS
# ============================================================

print("\nGenerando asignaciones completas...")

assignment_results = []

for split_name in ["train", "dev", "test"]:

    original_df, X_all = split_data[split_name]

    valid_info = valid_data[split_name]

    valid_mask = valid_info["mask"]

    labels = labels_by_split[split_name]

    # Inicialmente todos los documentos quedan sin asignar.
    output = original_df[
        [
            "document_id",
            "split",
            "category"
        ]
    ].copy()

    output["cluster_id"] = -1

    output["distance_to_centroid"] = np.nan

    output["assignment_status"] = (
        "Sin términos después de TF-IDF"
    )

    output["experiment_id"] = EXPERIMENT_ID

    # Asignar únicamente documentos válidos.
    if len(labels) > 0:

        X_valid = valid_info["X"]

        # Distancia de cada documento a todos los centroides.
        distances = model.transform(X_valid)

        # Distancia al centroide asignado.
        assigned_distances = distances[
            np.arange(len(labels)),
            labels
        ]

        valid_positions = np.where(valid_mask)[0]

        output.loc[
            valid_positions,
            "cluster_id"
        ] = labels

        output.loc[
            valid_positions,
            "distance_to_centroid"
        ] = assigned_distances

        output.loc[
            valid_positions,
            "assignment_status"
        ] = "Asignado"

    assignment_results.append(output)


assignments_df = pd.concat(
    assignment_results,
    ignore_index=True
)


# ============================================================
# 12. DISTRIBUCIÓN DE DOCUMENTOS POR CLÚSTER
# ============================================================

cluster_distribution_results = []

category_distribution_results = []

for split_name in ["train", "dev", "test"]:

    valid_df = valid_data[split_name]["df"]

    labels = labels_by_split[split_name]

    if len(labels) == 0:
        continue

    temp = valid_df[
        ["document_id", "category"]
    ].copy()

    temp["cluster_id"] = labels

    total_valid = len(temp)

    # Distribución de tamaños de clúster.
    for cluster_id in range(K):

        cluster_count = int(
            np.sum(labels == cluster_id)
        )

        cluster_distribution_results.append({
            "experiment_id": EXPERIMENT_ID,
            "split": split_name,
            "cluster_id": cluster_id,
            "documents": cluster_count,
            "percentage": (
                cluster_count / total_valid * 100
                if total_valid > 0
                else 0
            )
        })

    # Distribución de categorías por clúster.
    category_counts = (
        temp.groupby(
            ["cluster_id", "category"]
        )
        .size()
        .reset_index(name="documents")
    )

    category_counts["experiment_id"] = EXPERIMENT_ID
    category_counts["split"] = split_name

    category_distribution_results.append(
        category_counts
    )


# ============================================================
# 13. DOCUMENTOS REPRESENTATIVOS
# ============================================================

print("\nIdentificando documentos representativos...")

representative_results = []

# Los representantes se obtienen exclusivamente de TRAIN.
# TEST no interviene en esta selección.

train_distances = model.transform(X_train)

for cluster_id in range(K):

    cluster_indices = np.where(
        train_labels == cluster_id
    )[0]

    ordered_indices = cluster_indices[
        np.argsort(
            train_distances[
                cluster_indices,
                cluster_id
            ]
        )
    ]

    selected_indices = ordered_indices[
        :TOP_DOCUMENTS
    ]

    for rank, index in enumerate(
        selected_indices,
        start=1
    ):

        row = valid_data["train"]["df"].iloc[index]

        representative_results.append({
            "experiment_id": EXPERIMENT_ID,
            "cluster_id": cluster_id,
            "rank": rank,
            "document_id": row["document_id"],
            "category": row["category"],
            "distance_to_centroid": float(
                train_distances[index, cluster_id]
            ),
            "text": row["input_text"]
        })


# ============================================================
# 14. GUARDAR RESULTADOS
# ============================================================

print("\nGuardando resultados finales...")

pd.DataFrame(metrics_results).to_csv(
    OUTPUT_METRICS,
    sep="\t",
    index=False,
    encoding="utf-8"
)

pd.DataFrame(terms_results).to_csv(
    OUTPUT_TERMS,
    sep="\t",
    index=False,
    encoding="utf-8"
)

pd.DataFrame(representative_results).to_csv(
    OUTPUT_REPRESENTATIVES,
    sep="\t",
    index=False,
    encoding="utf-8"
)

assignments_df.to_csv(
    OUTPUT_ASSIGNMENTS,
    sep="\t",
    index=False,
    encoding="utf-8"
)

if category_distribution_results:

    pd.concat(
        category_distribution_results,
        ignore_index=True
    ).to_csv(
        OUTPUT_CATEGORY_DISTRIBUTION,
        sep="\t",
        index=False,
        encoding="utf-8"
    )

pd.DataFrame(
    cluster_distribution_results
).to_csv(
    OUTPUT_CLUSTER_DISTRIBUTION,
    sep="\t",
    index=False,
    encoding="utf-8"
)

pd.DataFrame(empty_rows).to_csv(
    OUTPUT_EMPTY,
    sep="\t",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 15. REGISTRO DEL EXPERIMENTO FINAL
# ============================================================

train_silhouette = metrics_results[0]["silhouette"]
dev_silhouette = metrics_results[1]["silhouette"]
test_silhouette = metrics_results[2]["silhouette"]

final_log = {
    "experiment_id": EXPERIMENT_ID,
    "algorithm": "KMeans",
    "representation": "TF-IDF",
    "preprocessing": "normalized_text + Spanish stopwords",
    "ngram_range": "1,2",
    "k": K,
    "min_df": 2,
    "max_df": 0.85,
    "max_features": 3000,
    "sublinear_tf": True,
    "norm": "l2",
    "n_init": 20,
    "max_iter": 300,
    "random_state": RANDOM_STATE,
    "inertia": float(model.inertia_),
    "silhouette_train": train_silhouette,
    "silhouette_dev": dev_silhouette,
    "silhouette_test": test_silhouette,
    "selected": True,
    "notes": (
        "Configuracion final congelada con K=5. "
        "Vectorizador y KMeans ajustados exclusivamente "
        "con TRAIN. DEV utilizado para seleccion previa. "
        "TEST evaluado despues de congelar la configuracion."
    )
}

append_experiment_log(final_log)


# ============================================================
# 16. RESUMEN FINAL
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN DE EVALUACIÓN FINAL")
print("=" * 70)

metrics_df = pd.DataFrame(metrics_results)

print(
    metrics_df[
        [
            "split",
            "total_documents",
            "valid_documents",
            "empty_documents",
            "observed_clusters",
            "inertia",
            "silhouette"
        ]
    ].to_string(index=False)
)

print("\nTamaños de clústeres:")

cluster_df = pd.DataFrame(
    cluster_distribution_results
)

print(
    cluster_df.pivot(
        index="cluster_id",
        columns="split",
        values="documents"
    ).fillna(0).to_string()
)

print("\nDocumentos asignados:")

print(
    assignments_df["assignment_status"]
    .value_counts()
    .to_string()
)

print("\nArchivos generados:")

for path in [
    OUTPUT_METRICS,
    OUTPUT_TERMS,
    OUTPUT_REPRESENTATIVES,
    OUTPUT_ASSIGNMENTS,
    OUTPUT_CATEGORY_DISTRIBUTION,
    OUTPUT_CLUSTER_DISTRIBUTION,
    OUTPUT_EMPTY,
    EXPERIMENT_LOG
]:

    print(f"- {path.relative_to(BASE_DIR)}")


print("\nEvaluación final terminada correctamente.")
