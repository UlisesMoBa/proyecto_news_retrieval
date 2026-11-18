
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

OUTPUT_METRICS = OUTPUT_DIR / "k04_cluster_metrics.tsv"
OUTPUT_TERMS = OUTPUT_DIR / "k04_cluster_terms.tsv"
OUTPUT_REPRESENTATIVES = (
    OUTPUT_DIR / "k04_representative_documents.tsv"
)
OUTPUT_ASSIGNMENTS = (
    OUTPUT_DIR / "k04_document_assignments.tsv"
)
OUTPUT_DISTRIBUTION = (
    OUTPUT_DIR / "k04_category_distribution.tsv"
)
OUTPUT_EMPTY = OUTPUT_DIR / "k04_empty_documents.tsv"

EXPERIMENT_LOG = BASE_DIR / "experiments.tsv"

RANDOM_STATE = 0

# ÚNICO CAMBIO EXPERIMENTAL RESPECTO A K02:
K_VALUES = [7, 8]

TOP_TERMS = 15
TOP_DOCUMENTS = 5


# ============================================================
# 2. STOPWORDS EN ESPAÑOL
# ============================================================

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

    # Palabras adicionales frecuentes en noticias
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

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return np.nan

    if len(unique_labels) >= X.shape[0]:
        return np.nan

    return silhouette_score(
        X,
        labels,
        metric="euclidean"
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
                "cluster_id": cluster_id,
                "rank": rank,
                "term": feature_names[index],
                "weight": float(center[index])
            })

    return results


def append_experiment_log(new_results):

    new_df = pd.DataFrame(new_results)

    if EXPERIMENT_LOG.exists():

        old_df = pd.read_csv(
            EXPERIMENT_LOG,
            sep="\t"
        )

        if "experiment_id" in old_df.columns:

            old_df = old_df[
                ~old_df["experiment_id"].isin(
                    new_df["experiment_id"]
                )
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
# 4. CARGAR CORPUS OFICIAL
# ============================================================

print("=" * 65)
print("EXPERIMENTO K04 - K-MEANS CON TF-IDF Y BIGRAMAS")
print("=" * 65)

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
df["split"] = df["split"].astype(str).str.lower()
df["category"] = df["category"].fillna("Sin categoría")
df["input_text"] = df[TEXT_COLUMN].apply(clean_text)

print(f"Total de documentos: {len(df)}")

print("\nDistribución oficial:")
print(df["split"].value_counts().to_string())

if set(df["split"].unique()) - {"train", "dev", "test"}:

    raise ValueError(
        "Se encontraron valores de split inesperados."
    )

train_df = df[df["split"] == "train"].copy()
dev_df = df[df["split"] == "dev"].copy()

print(f"\nTrain: {len(train_df)}")
print(f"Dev:   {len(dev_df)}")
print(f"Test:  {len(df[df['split'] == 'test'])}")

if train_df.empty or dev_df.empty:

    raise ValueError(
        "Train y Dev deben contener documentos."
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

print(
    "Ajustando vocabulario exclusivamente con TRAIN..."
)

X_train_all = vectorizer.fit_transform(
    train_df["input_text"]
)

X_dev_all = vectorizer.transform(
    dev_df["input_text"]
)

feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulario: {len(feature_names)}")


# ============================================================
# 6. IDENTIFICAR DOCUMENTOS VACÍOS
# ============================================================

train_valid_mask = np.asarray(
    X_train_all.getnnz(axis=1)
).ravel() > 0

dev_valid_mask = np.asarray(
    X_dev_all.getnnz(axis=1)
).ravel() > 0

train_valid = train_df.loc[
    train_valid_mask
].copy()

dev_valid = dev_df.loc[
    dev_valid_mask
].copy()

X_train = X_train_all[train_valid_mask]
X_dev = X_dev_all[dev_valid_mask]

print("\nDocumentos válidos después de TF-IDF:")

print(
    f"Train: {X_train.shape[0]} / {len(train_df)}"
)

print(
    f"Dev:   {X_dev.shape[0]} / {len(dev_df)}"
)

empty_rows = []

for original_df, valid_mask in [
    (train_df, train_valid_mask),
    (dev_df, dev_valid_mask)
]:

    invalid_df = original_df.loc[~valid_mask]

    for _, row in invalid_df.iterrows():

        empty_rows.append({
            "document_id": row["document_id"],
            "split": row["split"],
            "category": row["category"],
            "reason": "Sin términos después de TF-IDF"
        })

if X_train.shape[0] < max(K_VALUES):

    raise ValueError(
        "No hay suficientes documentos válidos en TRAIN."
    )

if X_dev.shape[0] == 0:
    print("Advertencia: no hay documentos válidos en DEV.")


# ============================================================
# 7. ENTRENAMIENTO Y COMPARACIÓN DE K-MEANS
# ============================================================

print("\nIniciando experimentos K-Means...")

metrics_results = []
terms_results = []
representative_results = []
assignment_results = []
distribution_results = []
experiment_log_results = []

for k in K_VALUES:

    print("\n" + "=" * 65)
    print(f"EXPERIMENTO K04_K{k}")
    print("=" * 65)

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=RANDOM_STATE
    )

    train_labels = model.fit_predict(X_train)

    if X_dev.shape[0] > 0:
        dev_labels = model.predict(X_dev)

    else:
        dev_labels = np.array([], dtype=int)

    train_silhouette = calculate_silhouette(
        X_train,
        train_labels
    )

    if len(dev_labels) > 0:

        dev_silhouette = calculate_silhouette(
            X_dev,
            dev_labels
        )

    else:
        dev_silhouette = np.nan

    cluster_sizes = np.bincount(
        train_labels,
        minlength=k
    )

    print(f"Inercia: {model.inertia_:.4f}")

    print(
        f"Silhouette TRAIN: {train_silhouette:.4f}"
        if not np.isnan(train_silhouette)
        else "Silhouette TRAIN: No disponible"
    )

    print(
        f"Silhouette DEV: {dev_silhouette:.4f}"
        if not np.isnan(dev_silhouette)
        else "Silhouette DEV: No disponible"
    )

    print(f"Tamaños de clústeres: {cluster_sizes.tolist()}")

    # --------------------------------------------------------
    # 7.1 MÉTRICAS GENERALES
    # --------------------------------------------------------

    metrics_results.append({
        "experiment_id": f"K04_K{k}",
        "algorithm": "KMeans",
        "k": k,
        "representation": "TF-IDF",
        "ngram_range": "1,2",
        "vocabulary_size": len(feature_names),
        "train_documents": X_train.shape[0],
        "dev_documents": X_dev.shape[0],
        "inertia": float(model.inertia_),
        "silhouette_train": train_silhouette,
        "silhouette_dev": dev_silhouette,
        "empty_train": int((~train_valid_mask).sum()),
        "empty_dev": int((~dev_valid_mask).sum()),
        "random_state": RANDOM_STATE
    })

    # --------------------------------------------------------
    # 7.2 TÉRMINOS REPRESENTATIVOS
    # --------------------------------------------------------

    terms_results.extend(
        get_top_terms(
            model,
            feature_names,
            TOP_TERMS
        )
    )

    # --------------------------------------------------------
    # 7.3 ASIGNACIONES DE DOCUMENTOS
    # --------------------------------------------------------

    train_output = train_valid.copy()

    train_output["cluster_id"] = train_labels
    train_output["experiment_id"] = f"K04_K{k}"
    train_output["split"] = "train"

    assignment_results.append(
        train_output[
            [
                "document_id",
                "split",
                "category",
                "cluster_id",
                "experiment_id"
            ]
        ]
    )

    if len(dev_labels) > 0:

        dev_output = dev_valid.copy()

        dev_output["cluster_id"] = dev_labels
        dev_output["experiment_id"] = f"K04_K{k}"
        dev_output["split"] = "dev"

        assignment_results.append(
            dev_output[
                [
                    "document_id",
                    "split",
                    "category",
                    "cluster_id",
                    "experiment_id"
                ]
            ]
        )

    # --------------------------------------------------------
    # 7.4 DISTRIBUCIÓN DE CATEGORÍAS POR CLÚSTER
    # --------------------------------------------------------

    for split_name, labels, split_df in [
        ("train", train_labels, train_valid),
        ("dev", dev_labels, dev_valid)
    ]:

        if len(labels) == 0:
            continue

        temp = split_df[
            ["document_id", "category"]
        ].copy()

        temp["cluster_id"] = labels

        counts = (
            temp.groupby(
                ["cluster_id", "category"]
            )
            .size()
            .reset_index(name="documents")
        )

        counts["experiment_id"] = f"K04_K{k}"
        counts["split"] = split_name

        distribution_results.append(counts)

    # --------------------------------------------------------
    # 7.5 DOCUMENTOS REPRESENTATIVOS
    # --------------------------------------------------------

    distances = model.transform(X_train)

    for cluster_id in range(k):

        cluster_indices = np.where(
            train_labels == cluster_id
        )[0]

        ordered_indices = cluster_indices[
            np.argsort(
                distances[cluster_indices, cluster_id]
            )
        ]

        selected_indices = ordered_indices[
            :TOP_DOCUMENTS
        ]

        for rank, index in enumerate(
            selected_indices,
            start=1
        ):

            row = train_valid.iloc[index]

            representative_results.append({
                "experiment_id": f"K04_K{k}",
                "cluster_id": cluster_id,
                "rank": rank,
                "document_id": row["document_id"],
                "category": row["category"],
                "distance_to_centroid": float(
                    distances[index, cluster_id]
                ),
                "text": row["input_text"]
            })

    # --------------------------------------------------------
    # 7.6 REGISTRO DEL EXPERIMENTO
    # --------------------------------------------------------

    experiment_log_results.append({
        "experiment_id": f"K04_K{k}",
        "algorithm": "KMeans",
        "representation": "TF-IDF",
        "preprocessing": "normalized_text + Spanish stopwords",
        "ngram_range": "1,2",
        "k": k,
        "min_df": 2,
        "max_df": 0.85,
        "max_features": 3000,
        "n_init": 20,
        "max_iter": 300,
        "random_state": RANDOM_STATE,
        "silhouette_train": train_silhouette,
        "silhouette_dev": dev_silhouette,
        "inertia": float(model.inertia_),
        "selected": False,
        "notes": "Experimento exploratorio K04: evaluación de K=7 y K=8"
    })


# ============================================================
# 8. GUARDAR RESULTADOS
# ============================================================

print("\nGuardando resultados...")

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

pd.concat(
    assignment_results,
    ignore_index=True
).to_csv(
    OUTPUT_ASSIGNMENTS,
    sep="\t",
    index=False,
    encoding="utf-8"
)

if distribution_results:

    pd.concat(
        distribution_results,
        ignore_index=True
    ).to_csv(
        OUTPUT_DISTRIBUTION,
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

append_experiment_log(
    experiment_log_results
)


# ============================================================
# 9. RESUMEN FINAL
# ============================================================

print("\n" + "=" * 65)
print("RESUMEN K04")
print("=" * 65)

metrics_df = pd.DataFrame(metrics_results)

print(
    metrics_df[
        [
            "experiment_id",
            "k",
            "vocabulary_size",
            "inertia",
            "silhouette_train",
            "silhouette_dev",
            "empty_train",
            "empty_dev"
        ]
    ].to_string(index=False)
)

print("\nArchivos generados:")

for path in [
    OUTPUT_METRICS,
    OUTPUT_TERMS,
    OUTPUT_REPRESENTATIVES,
    OUTPUT_ASSIGNMENTS,
    OUTPUT_DISTRIBUTION,
    OUTPUT_EMPTY,
    EXPERIMENT_LOG
]:

    print(f"- {path.relative_to(BASE_DIR)}")

print("\nExperimento K04 terminado correctamente.")
