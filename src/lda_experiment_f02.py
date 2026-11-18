
import unicodedata
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "labeled_documents_split.tsv"
)

OUTPUT_DIR = BASE_DIR / "results" / "lda"
EXPERIMENT_DIR = BASE_DIR / "experiments"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_EXPERIMENTS = EXPERIMENT_DIR / "experiments.tsv"

OUTPUT_TOPIC_WORDS = (
    OUTPUT_DIR / "f02_topic_words.tsv"
)

OUTPUT_REPRESENTATIVE_DOCS = (
    OUTPUT_DIR / "f02_representative_documents.tsv"
)

OUTPUT_TOPIC_RESULTS = (
    OUTPUT_DIR / "f02_topic_results.tsv"
)

OUTPUT_EMPTY_DOCS = (
    OUTPUT_DIR / "f02_empty_documents.tsv"
)

EXPERIMENT_ID = "F02"

RANDOM_STATE = 0

N_TOP_WORDS = 10
N_REPRESENTATIVE_DOCS = 2


# ============================================================
# 2. CONFIGURACIÓN DEL VECTORIZADOR F02
# ============================================================

VECTORIZER_CONFIG = {
    "lowercase": True,
    "stop_words": None,
    "strip_accents": "unicode",
    "ngram_range": (1, 1),
    "min_df": 3,
    "max_df": 0.85,
    "max_features": 3000,
    "token_pattern": r"(?u)\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+\b"
}


# ============================================================
# 3. CONFIGURACIÓN LDA F02
# ============================================================

LDA_CONFIG = {
    "n_components": 4,
    "max_iter": 50,
    "learning_method": "batch",
    "doc_topic_prior": None,
    "topic_word_prior": None,
    "random_state": RANDOM_STATE
}


# ============================================================
# 4. STOPWORDS EN ESPAÑOL
# ============================================================

SPANISH_STOPWORDS = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del",
    "se", "las", "por", "un", "para", "con", "no", "una",
    "su", "al", "lo", "como", "más", "pero", "sus", "le",
    "ya", "o", "este", "sí", "porque", "esta", "entre",
    "cuando", "muy", "sin", "sobre", "también", "hasta",
    "hay", "donde", "quien", "desde", "todo", "nos",
    "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto",
    "mí", "antes", "algunos", "qué", "unos", "yo",
    "otro", "otras", "otra", "él", "tanto", "esa",
    "estos", "mucho", "quienes", "nada", "muchos",
    "cual", "poco", "ella", "estar", "estas", "algunas",
    "algo", "nosotros", "ser", "es", "son", "fue",
    "han", "ha", "era", "está", "están", "fueron",
    "he", "haber", "tiene", "tienen", "tenía", "puede",
    "pueden", "podría", "hace", "hacen", "hacer",
    "cada", "parte", "tras", "según", "además"
]

EXTRA_STOPWORDS = [
    "méxico", "mexico",
    "hoy", "ayer", "mañana",
    "lunes", "martes", "miércoles", "jueves",
    "viernes", "sábado", "domingo",
    "enero", "febrero", "marzo", "abril",
    "mayo", "junio", "julio", "agosto",
    "septiembre", "octubre", "noviembre", "diciembre",
    "dónde", "cómo", "ver", "mientras"
]


# ============================================================
# 5. NORMALIZACIÓN DE STOPWORDS
# ============================================================

def normalize_stopword(word):
    """
    Normaliza las stopwords de la misma manera
    que CountVectorizer cuando strip_accents='unicode'.
    """

    word = word.lower()

    word = unicodedata.normalize(
        "NFKD",
        word
    )

    word = "".join(
        char
        for char in word
        if not unicodedata.combining(char)
    )

    return word


SPANISH_STOPWORDS = sorted(
    set(
        normalize_stopword(word)
        for word in SPANISH_STOPWORDS + EXTRA_STOPWORDS
    )
)

VECTORIZER_CONFIG["stop_words"] = SPANISH_STOPWORDS


# ============================================================
# 6. FUNCIONES AUXILIARES
# ============================================================

def get_nonzero_rows(matrix):
    """Identifica documentos con términos válidos."""

    return (
        np.asarray(matrix.sum(axis=1)).ravel() > 0
    )


def calculate_perplexity(model, matrix):
    """Calcula perplejidad evitando matrices vacías."""

    if matrix.shape[0] == 0:
        return np.nan

    if matrix.sum() == 0:
        return np.nan

    return float(
        model.perplexity(matrix)
    )


def get_topic_words(model, feature_names):
    """Extrae las palabras principales de cada tópico."""

    records = []

    for topic_idx, topic in enumerate(model.components_):

        top_indices = topic.argsort()[
            :-N_TOP_WORDS - 1:-1
        ]

        for position, index in enumerate(
            top_indices,
            start=1
        ):

            records.append({
                "experiment_id": EXPERIMENT_ID,
                "topic_id": topic_idx + 1,
                "position": position,
                "word": feature_names[index],
                "weight": float(topic[index])
            })

    return pd.DataFrame(records)


def get_representative_documents(
    model,
    matrix,
    documents,
    split_name
):
    """
    Obtiene los documentos con mayor probabilidad
    de pertenecer a cada tópico.
    """

    if matrix.shape[0] == 0:
        return pd.DataFrame()

    distributions = model.transform(matrix)

    documents = documents.reset_index(drop=True)

    records = []

    for topic_idx in range(model.n_components):

        probabilities = distributions[:, topic_idx]

        top_indices = np.argsort(
            probabilities
        )[::-1][:N_REPRESENTATIVE_DOCS]

        for rank, doc_idx in enumerate(
            top_indices,
            start=1
        ):

            records.append({
                "experiment_id": EXPERIMENT_ID,
                "topic_id": topic_idx + 1,
                "rank": rank,
                "document_id": documents.iloc[
                    doc_idx
                ]["document_id"],
                "title": documents.iloc[
                    doc_idx
                ]["title"],
                "text": documents.iloc[
                    doc_idx
                ]["text"],
                "topic_probability": float(
                    probabilities[doc_idx]
                ),
                "split": split_name
            })

    return pd.DataFrame(records)


def get_document_topic_records(
    model,
    matrix,
    documents,
    split_name
):
    """
    Obtiene la distribución temática de cada documento.
    """

    if matrix.shape[0] == 0:
        return pd.DataFrame()

    distributions = model.transform(matrix)

    documents = documents.reset_index(drop=True)

    records = []

    for index, distribution in enumerate(distributions):

        record = {
            "experiment_id": EXPERIMENT_ID,
            "document_id": documents.iloc[
                index
            ]["document_id"],
            "title": documents.iloc[
                index
            ]["title"],
            "split": split_name,
            "dominant_topic": int(
                np.argmax(distribution) + 1
            )
        }

        for topic_idx, probability in enumerate(
            distribution
        ):

            record[
                f"topic_{topic_idx + 1}_probability"
            ] = float(probability)

        records.append(record)

    return pd.DataFrame(records)


def update_experiment_log(record):
    """
    Agrega F02 a experiments.tsv.
    No modifica ni elimina experimentos anteriores.
    """

    if OUTPUT_EXPERIMENTS.exists():

        experiments = pd.read_csv(
            OUTPUT_EXPERIMENTS,
            sep="\t"
        )

        if "experiment_id" not in experiments.columns:
            raise ValueError(
                "experiments.tsv no contiene experiment_id."
            )

        existing = experiments[
            experiments["experiment_id"] == EXPERIMENT_ID
        ].index

        if len(existing) > 0:

            idx = existing[0]

            for key, value in record.items():
                experiments.loc[idx, key] = value

        else:

            experiments = pd.concat(
                [
                    experiments,
                    pd.DataFrame([record])
                ],
                ignore_index=True
            )

    else:

        experiments = pd.DataFrame([record])

    experiments.to_csv(
        OUTPUT_EXPERIMENTS,
        sep="\t",
        index=False,
        encoding="utf-8"
    )


# ============================================================
# 7. CARGAR CORPUS OFICIAL
# ============================================================

print("\n1. Cargando corpus oficial...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    sep="\t"
)

required_columns = [
    "document_id",
    "text",
    "category",
    "split"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Faltan columnas requeridas: {missing_columns}"
    )

if df["document_id"].duplicated().any():
    raise ValueError(
        "El corpus contiene document_id duplicados."
    )

if df["category"].isna().any():
    raise ValueError(
        "Hay documentos sin categoría."
    )

if df["text"].isna().any():
    raise ValueError(
        "Hay documentos sin texto."
    )

if "title" not in df.columns:
    df["title"] = (
        df["text"].astype(str).str[:150]
    )

df["document_id"] = df["document_id"].astype(str)
df["text"] = df["text"].astype(str)
df["title"] = df["title"].fillna("").astype(str)

df["split"] = df["split"].astype(str).str.lower()

if set(df["split"].unique()) != {"train", "dev", "test"}:
    raise ValueError(
        "Las particiones deben ser train, dev y test."
    )

print("Documentos:", len(df))

print("\nDistribución de particiones:")
print(df["split"].value_counts())

print("\nDistribución de categorías:")
print(df["category"].value_counts())


# ============================================================
# 8. RECUPERAR PARTICIONES OFICIALES
# ============================================================

print("\n2. Recuperando particiones oficiales...")

train_df = df[
    df["split"] == "train"
].copy().reset_index(drop=True)

dev_df = df[
    df["split"] == "dev"
].copy().reset_index(drop=True)

test_df = df[
    df["split"] == "test"
].copy().reset_index(drop=True)

print("Train:", len(train_df))
print("Dev:", len(dev_df))
print("Test:", len(test_df))


# ============================================================
# 9. VECTORIZACIÓN
# ============================================================

print("\n3. Aplicando CountVectorizer F02...")

vectorizer = CountVectorizer(
    **VECTORIZER_CONFIG
)

# El vocabulario se aprende exclusivamente con Train.
X_train = vectorizer.fit_transform(
    train_df["text"]
)

# Dev y Test solo se transforman.
X_dev = vectorizer.transform(
    dev_df["text"]
)

X_test = vectorizer.transform(
    test_df["text"]
)

feature_names = vectorizer.get_feature_names_out()

print("Tamaño del vocabulario:", len(feature_names))

if len(feature_names) == 0:
    raise ValueError(
        "El vocabulario está vacío. Revisa las stopwords y min_df."
    )


# ============================================================
# 10. IDENTIFICAR DOCUMENTOS SIN TÉRMINOS
# ============================================================

print("\n4. Identificando documentos sin términos...")

train_mask = get_nonzero_rows(X_train)
dev_mask = get_nonzero_rows(X_dev)
test_mask = get_nonzero_rows(X_test)

# Crear una tabla de documentos sin términos.
empty_train = train_df.loc[~train_mask].copy()
empty_dev = dev_df.loc[~dev_mask].copy()
empty_test = test_df.loc[~test_mask].copy()

empty_train["split"] = "train"
empty_dev["split"] = "dev"
empty_test["split"] = "test"

empty_documents = pd.concat(
    [
        empty_train,
        empty_dev,
        empty_test
    ],
    ignore_index=True
)

if not empty_documents.empty:

    empty_documents[
        [
            "document_id",
            "title",
            "category",
            "split",
            "text"
        ]
    ].to_csv(
        OUTPUT_EMPTY_DOCS,
        sep="\t",
        index=False,
        encoding="utf-8"
    )

    print(
        "Documentos sin términos guardados:",
        OUTPUT_EMPTY_DOCS
    )

else:

    print("No hay documentos vacíos.")

# Filtrar matrices y documentos válidos.
X_train_valid = X_train[train_mask]
X_dev_valid = X_dev[dev_mask]
X_test_valid = X_test[test_mask]

train_valid_df = train_df.loc[
    train_mask
].reset_index(drop=True)

dev_valid_df = dev_df.loc[
    dev_mask
].reset_index(drop=True)

test_valid_df = test_df.loc[
    test_mask
].reset_index(drop=True)

print(
    "Train válidos:",
    len(train_valid_df),
    "/",
    len(train_df)
)

print(
    "Dev válidos:",
    len(dev_valid_df),
    "/",
    len(dev_df)
)

print(
    "Test válidos:",
    len(test_valid_df),
    "/",
    len(test_df)
)

if X_train_valid.shape[0] == 0:
    raise ValueError(
        "No hay documentos válidos para entrenar LDA."
    )


# ============================================================
# 11. ENTRENAMIENTO LDA
# ============================================================

print("\n5. Entrenando modelo LDA F02...")

lda_model = LatentDirichletAllocation(
    **LDA_CONFIG
)

lda_model.fit(
    X_train_valid
)

print("Modelo entrenado correctamente.")


# ============================================================
# 12. PERPLEJIDAD
# ============================================================

print("\n6. Calculando perplejidad...")

train_perplexity = calculate_perplexity(
    lda_model,
    X_train_valid
)

dev_perplexity = calculate_perplexity(
    lda_model,
    X_dev_valid
)

test_perplexity = calculate_perplexity(
    lda_model,
    X_test_valid
)

print("Perplejidad Train:", train_perplexity)
print("Perplejidad Dev:", dev_perplexity)
print("Perplejidad Test:", test_perplexity)


# ============================================================
# 13. PALABRAS PRINCIPALES POR TÓPICO
# ============================================================

print("\n7. Extrayendo palabras principales...")

topic_words_df = get_topic_words(
    lda_model,
    feature_names
)

topic_words_df.to_csv(
    OUTPUT_TOPIC_WORDS,
    sep="\t",
    index=False,
    encoding="utf-8"
)

print("Archivo generado:", OUTPUT_TOPIC_WORDS)

for topic_id in range(
    1,
    LDA_CONFIG["n_components"] + 1
):

    words = topic_words_df[
        topic_words_df["topic_id"] == topic_id
    ]["word"].tolist()

    print(f"\nTópico {topic_id}:")
    print(", ".join(words))


# ============================================================
# 14. DOCUMENTOS REPRESENTATIVOS
# ============================================================

print("\n8. Obteniendo documentos representativos...")

representative_docs = get_representative_documents(
    lda_model,
    X_train_valid,
    train_valid_df,
    "train"
)

representative_docs.to_csv(
    OUTPUT_REPRESENTATIVE_DOCS,
    sep="\t",
    index=False,
    encoding="utf-8"
)

print(
    "Archivo generado:",
    OUTPUT_REPRESENTATIVE_DOCS
)

for topic_id in range(
    1,
    LDA_CONFIG["n_components"] + 1
):

    topic_docs = representative_docs[
        representative_docs["topic_id"] == topic_id
    ]

    print(f"\nTópico {topic_id}:")

    for _, row in topic_docs.iterrows():

        print(
            f"Documento {row['document_id']}: "
            f"{row['title']} "
            f"(probabilidad: "
            f"{row['topic_probability']:.4f})"
        )


# ============================================================
# 15. DISTRIBUCIONES TEMÁTICAS
# ============================================================

print("\n9. Calculando distribuciones temáticas...")

train_results = get_document_topic_records(
    lda_model,
    X_train_valid,
    train_valid_df,
    "train"
)

dev_results = get_document_topic_records(
    lda_model,
    X_dev_valid,
    dev_valid_df,
    "dev"
)

test_results = get_document_topic_records(
    lda_model,
    X_test_valid,
    test_valid_df,
    "test"
)

topic_results = pd.concat(
    [
        train_results,
        dev_results,
        test_results
    ],
    ignore_index=True
)

topic_results.to_csv(
    OUTPUT_TOPIC_RESULTS,
    sep="\t",
    index=False,
    encoding="utf-8"
)

print("Archivo generado:", OUTPUT_TOPIC_RESULTS)

print(
    "Documentos con distribución temática:",
    len(topic_results)
)


# ============================================================
# 16. REGISTRAR EXPERIMENTO F02
# ============================================================

print("\n10. Registrando experimento F02...")

experiment_record = {
    "experiment_id": EXPERIMENT_ID,
    "task": "LDA",

    "preprocessing": (
        "lowercase=True; "
        "stopwords=español+adicionales; "
        "stopwords_normalizadas=True; "
        "strip_accents=unicode; "
        "deduplicacion_previa=True; "
        "particiones_oficiales=True"
    ),

    "representation": "CountVectorizer",

    "ngram_range": "(1,1)",
    "min_df": 3,
    "max_df": 0.85,
    "max_features": 3000,

    "algorithm": "LatentDirichletAllocation",

    "hyperparameters": (
        "n_components=4; "
        "max_iter=50; "
        "learning_method=batch; "
        "doc_topic_prior=None; "
        "topic_word_prior=None; "
        "random_state=0"
    ),

    "train_perplexity": train_perplexity,
    "dev_metric": dev_perplexity,
    "dev_metric_name": "perplexity",
    "test_perplexity": test_perplexity,

    "vocabulary_size": len(feature_names),

    "train_documents": len(train_df),
    "dev_documents": len(dev_df),
    "test_documents": len(test_df),

    "empty_documents": len(empty_documents),

    "selected": False,
    "status": "success"
}

update_experiment_log(
    experiment_record
)

print(
    "Archivo actualizado:",
    OUTPUT_EXPERIMENTS
)


# ============================================================
# 17. RESUMEN FINAL
# ============================================================

print("\n" + "=" * 65)
print("RESUMEN FINAL - EXPERIMENTO LDA F02")
print("=" * 65)

print("\nCorpus:")
print("Documentos:", len(df))
print("Train:", len(train_df))
print("Dev:", len(dev_df))
print("Test:", len(test_df))

print("\nConfiguración:")
print("Tópicos:", LDA_CONFIG["n_components"])
print("Iteraciones:", LDA_CONFIG["max_iter"])
print("Método:", LDA_CONFIG["learning_method"])
print("Representación: CountVectorizer")
print("N-gramas:", VECTORIZER_CONFIG["ngram_range"])
print("Min DF:", VECTORIZER_CONFIG["min_df"])
print("Max DF:", VECTORIZER_CONFIG["max_df"])
print("Máximo de términos:", VECTORIZER_CONFIG["max_features"])
print("Stopwords normalizadas: Sí")

print("\nVocabulario:")
print(len(feature_names))

print("\nPerplejidad:")
print("Train:", train_perplexity)
print("Dev:", dev_perplexity)
print("Test:", test_perplexity)

print("\nDocumentos sin términos válidos:")
print(len(empty_documents))

print("\nArchivos generados:")
print("-", OUTPUT_TOPIC_WORDS)
print("-", OUTPUT_REPRESENTATIVE_DOCS)
print("-", OUTPUT_TOPIC_RESULTS)
print("-", OUTPUT_EMPTY_DOCS)
print("-", OUTPUT_EXPERIMENTS)

print("\nExperimento registrado:", EXPERIMENT_ID)
print("Seleccionado como modelo final: No")

print("\nProceso terminado correctamente.")
print("=" * 65)
