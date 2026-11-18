
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed" /
    "labeled_documents_split.tsv"
)

OUTPUT_DIR = ROOT / "results" / "lda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS_FILE = (
    ROOT / "experiments" / "experiments.tsv"
)

EXPERIMENT_ID = "F05"

OUTPUT_WORDS = OUTPUT_DIR / "f05_lda_topic_words.tsv"
OUTPUT_REPRESENTATIVES = (
    OUTPUT_DIR / "f05_lda_representative_documents.tsv"
)
OUTPUT_RESULTS = OUTPUT_DIR / "f05_topic_results.tsv"
OUTPUT_EMPTY = OUTPUT_DIR / "f05_lda_empty_documents.tsv"


# ============================================================
# 2. PARÁMETROS
# ============================================================

N_TOPICS = 5
MAX_ITER = 50

LEARNING_METHOD = "online"
RANDOM_STATE = 0

N_TOP_WORDS = 10
N_REPRESENTATIVE_DOCS = 2

MIN_DF = 3
MAX_DF = 0.85
MAX_FEATURES = 3000

NGRAM_RANGE = (1, 1)


# ============================================================
# 3. STOPWORDS DE F02
# ============================================================

EXTRA_STOPWORDS = {
    "méxico",
    "mexico",
    "2026",
    "2025",
    "2024",
    "hoy",
    "ayer",
    "mañana",
    "manana",

    "lunes",
    "martes",
    "miércoles",
    "miercoles",
    "jueves",
    "viernes",
    "sábado",
    "sabado",
    "domingo",

    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",

    "dónde",
    "donde",
    "cómo",
    "como",
    "ver",
    "mientras"
}


def normalize_word(word):
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


# ============================================================
# 4. STOPWORDS ESPAÑOLAS
# ============================================================

SPANISH_STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los",
    "del", "se", "las", "por", "un", "para", "con",
    "no", "una", "su", "al", "lo", "como", "más",
    "mas", "pero", "sus", "le", "ya", "o", "este",
    "sí", "si", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "también", "tambien",
    "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos",
    "uno", "les", "ni", "contra", "otros", "ese",
    "eso", "ante", "ellos", "e", "esto", "mí",
    "mi", "antes", "algunos", "qué", "que",
    "unos", "yo", "otro", "otras", "otra",
    "él", "ella", "tanto", "esa", "estos",
    "mucho", "quienes", "nada", "muchos",
    "cual", "poco", "estar", "ser", "es", "son",
    "fue", "han", "ha", "era", "será", "serán",
    "puede", "pueden", "hace", "hacer", "he",
    "sido", "tiene", "tienen", "tener", "está",
    "estan", "están", "estaba", "estaban"
}


ALL_STOPWORDS = (
    SPANISH_STOPWORDS.union(EXTRA_STOPWORDS)
)

NORMALIZED_STOPWORDS = {
    normalize_word(word)
    for word in ALL_STOPWORDS
}


# ============================================================
# 5. CARGA DEL CORPUS OFICIAL
# ============================================================

print("=" * 65)
print("EXPERIMENTO LDA F05")
print("=" * 65)

print("\nCargando corpus oficial...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    dtype=str
).fillna("")

required_columns = {
    "document_id",
    "text",
    "category",
    "split"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Faltan columnas: {missing_columns}"
    )

if df["document_id"].duplicated().any():
    raise ValueError(
        "Hay document_id duplicados."
    )

train_df = df[df["split"] == "train"].copy()
dev_df = df[df["split"] == "dev"].copy()
test_df = df[df["split"] == "test"].copy()

print(f"Total: {len(df)}")
print(f"Train: {len(train_df)}")
print(f"Dev:   {len(dev_df)}")
print(f"Test:  {len(test_df)}")


# ============================================================
# 6. PREPARACIÓN DEL TEXTO
# ============================================================

X_train_text = train_df["text"].astype(str).tolist()
X_dev_text = dev_df["text"].astype(str).tolist()
X_test_text = test_df["text"].astype(str).tolist()


# ============================================================
# 7. VECTORIZACIÓN
# ============================================================

print("\nConfigurando CountVectorizer...")

vectorizer = CountVectorizer(
    lowercase=True,
    stop_words=sorted(NORMALIZED_STOPWORDS),
    strip_accents="unicode",
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_df=MAX_DF,
    max_features=MAX_FEATURES,
    token_pattern=(
        r"(?u)\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+\b"
    )
)

print("Ajustando vocabulario exclusivamente con TRAIN...")

X_train = vectorizer.fit_transform(X_train_text)

X_dev = vectorizer.transform(X_dev_text)

X_test = vectorizer.transform(X_test_text)

feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulario: {len(feature_names)}")


# ============================================================
# 8. DETECCIÓN DE DOCUMENTOS VACÍOS
# ============================================================

def valid_rows(matrix):
    return np.asarray(
        matrix.sum(axis=1)
    ).ravel() > 0


train_valid = valid_rows(X_train)
dev_valid = valid_rows(X_dev)
test_valid = valid_rows(X_test)

X_train_valid = X_train[train_valid]
X_dev_valid = X_dev[dev_valid]
X_test_valid = X_test[test_valid]

print("\nDocumentos válidos:")

print(
    f"Train: {train_valid.sum()} / {len(train_valid)}"
)
print(
    f"Dev:   {dev_valid.sum()} / {len(dev_valid)}"
)
print(
    f"Test:  {test_valid.sum()} / {len(test_valid)}"
)


# ============================================================
# 9. GUARDAR DOCUMENTOS VACÍOS
# ============================================================

empty_records = []

for dataframe, valid_mask, split_name in [
    (train_df, train_valid, "train"),
    (dev_df, dev_valid, "dev"),
    (test_df, test_valid, "test")
]:

    empty_indices = np.where(~valid_mask)[0]

    for index in empty_indices:

        row = dataframe.iloc[index]

        empty_records.append({
            "experiment_id": EXPERIMENT_ID,
            "document_id": row["document_id"],
            "split": split_name,
            "category": row["category"],
            "title": row.get("title", ""),
            "text": row["text"]
        })


empty_df = pd.DataFrame(empty_records)

empty_df.to_csv(
    OUTPUT_EMPTY,
    sep="\t",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 10. ENTRENAMIENTO LDA
# ============================================================

if X_train_valid.shape[0] == 0:
    raise ValueError(
        "No existen documentos válidos para entrenar."
    )

print("\nEntrenando LDA...")

lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    max_iter=MAX_ITER,
    learning_method=LEARNING_METHOD,
    random_state=RANDOM_STATE,
    evaluate_every=-1,
    n_jobs=-1
)

lda.fit(X_train_valid)

print("Modelo entrenado correctamente.")


# ============================================================
# 11. PERPLEJIDAD
# ============================================================

def calculate_perplexity(model, matrix):

    if matrix.shape[0] == 0:
        return np.nan

    return model.perplexity(matrix)


train_perplexity = calculate_perplexity(
    lda,
    X_train_valid
)

dev_perplexity = calculate_perplexity(
    lda,
    X_dev_valid
)

test_perplexity = calculate_perplexity(
    lda,
    X_test_valid
)

print("\nPerplejidad:")

print(f"Train: {train_perplexity:.4f}")
print(f"Dev:   {dev_perplexity:.4f}")
print(f"Test:  {test_perplexity:.4f}")


# ============================================================
# 12. PALABRAS PRINCIPALES POR TÓPICO
# ============================================================

topic_word_records = []

print("\nPalabras principales por tópico:")

for topic_index, topic in enumerate(lda.components_):

    top_indices = topic.argsort()[
        -N_TOP_WORDS:
    ][::-1]

    top_words = [
        feature_names[index]
        for index in top_indices
    ]

    print(f"\nTópico {topic_index + 1}:")
    print(", ".join(top_words))

    for rank, word in enumerate(top_words, start=1):

        topic_word_records.append({
            "experiment_id": EXPERIMENT_ID,
            "topic_id": topic_index + 1,
            "rank": rank,
            "word": word
        })


topic_words_df = pd.DataFrame(
    topic_word_records
)

topic_words_df.to_csv(
    OUTPUT_WORDS,
    sep="\t",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 13. DISTRIBUCIONES TEMÁTICAS
# ============================================================

result_records = []


def save_topic_distributions(
    dataframe,
    matrix,
    valid_mask,
    split_name
):

    if matrix.shape[0] == 0:
        return

    distributions = lda.transform(matrix)

    valid_indices = np.where(valid_mask)[0]

    for local_index, original_index in enumerate(valid_indices):

        row = dataframe.iloc[original_index]

        distribution = distributions[local_index]

        dominant_topic = int(
            np.argmax(distribution) + 1
        )

        record = {
            "experiment_id": EXPERIMENT_ID,
            "document_id": row["document_id"],
            "split": split_name,
            "category": row["category"],
            "dominant_topic": dominant_topic,
            "topic_probability": float(
                distribution[dominant_topic - 1]
            )
        }

        for topic_index, probability in enumerate(
            distribution,
            start=1
        ):

            record[f"topic_{topic_index}"] = float(
                probability
            )

        result_records.append(record)


save_topic_distributions(
    train_df,
    X_train_valid,
    train_valid,
    "train"
)

save_topic_distributions(
    dev_df,
    X_dev_valid,
    dev_valid,
    "dev"
)

save_topic_distributions(
    test_df,
    X_test_valid,
    test_valid,
    "test"
)


results_df = pd.DataFrame(result_records)

results_df.to_csv(
    OUTPUT_RESULTS,
    sep="\t",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 14. DOCUMENTOS REPRESENTATIVOS
# ============================================================

print("\nDocumentos representativos por tópico:")

train_distributions = lda.transform(
    X_train_valid
)

valid_train_df = train_df.iloc[
    np.where(train_valid)[0]
].copy()

representative_records = []

for topic_index in range(N_TOPICS):

    probabilities = train_distributions[:, topic_index]

    best_indices = np.argsort(
        probabilities
    )[::-1][:N_REPRESENTATIVE_DOCS]

    print(f"\nTópico {topic_index + 1}:")

    for rank, index in enumerate(best_indices, start=1):

        row = valid_train_df.iloc[index]

        probability = probabilities[index]

        title = row.get("title", "")

        if not title:
            title = row["text"][:150]

        print(
            f"  {rank}. {title} "
            f"(probabilidad: {probability:.4f})"
        )

        representative_records.append({
            "experiment_id": EXPERIMENT_ID,
            "topic_id": topic_index + 1,
            "rank": rank,
            "document_id": row["document_id"],
            "category": row["category"],
            "title": title,
            "topic_probability": float(probability),
            "text": row["text"]
        })


representatives_df = pd.DataFrame(
    representative_records
)

representatives_df.to_csv(
    OUTPUT_REPRESENTATIVES,
    sep="\t",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 15. RESUMEN DEL EXPERIMENTO
# ============================================================

print("\n" + "=" * 65)
print("RESUMEN F05")
print("=" * 65)

print(f"Tópicos: {N_TOPICS}")
print(f"Iteraciones: {MAX_ITER}")
print(f"Learning method: {LEARNING_METHOD}")
print(f"N-gramas: {NGRAM_RANGE}")
print(f"min_df: {MIN_DF}")
print(f"max_df: {MAX_DF}")
print(f"Vocabulario: {len(feature_names)}")

print(f"\nPerplejidad Train: {train_perplexity:.4f}")
print(f"Perplejidad Dev:   {dev_perplexity:.4f}")
print(f"Perplejidad Test:  {test_perplexity:.4f}")

print(f"\nDocumentos vacíos: {len(empty_df)}")


# ============================================================
# 16. REGISTRO DEL EXPERIMENTO
# ============================================================

EXPERIMENTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

experiment_record = {
    "experiment_id": EXPERIMENT_ID,
    "task": "LDA",
    "n_components": N_TOPICS,
    "max_iter": MAX_ITER,
    "learning_method": LEARNING_METHOD,
    "random_state": RANDOM_STATE,
    "ngram_range": str(NGRAM_RANGE),
    "min_df": MIN_DF,
    "max_df": MAX_DF,
    "max_features": MAX_FEATURES,
    "vocabulary_size": len(feature_names),
    "train_documents": len(train_df),
    "dev_documents": len(dev_df),
    "test_documents": len(test_df),
    "train_valid_documents": int(train_valid.sum()),
    "dev_valid_documents": int(dev_valid.sum()),
    "test_valid_documents": int(test_valid.sum()),
    "empty_documents": len(empty_df),
    "train_perplexity": train_perplexity,
    "dev_perplexity": dev_perplexity,
    "test_perplexity": test_perplexity,
    "preprocessing": (
        "F02 preprocessing; normalized Spanish stopwords; "
        "extra stopwords; lowercase; strip_accents=unicode; "
        "unigrams"
    ),
    "selected": False
}

if EXPERIMENTS_FILE.exists():

    experiments_df = pd.read_csv(
        EXPERIMENTS_FILE,
        sep="\t"
    )

    experiments_df = experiments_df[
        experiments_df["experiment_id"].astype(str)
        != EXPERIMENT_ID
    ]

    experiments_df = pd.concat(
        [
            experiments_df,
            pd.DataFrame([experiment_record])
        ],
        ignore_index=True
    )

else:

    experiments_df = pd.DataFrame(
        [experiment_record]
    )


experiments_df.to_csv(
    EXPERIMENTS_FILE,
    sep="\t",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 17. FINALIZACIÓN
# ============================================================

print("\nArchivos generados:")

print(OUTPUT_WORDS)
print(OUTPUT_REPRESENTATIVES)
print(OUTPUT_RESULTS)
print(OUTPUT_EMPTY)
print(EXPERIMENTS_FILE)

print("\nExperimento F05 finalizado correctamente.")
