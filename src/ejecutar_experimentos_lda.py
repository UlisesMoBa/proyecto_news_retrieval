
import os
import json
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR, "data", "processed", "labeled_documents.tsv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "experiments")

RANDOM_STATE = 0
N_TOP_WORDS = 10
N_REPRESENTATIVE_DOCS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. STOPWORDS EN ESPAÑOL
# ============================================================

SPANISH_STOPWORDS = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del",
    "se", "las", "por", "un", "para", "con", "no", "una",
    "su", "al", "lo", "como", "más", "pero", "sus", "le",
    "ya", "o", "este", "sí", "porque", "esta", "entre",
    "cuando", "muy", "sin", "sobre", "también", "me",
    "hasta", "hay", "donde", "quien", "desde", "todo",
    "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto",
    "mí", "antes", "algunos", "qué", "unos", "yo", "otro",
    "otras", "otra", "él", "tanto", "esa", "estos", "mucho",
    "quienes", "nada", "muchos", "cual", "poco", "ella",
    "estar", "estas", "algunas", "algo", "nosotros",
    "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas",
    "nosotras", "vosotros", "vosotras", "os", "mío",
    "mía", "míos", "mías", "tuyo", "tuya", "tuyos",
    "tuyas", "suyo", "suya", "suyos", "suyas",
    "nuestro", "nuestra", "nuestros", "nuestras",
    "vuestro", "vuestra", "vuestros", "vuestras",
    "esos", "esas", "estoy", "estás", "está", "estamos",
    "están", "esté", "estés", "estemos", "estén",
    "ser", "soy", "eres", "somos", "son", "era",
    "eras", "éramos", "eran", "fui", "fue", "fueron",
    "sea", "sean", "siendo", "sido", "tiene", "tienen",
    "tener", "tenía", "tenían", "hacer", "hace", "hacen",
    "he", "ha", "han", "había", "habían", "haber",
    "puede", "pueden", "podría", "podrían",
    "dijo", "dicen", "según", "cada", "tras"
]


# Palabras adicionales utilizadas en U14 y U15.
EXTRA_STOPWORDS = [
    "méxico", "mexico",
    "2026", "2025", "2024",
    "hoy", "ayer", "mañana",
    "miércoles", "jueves", "viernes",
    "sábado", "domingo", "lunes", "martes",
    "enero", "febrero", "marzo", "abril",
    "mayo", "junio", "julio", "agosto",
    "septiembre", "octubre", "noviembre",
    "diciembre",
    "dónde", "cómo", "ver", "mientras"
]


# ============================================================
# 3. CONFIGURACIONES DE LOS EXPERIMENTOS
# ============================================================

EXPERIMENTS = {

    "U01": {
        "description": "Modelo base sin stopwords",
        "stop_words": None,
        "strip_accents": None,
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 3000,
        "token_pattern": r"(?u)\b\w\w+\b",
        "n_components": 4,
        "max_iter": 15,
        "learning_method": "batch",
        "doc_topic_prior": None,
        "topic_word_prior": None
    },

    "U02": {
        "description": "Stopwords estándar en español",
        "stop_words": SPANISH_STOPWORDS,
        "strip_accents": None,
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 3000,
        "token_pattern": r"(?u)\b\w\w+\b",
        "n_components": 4,
        "max_iter": 15,
        "learning_method": "batch",
        "doc_topic_prior": None,
        "topic_word_prior": None
    },

    "U14": {
        "description": "Stopwords estándar y específicas del corpus",
        "stop_words": list(set(SPANISH_STOPWORDS + EXTRA_STOPWORDS)),
        "strip_accents": None,
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 3000,
        "token_pattern": r"(?u)\b\w\w+\b",
        "n_components": 4,
        "max_iter": 15,
        "learning_method": "batch",
        "doc_topic_prior": None,
        "topic_word_prior": None
    },

    "U15": {
        "description": "Stopwords específicas, bigramas y normalización de acentos",
        "stop_words": list(set(SPANISH_STOPWORDS + EXTRA_STOPWORDS)),
        "strip_accents": "unicode",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 3000,
        "token_pattern": r"(?u)\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+\b",
        "n_components": 4,
        "max_iter": 30,
        "learning_method": "batch",
        "doc_topic_prior": None,
        "topic_word_prior": None
    }
}


# ============================================================
# 4. FUNCIONES AUXILIARES
# ============================================================

def obtener_documentos_representativos(
    model, matrix, documents, topic_words
):
    """
    Obtiene los documentos con mayor probabilidad
    de pertenecer a cada tópico.
    """

    distributions = model.transform(matrix)
    results = []

    for topic_index in range(model.n_components):

        sorted_indices = np.argsort(
            distributions[:, topic_index]
        )[::-1]

        selected_indices = sorted_indices[
            :N_REPRESENTATIVE_DOCS
        ]

        for rank, index in enumerate(selected_indices, start=1):

            results.append({
                "topic": topic_index + 1,
                "rank": rank,
                "document_id": documents.iloc[index]["document_id"],
                "title": documents.iloc[index]["title"],
                "category": documents.iloc[index]["category"],
                "probability": distributions[index, topic_index],
                "text": documents.iloc[index]["text"]
            })

    return pd.DataFrame(results), distributions


def obtener_palabras_topicas(model, vectorizer):

    feature_names = vectorizer.get_feature_names_out()
    results = []

    for topic_index, topic in enumerate(model.components_):

        top_indices = topic.argsort()[:-N_TOP_WORDS - 1:-1]

        for rank, word_index in enumerate(top_indices, start=1):

            results.append({
                "topic": topic_index + 1,
                "rank": rank,
                "word": feature_names[word_index],
                "weight": topic[word_index]
            })

    return pd.DataFrame(results)


def guardar_distribuciones(
    model, matrix, documents, split_name, output_file
):

    distributions = model.transform(matrix)
    results = []

    for index, row in documents.reset_index(drop=True).iterrows():

        result = {
            "document_id": row["document_id"],
            "split": split_name,
            "category": row["category"]
        }

        for topic_index in range(model.n_components):
            result[f"topic_{topic_index + 1}"] = (
                distributions[index, topic_index]
            )

        results.append(result)

    pd.DataFrame(results).to_csv(
        output_file,
        sep="\t",
        index=False,
        encoding="utf-8"
    )


def guardar_configuracion(config, output_file):

    config_to_save = {}

    for key, value in config.items():

        if key == "stop_words":
            if value is None:
                config_to_save[key] = None
            else:
                config_to_save[key] = sorted(list(value))

        elif isinstance(value, tuple):
            config_to_save[key] = list(value)

        else:
            config_to_save[key] = value

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            config_to_save,
            file,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# 5. CARGAR CORPUS Y VALIDAR PARTICIÓN
# ============================================================

print("\nCargando corpus...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"No se encontró el archivo: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE, sep="\t")

required_columns = [
    "document_id",
    "text",
    "split"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Faltan columnas requeridas: {missing_columns}"
    )

# El título es opcional.
if "title" not in df.columns:
    df["title"] = df["text"].astype(str).str[:150]

# La categoría se conserva para identificar los documentos,
# pero NO se utiliza para entrenar LDA.
if "category" not in df.columns:
    df["category"] = ""

df["split"] = df["split"].astype(str).str.lower().str.strip()

# Eliminar documentos sin texto.
df["text"] = df["text"].fillna("").astype(str)
df = df[df["text"].str.strip() != ""].copy()

train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
dev_df = df[df["split"] == "dev"].copy().reset_index(drop=True)
test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

if train_df.empty or dev_df.empty or test_df.empty:
    raise ValueError(
        "La columna split debe contener documentos en train, dev y test. "
        "No se realizará una nueva división automáticamente."
    )

print(f"Train: {len(train_df)} documentos")
print(f"Dev:   {len(dev_df)} documentos")
print(f"Test:  {len(test_df)} documentos")

print("\nSe conservará la partición original.")


# ============================================================
# 6. EJECUCIÓN DE EXPERIMENTOS
# ============================================================

experiment_results = []

for experiment_id, config in EXPERIMENTS.items():

    print("\n" + "=" * 65)
    print(f" EJECUTANDO {experiment_id}")
    print("=" * 65)

    experiment_dir = os.path.join(
        OUTPUT_DIR, experiment_id
    )

    os.makedirs(experiment_dir, exist_ok=True)

    # --------------------------------------------------------
    # 6.1. Crear vectorizador
    # --------------------------------------------------------

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=config["stop_words"],
        strip_accents=config["strip_accents"],
        ngram_range=config["ngram_range"],
        min_df=config["min_df"],
        max_df=config["max_df"],
        max_features=config["max_features"],
        token_pattern=config["token_pattern"]
    )

    # Ajustar SOLO con Train.
    X_train = vectorizer.fit_transform(train_df["text"])

    # Transformar Dev y Test sin volver a ajustar.
    X_dev = vectorizer.transform(dev_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    print("Tamaño del vocabulario:", len(vectorizer.vocabulary_))

    # --------------------------------------------------------
    # 6.2. Eliminar documentos sin palabras reconocidas
    # --------------------------------------------------------

    train_mask = np.asarray(X_train.sum(axis=1)).ravel() > 0
    dev_mask = np.asarray(X_dev.sum(axis=1)).ravel() > 0
    test_mask = np.asarray(X_test.sum(axis=1)).ravel() > 0

    X_train_valid = X_train[train_mask]
    X_dev_valid = X_dev[dev_mask]
    X_test_valid = X_test[test_mask]

    train_valid = train_df.loc[train_mask].reset_index(drop=True)
    dev_valid = dev_df.loc[dev_mask].reset_index(drop=True)
    test_valid = test_df.loc[test_mask].reset_index(drop=True)

    if X_train_valid.shape[0] == 0:
        print("No hay documentos válidos en Train. Se omite.")
        continue

    if X_dev_valid.shape[0] == 0:
        print("No hay documentos válidos en Dev. Se omite.")
        continue

    print("Documentos válidos Train:", X_train_valid.shape[0])
    print("Documentos válidos Dev:", X_dev_valid.shape[0])
    print("Documentos válidos Test:", X_test_valid.shape[0])

    # --------------------------------------------------------
    # 6.3. Entrenar LDA únicamente con Train
    # --------------------------------------------------------

    lda = LatentDirichletAllocation(
        n_components=config["n_components"],
        max_iter=config["max_iter"],
        learning_method=config["learning_method"],
        doc_topic_prior=config["doc_topic_prior"],
        topic_word_prior=config["topic_word_prior"],
        random_state=RANDOM_STATE
    )

    lda.fit(X_train_valid)

    # --------------------------------------------------------
    # 6.4. Evaluación de Train y Dev
    # --------------------------------------------------------

    train_perplexity = lda.perplexity(X_train_valid)
    dev_perplexity = lda.perplexity(X_dev_valid)

    print(f"Perplexity Train: {train_perplexity:.4f}")
    print(f"Perplexity Dev:   {dev_perplexity:.4f}")

    # --------------------------------------------------------
    # 6.5. Obtener palabras principales
    # --------------------------------------------------------

    topic_words_df = obtener_palabras_topicas(
        lda, vectorizer
    )

    topic_words_df.to_csv(
        os.path.join(experiment_dir, "topic_words.tsv"),
        sep="\t",
        index=False,
        encoding="utf-8"
    )

    print("\nPalabras principales por tópico:")

    for topic_index in range(config["n_components"]):

        words = topic_words_df[
            topic_words_df["topic"] == topic_index + 1
        ]["word"].tolist()

        print(f"Tópico {topic_index + 1}: {', '.join(words)}")

    # --------------------------------------------------------
    # 6.6. Documentos representativos de Train
    # --------------------------------------------------------

    representative_df, train_distributions = (
        obtener_documentos_representativos(
            lda,
            X_train_valid,
            train_valid,
            topic_words_df
        )
    )

    representative_df.to_csv(
        os.path.join(
            experiment_dir,
            "representative_documents.tsv"
        ),
        sep="\t",
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 6.7. Distribuciones de tópicos
    # --------------------------------------------------------

    distributions_df = pd.DataFrame(
        train_distributions,
        columns=[
            f"topic_{i + 1}"
            for i in range(config["n_components"])
        ]
    )

    train_output = train_valid[
        ["document_id", "category"]
    ].reset_index(drop=True)

    train_output = pd.concat(
        [train_output, distributions_df],
        axis=1
    )

    train_output.insert(1, "split", "train")

    train_output.to_csv(
        os.path.join(
            experiment_dir,
            "train_topic_distributions.tsv"
        ),
        sep="\t",
        index=False,
        encoding="utf-8"
    )

    guardar_distribuciones(
        lda,
        X_dev_valid,
        dev_valid,
        "dev",
        os.path.join(
            experiment_dir,
            "dev_topic_distributions.tsv"
        )
    )

    # Test se transforma, pero no se evalúa ni se utiliza
    # para seleccionar el experimento.
    guardar_distribuciones(
        lda,
        X_test_valid,
        test_valid,
        "test",
        os.path.join(
            experiment_dir,
            "test_topic_distributions.tsv"
        )
    )

    # --------------------------------------------------------
    # 6.8. Guardar configuración exacta
    # --------------------------------------------------------

    guardar_configuracion(
        config,
        os.path.join(experiment_dir, "configuration.json")
    )

    # --------------------------------------------------------
    # 6.9. Guardar resultados generales
    # --------------------------------------------------------

    result = {
        "experiment_id": experiment_id,
        "description": config["description"],
        "vocabulary_size": len(vectorizer.vocabulary_),
        "n_topics": config["n_components"],
        "max_iter": config["max_iter"],
        "learning_method": config["learning_method"],
        "ngram_range": str(config["ngram_range"]),
        "min_df": config["min_df"],
        "max_df": config["max_df"],
        "max_features": config["max_features"],
        "train_documents": len(train_valid),
        "dev_documents": len(dev_valid),
        "test_documents": len(test_valid),
        "train_perplexity": train_perplexity,
        "dev_perplexity": dev_perplexity
    }

    experiment_results.append(result)

    print(f"\n{experiment_id} terminado.")
    print(f"Resultados guardados en: {experiment_dir}")


# ============================================================
# 7. GUARDAR COMPARACIÓN GENERAL
# ============================================================

if experiment_results:

    results_df = pd.DataFrame(experiment_results)

    results_file = os.path.join(
        OUTPUT_DIR,
        "experiments.tsv"
    )

    results_df.to_csv(
        results_file,
        sep="\t",
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("COMPARACIÓN GENERAL DE EXPERIMENTOS")
    print("=" * 65)

    print(
        results_df[
            [
                "experiment_id",
                "vocabulary_size",
                "train_perplexity",
                "dev_perplexity"
            ]
        ].to_string(index=False)
    )

    print("\nArchivo general guardado en:")
    print(results_file)

else:
    print("No se pudo completar ningún experimento.")

print("\nProceso finalizado.")
