
import os
import re
import unicodedata
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR, "data", "processed", "labeled_documents.tsv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Umbral para considerar dos noticias como posibles duplicados
SIMILARITY_THRESHOLD = 0.80


def normalizar(texto):
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto)

    texto = "".join(
        c for c in texto
        if not unicodedata.combining(c)
    )

    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


print("Cargando corpus...")

df = pd.read_csv(INPUT_FILE, sep="\t")

if "text" not in df.columns or "split" not in df.columns:
    raise ValueError("Faltan las columnas text o split.")

if "document_id" not in df.columns:
    df["document_id"] = range(1, len(df) + 1)

df["text"] = df["text"].fillna("").astype(str)
df["normalized_text"] = df["text"].apply(normalizar)

# Eliminar documentos vacíos de la comparación
valid = df[df["normalized_text"].str.len() > 0].copy()

print("Documentos analizados:", len(valid))

# TF-IDF de palabras y bigramas
vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    max_df=1.0,
    sublinear_tf=True
)

X = vectorizer.fit_transform(valid["normalized_text"])

# Matriz de similitud
similarities = cosine_similarity(X, dense_output=True)

# Buscar pares similares
pairs = []

for i in range(len(valid)):
    for j in range(i + 1, len(valid)):
        similarity = similarities[i, j]

        if similarity >= SIMILARITY_THRESHOLD:
            a = valid.iloc[i]
            b = valid.iloc[j]

            pairs.append({
                "document_id_1": a["document_id"],
                "document_id_2": b["document_id"],
                "split_1": a["split"],
                "split_2": b["split"],
                "category_1": a.get("category", ""),
                "category_2": b.get("category", ""),
                "similarity": round(float(similarity), 4),
                "crosses_splits": a["split"] != b["split"],
                "text_1": a["text"],
                "text_2": b["text"]
            })

# Guardar resultados
results = pd.DataFrame(pairs)

if not results.empty:
    results = results.sort_values(
        "similarity",
        ascending=False
    )

results.to_csv(
    os.path.join(OUTPUT_DIR, "similar_documents.tsv"),
    sep="\t",
    index=False,
    encoding="utf-8"
)

# Resumen
print("\nAUDITORÍA DE SIMILITUD")
print("Pares similares encontrados:", len(results))

if not results.empty:
    print(
        "Pares que cruzan particiones:",
        results["crosses_splits"].sum()
    )

    print("\nPRINCIPALES COINCIDENCIAS:")
    print(
        results[
            [
                "document_id_1",
                "document_id_2",
                "similarity",
                "split_1",
                "split_2"
            ]
        ].head(15).to_string(index=False)
    )

print("\nArchivo generado:")
print(os.path.join(OUTPUT_DIR, "similar_documents.tsv"))
