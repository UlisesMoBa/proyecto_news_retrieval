
import os
import re
import hashlib
import unicodedata
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR, "data", "processed", "labeled_documents.tsv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CLEAN = os.path.join(
    OUTPUT_DIR, "labeled_documents_deduplicated.tsv"
)

OUTPUT_REMOVED = os.path.join(
    OUTPUT_DIR, "removed_duplicates.tsv"
)

OUTPUT_REVIEW = os.path.join(
    OUTPUT_DIR, "similarity_review.tsv"
)

SIMILARITY_THRESHOLD = 0.80


def normalize(text):
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("Cargando corpus original...")

df = pd.read_csv(INPUT_FILE, sep="\t")

required = ["document_id", "text", "category"]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Falta la columna: {col}")

df["text"] = df["text"].fillna("").astype(str)
df["normalized_text"] = df["text"].apply(normalize)

# --------------------------------------------------
# 1. Eliminar textos exactamente duplicados
# --------------------------------------------------

df["text_hash"] = df["normalized_text"].apply(
    lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
)

# Mantener el primer registro de cada grupo
df["is_duplicate"] = df.duplicated(
    subset="text_hash",
    keep="first"
)

removed = df[df["is_duplicate"]].copy()
clean = df[~df["is_duplicate"]].copy()

# Registrar el documento original que se conserva
canonical_ids = (
    clean.drop_duplicates("text_hash")
    .set_index("text_hash")["document_id"]
    .to_dict()
)

removed["kept_document_id"] = removed["text_hash"].map(
    canonical_ids
)

removed["removal_reason"] = "Duplicado exacto normalizado"

# --------------------------------------------------
# 2. Detectar similitudes para revisión manual
# --------------------------------------------------

print("Buscando noticias similares...")

similar_pairs = []

if len(clean) > 1:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(clean["normalized_text"])

    similarities = cosine_similarity(X)

    for i in range(len(clean)):
        for j in range(i + 1, len(clean)):
            score = similarities[i, j]

            if score >= SIMILARITY_THRESHOLD:
                a = clean.iloc[i]
                b = clean.iloc[j]

                similar_pairs.append({
                    "document_id_1": a["document_id"],
                    "document_id_2": b["document_id"],
                    "category_1": a["category"],
                    "category_2": b["category"],
                    "similarity": round(float(score), 4),
                    "text_1": a["text"],
                    "text_2": b["text"],
                    "decision": "PENDIENTE DE REVISION"
                })

review = pd.DataFrame(similar_pairs)

# --------------------------------------------------
# 3. Guardar resultados
# --------------------------------------------------

# Eliminar columnas auxiliares
clean = clean.drop(
    columns=["normalized_text", "text_hash", "is_duplicate"]
)

removed = removed.drop(
    columns=["normalized_text", "text_hash", "is_duplicate"]
)

clean.to_csv(
    OUTPUT_CLEAN,
    sep="\t",
    index=False,
    encoding="utf-8"
)

removed.to_csv(
    OUTPUT_REMOVED,
    sep="\t",
    index=False,
    encoding="utf-8"
)

review.to_csv(
    OUTPUT_REVIEW,
    sep="\t",
    index=False,
    encoding="utf-8"
)

# --------------------------------------------------
# 4. Resumen
# --------------------------------------------------

print("\n========== RESULTADOS ==========")

print("Documentos originales:", len(df))
print("Duplicados exactos eliminados:", len(removed))
print("Documentos conservados:", len(clean))
print("Pares similares por revisar:", len(review))

print("\nArchivos generados:")
print(OUTPUT_CLEAN)
print(OUTPUT_REMOVED)
print(OUTPUT_REVIEW)
