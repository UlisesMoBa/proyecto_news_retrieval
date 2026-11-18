
import os
import re
import hashlib
import unicodedata
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "labeled_documents.tsv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

# Validar columnas necesarias
if "text" not in df.columns:
    raise ValueError(
        "El archivo no contiene la columna 'text'. "
        f"Columnas disponibles: {list(df.columns)}"
    )

if "split" not in df.columns:
    raise ValueError("El archivo no contiene la columna 'split'.")

if "document_id" not in df.columns:
    df["document_id"] = range(1, len(df) + 1)

df["text"] = df["text"].fillna("").astype(str)

# Si no existe title, generar uno auxiliar desde text
if "title" not in df.columns:
    print("No se encontró la columna 'title'.")
    print("Se generarán títulos auxiliares desde el texto.")
    df["title"] = df["text"].str[:150]

df["title"] = df["title"].fillna("").astype(str)

# Normalización
df["normalized_text"] = df["text"].apply(normalizar)
df["normalized_title"] = df["title"].apply(normalizar)

# Identificador para detectar textos idénticos
df["text_hash"] = df["normalized_text"].apply(
    lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
)

# Documentos con textos duplicados
duplicates_text = df[
    df.duplicated("text_hash", keep=False)
].copy()

# Documentos con títulos duplicados
duplicates_title = df[
    df["normalized_title"].ne("") &
    df.duplicated("normalized_title", keep=False)
].copy()

duplicates_text = duplicates_text.sort_values(
    ["text_hash", "split", "document_id"]
)

duplicates_title = duplicates_title.sort_values(
    ["normalized_title", "split", "document_id"]
)

# Resumen por partición
summary = []

for split, group in df.groupby("split"):
    repeated = group[
        group.duplicated("text_hash", keep=False)
    ]

    summary.append({
        "split": split,
        "total_documents": len(group),
        "documents_with_exact_duplicates": len(repeated),
        "unique_documents": group["text_hash"].nunique()
    })

summary_df = pd.DataFrame(summary)

# Agrupar documentos duplicados y comprobar si cruzan particiones
cross_split = []

for text_hash, group in df.groupby("text_hash"):
    if len(group) > 1:
        splits = sorted(group["split"].unique())

        cross_split.append({
            "text_hash": text_hash,
            "copies": len(group),
            "splits": ", ".join(splits),
            "crosses_splits": len(splits) > 1,
            "document_ids": ", ".join(
                group["document_id"].astype(str)
            ),
            "titles": " || ".join(
                group["title"].astype(str)
            )
        })

cross_split_df = pd.DataFrame(cross_split)

# Guardar reportes
duplicates_text.to_csv(
    os.path.join(OUTPUT_DIR, "duplicate_text_audit.tsv"),
    sep="\t",
    index=False,
    encoding="utf-8"
)

duplicates_title.to_csv(
    os.path.join(OUTPUT_DIR, "duplicate_title_audit.tsv"),
    sep="\t",
    index=False,
    encoding="utf-8"
)

summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "duplicate_summary.tsv"),
    sep="\t",
    index=False,
    encoding="utf-8"
)

cross_split_df.to_csv(
    os.path.join(OUTPUT_DIR, "duplicate_groups.tsv"),
    sep="\t",
    index=False,
    encoding="utf-8"
)

# Mostrar resultados
print("\nRESUMEN POR PARTICIÓN")
print(summary_df.to_string(index=False))

print("\nGRUPOS DE DOCUMENTOS DUPLICADOS:")
print(len(cross_split_df))

print("\nGRUPOS QUE APARECEN EN VARIAS PARTICIONES:")
if not cross_split_df.empty:
    print(cross_split_df["crosses_splits"].sum())
else:
    print(0)

print("\nReportes guardados en:")
print(OUTPUT_DIR)
