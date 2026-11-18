
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "labeled_documents_deduplicated.tsv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "labeled_documents_final.tsv"
)

# ID que conservaremos del duplicado T-MEC
KEEP_ID = "1304bf20-2302-4c18-976c-77ad8b858b7a"

# ID que eliminaremos
REMOVE_ID = "2e7fb33f-09ff-4783-90a2-d1eabf63bebe"

df = pd.read_csv(INPUT_FILE, sep="\t")

# Comprobar que ambos documentos existen
ids = set(df["document_id"].astype(str))

if KEEP_ID not in ids or REMOVE_ID not in ids:
    raise ValueError("No se encontraron ambos documentos T-MEC.")

# Eliminar únicamente el duplicado confirmado
df = df[df["document_id"].astype(str) != REMOVE_ID].copy()

# Guardar corpus definitivo
df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False,
    encoding="utf-8"
)

print("CORPUS DEFINITIVO")
print("Documentos:", len(df))
print("\nDistribución por categoría:")
print(df["category"].value_counts())

print("\nArchivo generado:")
print(OUTPUT_FILE)
