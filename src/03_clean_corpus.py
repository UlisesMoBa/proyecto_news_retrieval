
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

INPUT_FILE = Path("news_raw_corpus.tsv")

OUTPUT_CLEANED = Path("news_raw_corpus_cleaned.tsv")
OUTPUT_LABELED = Path("labeled_documents.tsv")
OUTPUT_UNLABELED = Path("news_without_category.tsv")
OUTPUT_EMPTY = Path("news_empty_text.tsv")

RANDOM_STATE = 0

# ============================================================
# 1. CARGAR CORPUS ORIGINAL
# ============================================================

print("1. Cargando corpus original...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE, sep="\t")

print(f"Documentos originales: {len(df)}")

# ============================================================
# 2. VALIDAR COLUMNAS E IDENTIFICADORES
# ============================================================

print("\n2. Validando identificadores...")

required_columns = [
    "id",
    "title",
    "description",
    "category"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Faltan las siguientes columnas: {missing_columns}"
    )

if df["id"].isna().any():
    raise ValueError(
        "Existen documentos sin identificador."
    )

if df["id"].duplicated().any():
    raise ValueError(
        "Existen identificadores duplicados."
    )

# Conservar el identificador original
df["document_id"] = df["id"]

# ============================================================
# 3. CONSTRUIR TEXTO DE LAS NOTICIAS
# ============================================================

print("\n3. Construyendo texto de las noticias...")

df["text"] = (
    df["title"].fillna("").astype(str).str.strip()
    + " "
    + df["description"].fillna("").astype(str).str.strip()
).str.strip()

# Identificar documentos sin texto
empty_mask = df["text"].eq("")

empty_documents = df[empty_mask].copy()

if not empty_documents.empty:
    empty_documents.to_csv(
        OUTPUT_EMPTY,
        sep="\t",
        index=False
    )

    print(
        f"Se encontraron {len(empty_documents)} documentos sin texto."
    )
    print(f"Archivo generado: {OUTPUT_EMPTY}")

# Excluir documentos vacíos
df = df[~empty_mask].copy()

print(f"Documentos con texto: {len(df)}")

# ============================================================
# 4. REVISAR Y NORMALIZAR CATEGORÍAS
# ============================================================

print("\n4. Revisando categorías...")

print("\nDistribución original de categorías:")
print(df["category"].value_counts(dropna=False))

# Normalizar etiquetas equivalentes.
# Se considera que "Ciencias" y "Ciencia" representan
# la misma categoría.
df["category"] = df["category"].replace({
    "Ciencias": "Ciencia"
})

print("\nDistribución de categorías normalizada:")
print(df["category"].value_counts(dropna=False))

# Identificar documentos sin categoría
missing_category_mask = (
    df["category"].isna()
    | df["category"].astype("string").str.strip().eq("")
).fillna(True)

unlabeled_documents = df[missing_category_mask].copy()

labeled_documents = df[~missing_category_mask].copy()

if not unlabeled_documents.empty:
    unlabeled_documents.to_csv(
        OUTPUT_UNLABELED,
        sep="\t",
        index=False
    )

    print(
        f"\nDocumentos sin categoría: {len(unlabeled_documents)}"
    )
    print(f"Archivo generado: {OUTPUT_UNLABELED}")

if labeled_documents.empty:
    raise ValueError(
        "No existen documentos con categorías válidas."
    )

# ============================================================
# 5. GUARDAR CORPUS LIMPIO
# ============================================================

print("\n5. Guardando corpus limpio...")

df.to_csv(
    OUTPUT_CLEANED,
    sep="\t",
    index=False
)

print(f"Archivo generado: {OUTPUT_CLEANED}")

# ============================================================
# 6. CREAR PARTICIONES TRAIN, DEV Y TEST
# ============================================================

print("\n6. Creando particiones Train, Dev y Test...")

# Revisar cantidad de documentos por categoría
class_counts = labeled_documents["category"].value_counts()

print("\nDocumentos disponibles por categoría:")
print(class_counts)

# Para realizar la división estratificada en dos etapas,
# cada categoría necesita suficientes documentos.
if (class_counts < 3).any():
    raise ValueError(
        "Existen categorías con menos de 3 documentos. "
        "No es posible garantizar su representación "
        "en Train, Dev y Test mediante esta división."
    )

try:

    # Primera división: 70% Train y 30% temporal
    train_df, temp_df = train_test_split(
        labeled_documents,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=labeled_documents["category"]
    )

    # Segunda división: 15% Dev y 15% Test
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=temp_df["category"]
    )

except ValueError as error:

    raise ValueError(
        "\nNo fue posible realizar la división estratificada.\n"
        "Revisa la cantidad de documentos por categoría.\n"
        f"Detalle: {error}"
    )

# ============================================================
# 7. ASIGNAR IDENTIFICADORES DE PARTICIÓN
# ============================================================

print("\n7. Asignando particiones...")

train_df = train_df.copy()
dev_df = dev_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
dev_df["split"] = "dev"
test_df["split"] = "test"

# Unir los tres conjuntos
labeled_docs = pd.concat(
    [train_df, dev_df, test_df],
    ignore_index=True
)

# Seleccionar columnas del entregable
labeled_docs_out = labeled_docs[
    [
        "document_id",
        "text",
        "category",
        "split"
    ]
].copy()

# ============================================================
# 8. VALIDAR PARTICIONES
# ============================================================

print("\n8. Validando resultados...")

if not labeled_docs_out["document_id"].is_unique:
    raise ValueError(
        "Hay identificadores duplicados en las particiones."
    )

if not labeled_docs_out["text"].str.strip().ne("").all():
    raise ValueError(
        "Existen documentos sin texto."
    )

if labeled_docs_out["category"].isna().any():
    raise ValueError(
        "Existen documentos sin categoría."
    )

valid_splits = {"train", "dev", "test"}

if not set(labeled_docs_out["split"].unique()).issubset(valid_splits):
    raise ValueError(
        "Se encontraron nombres de partición no válidos."
    )

# Comprobar que ningún documento aparece en más de una partición
if (
    set(train_df["document_id"]) & set(dev_df["document_id"])
    or set(train_df["document_id"]) & set(test_df["document_id"])
    or set(dev_df["document_id"]) & set(test_df["document_id"])
):
    raise ValueError(
        "Se encontraron documentos repetidos entre particiones."
    )

# ============================================================
# 9. GUARDAR LABELED_DOCUMENTS.TSV
# ============================================================

print("\n9. Guardando labeled_documents.tsv...")

labeled_docs_out.to_csv(
    OUTPUT_LABELED,
    sep="\t",
    index=False
)

print(f"Archivo generado: {OUTPUT_LABELED}")

# ============================================================
# 10. MOSTRAR RESUMEN FINAL
# ============================================================

print("\n" + "=" * 55)
print("RESUMEN DE PARTICIONES")
print("=" * 55)

print("\nCantidad de documentos por partición:")
print(labeled_docs_out["split"].value_counts())

print("\nDistribución de categorías por partición:")
print(
    pd.crosstab(
        labeled_docs_out["category"],
        labeled_docs_out["split"]
    )
)

print("\nPorcentajes de las particiones:")
print(
    (
        labeled_docs_out["split"].value_counts(normalize=True) * 100
    ).round(2)
)

print("\nProceso terminado correctamente.")
print("Archivos generados:")
print(f"- {OUTPUT_CLEANED}")
print(f"- {OUTPUT_LABELED}")

if not unlabeled_documents.empty:
    print(f"- {OUTPUT_UNLABELED}")

if not empty_documents.empty:
    print(f"- {OUTPUT_EMPTY}")
