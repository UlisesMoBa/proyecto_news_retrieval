
import os
import re
import unicodedata
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "labeled_documents.tsv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "labeled_documents_clean.tsv"
)

DUPLICATES_FILE = os.path.join(
    EXPERIMENTS_DIR,
    "removed_duplicates.tsv"
)

CONFLICTS_FILE = os.path.join(
    EXPERIMENTS_DIR,
    "duplicate_category_conflicts.tsv"
)

RANDOM_STATE = 0


# ============================================================
# NORMALIZACIÓN
# ============================================================

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


# ============================================================
# CARGAR CORPUS
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

print("Cargando corpus original...")

df = pd.read_csv(INPUT_FILE, sep="\t")

columnas_requeridas = ["document_id", "text", "category"]

for columna in columnas_requeridas:
    if columna not in df.columns:
        raise ValueError(
            f"No se encontró la columna obligatoria: {columna}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

df["text"] = df["text"].fillna("").astype(str)
df["category"] = df["category"].fillna("").astype(str)

print("Documentos originales:", len(df))


# ============================================================
# VALIDAR DOCUMENTOS VACÍOS
# ============================================================

df["normalized_text"] = df["text"].apply(normalizar)

empty_documents = df[df["normalized_text"] == ""]

if not empty_documents.empty:
    print("\nERROR: Se encontraron documentos sin contenido.")
    print(empty_documents[["document_id", "category"]].to_string(index=False))

    raise ValueError(
        "Revisa los documentos vacíos antes de continuar."
    )


# ============================================================
# DETECTAR DUPLICADOS
# ============================================================

print("\nBuscando documentos duplicados...")

duplicate_mask = df.duplicated(
    subset=["normalized_text"],
    keep=False
)

duplicate_groups = df[duplicate_mask].groupby("normalized_text")

conflicts = []
duplicates_to_remove = []

for normalized_text, group in duplicate_groups:

    categories = group["category"].unique()

    # Si el mismo contenido tiene categorías diferentes,
    # no debemos eliminarlo automáticamente.
    if len(categories) > 1:

        conflicts.append({
            "document_ids": ", ".join(
                group["document_id"].astype(str)
            ),
            "categories": ", ".join(categories),
            "titles_or_text": " || ".join(
                group["text"].str[:200]
            )
        })

    else:
        # Conservar el primer registro y marcar los restantes.
        duplicates_to_remove.extend(
            group.index.tolist()[1:]
        )


# ============================================================
# REVISAR CONFLICTOS DE CATEGORÍA
# ============================================================

if conflicts:

    conflicts_df = pd.DataFrame(conflicts)

    conflicts_df.to_csv(
        CONFLICTS_FILE,
        sep="\t",
        index=False,
        encoding="utf-8"
    )

    print("\nSe encontraron duplicados con categorías diferentes.")
    print(conflicts_df.to_string(index=False))

    raise ValueError(
        "\nProceso detenido. Revisa el archivo:\n"
        + CONFLICTS_FILE
        + "\nNo se generó el nuevo corpus."
    )


# ============================================================
# ELIMINAR DUPLICADOS
# ============================================================

duplicates_df = df.loc[duplicates_to_remove].copy()

clean_df = df.drop(
    index=duplicates_to_remove
).copy()

clean_df = clean_df.drop(
    columns=["normalized_text"]
)

duplicates_df.to_csv(
    DUPLICATES_FILE,
    sep="\t",
    index=False,
    encoding="utf-8"
)

print("\nRESULTADO DE LA LIMPIEZA")
print("Documentos originales:", len(df))
print("Documentos eliminados:", len(duplicates_df))
print("Documentos restantes:", len(clean_df))


# ============================================================
# VERIFICAR CANTIDAD POR CATEGORÍA
# ============================================================

print("\nDOCUMENTOS POR CATEGORÍA")

print(
    clean_df["category"]
    .value_counts()
    .to_string()
)

category_counts = clean_df["category"].value_counts()

if category_counts.min() < 3:
    raise ValueError(
        "Alguna categoría tiene menos de 3 documentos. "
        "No es posible generar las particiones solicitadas "
        "con estratificación."
    )


# ============================================================
# GENERAR PARTICIONES 70 / 15 / 15
# ============================================================

print("\nGenerando particiones...")

train_df, temp_df = train_test_split(
    clean_df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=clean_df["category"]
)

dev_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["category"]
)

train_df = train_df.copy()
dev_df = dev_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
dev_df["split"] = "dev"
test_df["split"] = "test"


# ============================================================
# UNIFICAR CORPUS
# ============================================================

final_df = pd.concat(
    [train_df, dev_df, test_df],
    ignore_index=True
)

# Mezclar registros para evitar que queden agrupados por split.
final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# VALIDACIONES FINALES
# ============================================================

print("\nVALIDACIÓN FINAL")

print("Total de documentos:", len(final_df))

print("\nDistribución de particiones:")

for split in ["train", "dev", "test"]:

    cantidad = (final_df["split"] == split).sum()
    porcentaje = cantidad / len(final_df) * 100

    print(
        f"{split.upper():5} | "
        f"{cantidad:4} documentos | "
        f"{porcentaje:.2f}%"
    )


print("\nDistribución de categorías por partición:")

print(
    pd.crosstab(
        final_df["split"],
        final_df["category"]
    )
)


# Comprobar que no existan textos repetidos
normalized_final = final_df["text"].apply(normalizar)

if normalized_final.duplicated().any():
    raise ValueError(
        "ERROR: Todavía existen documentos duplicados."
    )


# Comprobar que los identificadores sean únicos
if final_df["document_id"].duplicated().any():
    raise ValueError(
        "ERROR: Existen identificadores de documentos repetidos."
    )


# ============================================================
# GUARDAR NUEVO CORPUS
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False,
    encoding="utf-8"
)

print("\nCORPUS LIMPIO GENERADO CORRECTAMENTE")
print(OUTPUT_FILE)

print("\nArchivos adicionales:")
print(DUPLICATES_FILE)

print("\nProceso terminado.")
