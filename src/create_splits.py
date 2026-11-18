
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================
# 1. RUTAS
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "labeled_documents_final.tsv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "labeled_documents_split.tsv"
)

# =====================================================
# 2. CARGAR CORPUS DEFINITIVO
# =====================================================

print("Cargando corpus definitivo...")

df = pd.read_csv(INPUT_FILE, sep="\t")

if df["document_id"].duplicated().any():
    raise ValueError("Existen document_id duplicados.")

if df["category"].isna().any():
    raise ValueError("Existen documentos sin categoría.")

if df["text"].isna().any():
    raise ValueError("Existen documentos sin texto.")

print("Documentos:", len(df))
print("\nDistribución original:")
print(df["category"].value_counts())

# =====================================================
# 3. PRIMERA PARTICIÓN: 70% TRAIN, 30% TEMPORAL
# =====================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=0,
    stratify=df["category"]
)

# =====================================================
# 4. SEGUNDA PARTICIÓN: 15% DEV, 15% TEST
# =====================================================

dev_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=0,
    stratify=temp_df["category"]
)

# =====================================================
# 5. ASIGNAR ETIQUETAS DE PARTICIÓN
# =====================================================

train_df = train_df.copy()
dev_df = dev_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
dev_df["split"] = "dev"
test_df["split"] = "test"

# =====================================================
# 6. UNIFICAR Y ORDENAR
# =====================================================

final_df = pd.concat(
    [train_df, dev_df, test_df],
    ignore_index=True
)

# Mantener un orden estable por identificador
final_df = final_df.sort_values(
    "document_id"
).reset_index(drop=True)

# =====================================================
# 7. GUARDAR
# =====================================================

final_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False,
    encoding="utf-8"
)

# =====================================================
# 8. MOSTRAR RESULTADOS
# =====================================================

print("\nPARTICIONES GENERADAS")
print("----------------------")

print("\nCantidad por partición:")
print(final_df["split"].value_counts())

print("\nDistribución por categoría y partición:")
print(
    pd.crosstab(
        final_df["category"],
        final_df["split"],
        margins=True
    )
)

print("\nPorcentajes:")
print(
    final_df["split"].value_counts(normalize=True).mul(100).round(2)
)

print("\nArchivo generado:")
print(OUTPUT_FILE)

print("\nProceso terminado correctamente.")
