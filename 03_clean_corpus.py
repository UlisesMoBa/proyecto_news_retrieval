import pandas as pd
from sklearn.model_selection import train_test_split

print("1. Cargando corpus original...")
df = pd.read_csv('news_raw_corpus.tsv', sep='\t')

# Función de limpieza y reasignación de etiquetas erróneas
def limpiar_y_reclasificar(row):
    cat = row['category']
    if cat == 'Ciencias':
        cat = 'Ciencia'
        
    if cat == 'Ciencia':
        text = str(row['title']).lower() + " " + str(row['description']).lower()
        
        eco_keywords = ['economía', 'peso', 'dólar', 'inflación', 'finanzas', 'banco', 'pib', 
                        't-mec', 'deuda', 'tipo de cambio', 'fmi', 'pemex', 'arancel']
        biz_keywords = ['empresas', 'mercado', 'inversión', 'negocio', 'compañía', 'alsea', 
                        'wall street', 'tesla', 'spacex', 'prudential']
        
        if any(kw in text for kw in eco_keywords):
            return 'Economía'
        elif any(kw in text for kw in biz_keywords):
            return 'Negocios'
            
    return cat

print("2. Aplicando limpieza y corrección de categorías...")
df['category'] = df.apply(limpiar_y_reclasificar, axis=1)

# Guardar el corpus limpio oficial
df.to_csv('news_raw_corpus_cleaned.tsv', sep='\t', index=False)
print("-> Archivo 'news_raw_corpus_cleaned.tsv' guardado exitosamente.")

# Preparar texto y ID para particionado
df['text'] = df['title'].fillna('') + " " + df['description'].fillna('')
df['document_id'] = df['id']

print("3. Creando particiones Train (70%), Dev (15%) y Test (15%)...")
# Separación train / temporal (dev + test) con estratificación
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df['category']
)

# Separación dev / test
dev_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df['category']
)

train_df = train_df.copy()
dev_df = dev_df.copy()
test_df = test_df.copy()

train_df['split'] = 'train'
dev_df['split'] = 'dev'
test_df['split'] = 'test'

labeled_docs = pd.concat([train_df, dev_df, test_df]).reset_index(drop=True)

# Generar labeled_documents.tsv requerido por el entregable
labeled_docs_out = labeled_docs[['document_id', 'text', 'category', 'split']]
labeled_docs_out.to_csv('labeled_documents.tsv', sep='\t', index=False)
print("-> Archivo 'labeled_documents.tsv'[cite: 9] generado con éxito.")
print(labeled_docs_out['split'].value_counts())