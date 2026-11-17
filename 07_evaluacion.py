import pandas as pd
import joblib
import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("Cargando modelos y datos para la evaluación...")
nlp = spacy.load("es_core_news_sm")
vectorizador = joblib.load('desc_tfidf_vectorizer.pkl')
matriz_documentos = joblib.load('desc_tfidf_matrix.pkl')

df_queries = pd.read_csv("queries.tsv", sep="\t")
df_qrels = pd.read_csv("qrels.tsv", sep="\t")
df_corpus = pd.read_csv("news_raw_corpus_cleaned.tsv", sep="\t")

# === LA SOLUCIÓN: FORZAR QUE TODOS LOS IDs SEAN TEXTO LIMPIO ===
df_queries['query_id'] = df_queries['query_id'].astype(str).str.strip()

# Asegurar compatibilidad de columnas en qrels (por si usaste espacio o guion bajo)
if 'document id' in df_qrels.columns:
    df_qrels.rename(columns={'document id': 'document_id'}, inplace=True)

df_qrels['query_id'] = df_qrels['query_id'].astype(str).str.strip()
df_qrels['document_id'] = df_qrels['document_id'].astype(str).str.strip()

if 'id' in df_corpus.columns:
    df_corpus['id'] = df_corpus['id'].astype(str).str.strip()
else:
    df_corpus['id'] = [str(i) for i in range(len(df_corpus))]
# ===============================================================

def limpiar_y_normalizar(texto):
    if not isinstance(texto, str): return ""
    texto = re.sub(r'http[s]?://\S+', '', texto)
    texto = re.sub(r'[^\w\sáéíóúÁÉÍÓÚñÑüÜ]', ' ', texto).lower()
    texto = re.sub(r'\s+', ' ', texto).strip()
    doc = nlp(texto)
    return " ".join([t.lemma_ for t in doc if t.pos_ not in {"DET", "ADP", "CCONJ", "SCONJ", "PRON"}])

rankings = []
metricas = []

print("\nEjecutando consultas y calculando métricas @10...\n")

for _, row in df_queries.iterrows():
    qid = str(row['query_id']).strip()
    consulta_texto = row['query']
    
    consulta_norm = limpiar_y_normalizar(consulta_texto)
    vec_consulta = vectorizador.transform([consulta_norm])
    similitudes = cosine_similarity(vec_consulta, matriz_documentos).flatten()
    
    # Obtener siempre exactamente los Top 10 índices
    top_10_indices = similitudes.argsort()[::-1][:10]
    top_10_docs = []
    
    for rank, idx in enumerate(top_10_indices):
        score = similitudes[idx]
        doc_id = str(df_corpus.iloc[idx]['id']).strip()
        top_10_docs.append(doc_id)
        
        rankings.append({
            "query_id": qid, 
            "document_id": doc_id, 
            "rank": rank + 1,
            "score": score
        })
            
    recuperados = set(top_10_docs)
    relevantes_reales = set(df_qrels[df_qrels['query_id'] == qid]['document_id'])
    
    if len(relevantes_reales) > 0:
        aciertos = recuperados.intersection(relevantes_reales)
        
        precision = len(aciertos) / 10.0
        recall = len(aciertos) / len(relevantes_reales)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metricas.append({"query_id": qid, "Precision@10": precision, "Recall@10": recall, "F1@10": f1})
        print(f"{qid} | Precisión: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}")

# Exportar conservando el nombre 'document_id'
df_rankings = pd.DataFrame(rankings)
df_rankings.to_csv("rankings.tsv", sep="\t", index=False)
print("\n✅ Archivo rankings.tsv generado exitosamente.")

df_metricas = pd.DataFrame(metricas)
if not df_metricas.empty:
    print("\n=== PROMEDIO GLOBAL DEL SISTEMA ===")
    print(f"Mean Precision@10: {df_metricas['Precision@10'].mean():.4f}")
    print(f"Mean Recall@10:    {df_metricas['Recall@10'].mean():.4f}")
    print(f"Mean F1@10:        {df_metricas['F1@10'].mean():.4f}")