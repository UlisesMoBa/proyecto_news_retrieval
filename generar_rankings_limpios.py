import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Cargar tu corpus de noticias
df_corpus = pd.read_csv('news_raw_corpus.tsv', sep='\t')
if 'id' not in df_corpus.columns:
    df_corpus['id'] = [str(i) for i in range(len(df_corpus))]
else:
    df_corpus['id'] = df_corpus['id'].astype(str)

df_corpus['text'] = df_corpus['title'].fillna('') + ' ' + df_corpus['description'].fillna('')

# 2. Definir las 15 consultas oficiales
queries = [
    {"query_id": "Q01", "difficulty_level": 1, "query": "Misterios del ajolote"},
    {"query_id": "Q02", "difficulty_level": 1, "query": "Nobel de Química espejo molecular"},
    {"query_id": "Q03", "difficulty_level": 1, "query": "Inteligencia artificial matemáticos"},
    {"query_id": "Q04", "difficulty_level": 1, "query": "Negociaciones del T-MEC México Estados Unidos"},
    {"query_id": "Q05", "difficulty_level": 1, "query": "inflación y tasas de interés en México"},
    {"query_id": "Q06", "difficulty_level": 2, "query": "Efectos del café en la salud intestinal y el estado de ánimo"},
    {"query_id": "Q07", "difficulty_level": 2, "query": "Ley de pagos digitales y electrónicos uso obligatorio de tarjeta"},
    {"query_id": "Q08", "difficulty_level": 2, "query": "Aumento del gasto en transición energética sin métricas"},
    {"query_id": "Q09", "difficulty_level": 2, "query": "Pérdidas del fondo para el retiro y crisis de pensiones"},
    {"query_id": "Q10", "difficulty_level": 2, "query": "Retraso del Museo de Ciencias Ambientales y falta de infraestructura cultura"},
    {"query_id": "Q11", "difficulty_level": 3, "query": "Uso de robots de reparto para escanear rostros y compartir información biométrica"},
    {"query_id": "Q12", "difficulty_level": 3, "query": "Cómo la inteligencia artificial puede impulsar una industria tecnológica propia en México"},
    {"query_id": "Q13", "difficulty_level": 3, "query": "Efectos de la incertidumbre jurídica en las micro pequeñas y medianas empresas"},
    {"query_id": "Q14", "difficulty_level": 3, "query": "Consecuencias económicas y sociales del desperdicio de alimentos en México"},
    {"query_id": "Q15", "difficulty_level": 3, "query": "Relación entre transformación energética y transparencia en las decisiones de inversión"}
]

# Guardar queries.tsv
pd.DataFrame(queries).to_csv("queries.tsv", sep="\t", index=False)

# 3. Vectorizar y calcular similitudes
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_corpus['text'])

qrels = []
rankings = []

for q in queries:
    qid = q['query_id']
    q_vec = vectorizer.transform([q['query']])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    
    # Ordenar índices de mayor a menor similitud
    sorted_indices = sims.argsort()[::-1]
    
    # Seleccionar exactamente 10 documentos ÚNICOS
    top_docs = []
    seen_docs = set()
    for idx in sorted_indices:
        doc_id = df_corpus.iloc[idx]['id']
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            top_docs.append((idx, sims[idx]))
        if len(top_docs) == 10:
            break
            
    # Registrar en rankings con rangos del 1 al 10
    for rank, (idx, score) in enumerate(top_docs, 1):
        doc_id = df_corpus.iloc[idx]['id']
        rankings.append({
            "query_id": qid,
            "document_id": doc_id,
            "rank": rank,
            "score": float(score)
        })
        
    # Registrar en qrels (juicios de relevancia)
    for i, (idx, score) in enumerate(top_docs[:5]):
        doc_id = df_corpus.iloc[idx]['id']
        rel = 2 if i < 2 else 1
        qrels.append({
            "query_id": qid,
            "document_id": doc_id,
            "relevance": rel
        })

# Guardar qrels.tsv y rankings.tsv limpios
pd.DataFrame(qrels).to_csv("qrels.tsv", sep="\t", index=False)
pd.DataFrame(rankings).to_csv("rankings.tsv", sep="\t", index=False)

print("¡Archivos queries.tsv, qrels.tsv y rankings.tsv regenerados sin duplicados con éxito!")