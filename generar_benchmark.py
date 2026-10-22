import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# 1. consultas del benchmark
consultas = [
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

# 2. Generar el archivo queries.tsv benchmark
df_queries = pd.DataFrame(consultas)
df_queries.to_csv("queries.tsv", sep="\t", index=False)
print("✅ Archivo queries.tsv generado exitosamente.")

# 3. Leer tu corpus más reciente para generar el qrels.tsv automáticamente
try:
    df_corpus = pd.read_csv("news_raw_corpus.tsv", sep="\t")
    
    # Combinamos título y descripción para buscar mejor
    df_corpus['texto_completo'] = df_corpus['title'].fillna('') + ' ' + df_corpus['description'].fillna('')
    
    # Vectorizar rápidamente para encontrar similitudes
    vectorizador = TfidfVectorizer()
    matriz_tfidf = vectorizador.fit_transform(df_corpus['texto_completo'])
    
    qrels = []
    
    for q in consultas:
        # Transformar la consulta al mismo espacio
        vector_q = vectorizador.transform([q['query']])
        similitudes = cosine_similarity(vector_q, matriz_tfidf).flatten()
        
        # Obtener los top 5 documentos para esta consulta
        indices_top = similitudes.argsort()[::-1][:5]
        
        for i, idx in enumerate(indices_top):
            score = similitudes[idx]
            # Solo guardamos si hay una similitud real (evita guardar basura en consultas como "dinosaurios")
            if score > 0.05:
                # Asignamos relevancia 2 a los dos primeros lugares, y 1 a los siguientes
                relevancia = 2 if i < 2 else 1
                doc_id = df_corpus.iloc[idx]['id']
                qrels.append({"query_id": q['query_id'], "document_id": doc_id, "relevance": relevancia})
                
    # Generar el archivo qrels.tsv
    df_qrels = pd.DataFrame(qrels)
    df_qrels.to_csv("qrels.tsv", sep="\t", index=False)
    print(f"✅ Archivo qrels.tsv generado exitosamente con {len(df_qrels)} juicios de relevancia adaptados a tu corpus.")

except FileNotFoundError:
    print("❌ Error: No se encontró 'news_raw_corpus.tsv'. Asegúrate de que el nombre sea correcto.")