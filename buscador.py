import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import sys

# 1. Cargar el modelo de lenguaje para limpiar la consulta del usuario
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Error: No se encontró el modelo. Ejecuta: python -m spacy download es_core_news_sm")
    sys.exit(1)

def normalizar_consulta(texto):
    """Aplica la misma limpieza a la búsqueda que la que usaste en el corpus"""
    doc = nlp(texto.lower())
    # Filtra signos de puntuación, stopwords y extrae el lema
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(tokens)

def cargar_datos():
    try:
        # AQUÍ CARGAMOS AMBOS ARCHIVOS
        df_raw = pd.read_csv('news_raw_corpus.tsv', sep='\t')
        df_norm = pd.read_csv('news_normalized_corpus.tsv', sep='\t')
        
        # Preparamos la columna "contenido" en ambos DataFrames
        df_raw['content'] = df_raw['title'].fillna('') + ' ' + df_raw['description'].fillna('')
        df_norm['content'] = df_norm['title'].fillna('') + ' ' + df_norm['description'].fillna('')
        
        # Aseguramos que tengan IDs para mostrarlos
        if 'id' not in df_raw.columns:
            df_raw['id'] = [str(i) for i in range(len(df_raw))]
            
        return df_raw, df_norm
    except FileNotFoundError as e:
        print(f"Error: Faltan archivos del corpus. Detalle: {e}")
        sys.exit(1)

def cargar_benchmark():
    try:
        return pd.read_csv('queries.tsv', sep='\t')
    except FileNotFoundError:
        print("Error: No se encontró 'queries.tsv'.")
        return None

def configurar_parametros():
    print("\n" + "="*40)
    print(" CONFIGURACIÓN DEL ESPACIO VECTORIAL")
    print("="*40)
    
    print("\n1. ¿En qué campo deseas buscar?")
    print("   [1] Título")
    print("   [2] Contenido (Título + Descripción)")
    campo = 'title' if input("Elige (1/2): ").strip() == '1' else 'content'
    
    print("\n2. ¿Qué N-gramas deseas utilizar?")
    print("   [1] Unigrama (Palabras individuales)")
    print("   [2] Bigrama (Pares de palabras)")
    ngram_range = (1, 1) if input("Elige (1/2): ").strip() == '1' else (2, 2)
    
    print("\n3. Representación mediante el modelo de espacio vectorial:")
    print("   [1] Frecuencia (TF)")
    print("   [2] Binaria")
    print("   [3] TF-IDF")
    m_opt = input("Elige (1/2/3): ").strip()
    
    if m_opt == '1':
        vectorizer = CountVectorizer(ngram_range=ngram_range, binary=False)
    elif m_opt == '2':
        vectorizer = CountVectorizer(ngram_range=ngram_range, binary=True)
    else:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        
    return campo, vectorizer

def ejecutar_busqueda(query_text, df_raw, df_norm, campo, vectorizer):
    # LIMPIEZA DE LA CONSULTA
    query_limpia = normalizar_consulta(query_text)
    print(f"\n[Interpretando consulta como]: '{query_limpia}'")
    
    # MATEMÁTICAS: Usamos el df_norm (CORPUS NORMALIZADO) para entrenar la matriz
    matriz_corpus = vectorizer.fit_transform(df_norm[campo])
    vector_query = vectorizer.transform([query_limpia])
    
    similitudes = cosine_similarity(vector_query, matriz_corpus).flatten()
    indices_top10 = similitudes.argsort()[::-1][:10]
    
    print("\n--- TOP 10 RESULTADOS ---")
    for rank, idx in enumerate(indices_top10, 1):
        score = similitudes[idx]
        if score > 0:
            # VITRINA: Usamos el df_raw (CORPUS CRUDO) para imprimir un texto legible
            print(f"{rank}. Score: {score:.4f} | ID: {df_raw.iloc[idx]['id']}")
            print(f"   Título: {df_raw.iloc[idx]['title'][:80]}...")
        else:
            print(f"{rank}. [Sin coincidencias suficientes]")

def main():
    print("Cargando modelo de lenguaje y corpus...")
    df_raw, df_norm = cargar_datos()
    
    while True:
        print("\n" + "="*40)
        print(" SISTEMA DE RECUPERACIÓN DE NOTICIAS")
        print("="*40)
        print("[1] Búsqueda Dirigida (Consulta libre)")
        print("[2] Búsqueda sobre el Benchmark")
        print("[3] Salir")
        
        opcion = input("\nElige una opción (1/2/3): ").strip()
        
        if opcion == '3':
            break
        elif opcion == '1':
            campo, vectorizer = configurar_parametros()
            query = input("\nIngresa tu búsqueda dirigida: ").strip()
            if query:
                ejecutar_busqueda(query, df_raw, df_norm, campo, vectorizer)
        elif opcion == '2':
            df_queries = cargar_benchmark()
            if df_queries is not None:
                for index, row in df_queries.iterrows():
                    print(f"{row['query_id']}: {row['query']}")
                q_id = input("\nIngresa el ID de la consulta (ej. Q01): ").strip().upper()
                seleccion = df_queries[df_queries['query_id'] == q_id]
                if not seleccion.empty:
                    campo, vectorizer = configurar_parametros()
                    ejecutar_busqueda(seleccion.iloc[0]['query'], df_raw, df_norm, campo, vectorizer)

if __name__ == "__main__":
    main()