import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline

# 1. Cargar el corpus limpio oficial
print("Cargando corpus limpio...")
df = pd.read_csv('news_raw_corpus_cleaned.tsv', sep='\t')

# Concatenar título y descripción para el análisis de tópicos
X_corpus = df['title'].fillna('') + " " + df['description'].fillna('')

# 2. Lista de stopwords en español para filtrar ruido léxico común
spanish_stop_words = [
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 
    'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 
    'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 
    'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 
    'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 
    'antes', 'algunos', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho'
]

# 3. Configurar el Pipeline de LDA
# Nota: LDA requiere CountVectorizer (frecuencias absolutas de conteo), no TF-IDF.
lda_pipeline = Pipeline([
    ('text_representation', CountVectorizer(
        stop_words=spanish_stop_words, 
        min_df=2, 
        max_df=0.9, 
        max_features=3000
    )),
    ('topic_modeling', LatentDirichletAllocation(
        n_components=4,         # 4 tópicos temáticos (equivalentes a tus 4 categorías)
        random_state=42,        # Semilla reproducible
        learning_method='batch', 
        max_iter=20
    ))
])

# 4. Entrenar el modelo de forma no supervisada
print("Entrenando modelo LDA sobre el corpus de noticias...")
lda_pipeline.fit(X_corpus)

# 5. Extraer e imprimir las 10 palabras más representativas de cada tópico
vectorizer = lda_pipeline.named_steps['text_representation']
lda_model = lda_pipeline.named_steps['topic_modeling']
feature_names = vectorizer.get_feature_names_out()

print("\n--- TÓPICOS DESCUBRIDOS POR LDA (K=4) ---")
for topic_idx, topic in enumerate(lda_model.components_):
    top_features_ind = topic.argsort()[:-10 - 1:-1]
    top_features = [feature_names[i] for i in top_features_ind]
    print(f"Tópico #{topic_idx + 1}: {', '.join(top_features)}")