import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# 1. Cargar directamente el corpus LIMPIO
df = pd.read_csv('news_raw_corpus_cleaned.tsv', sep='\t')

# 2. Definir características (título + descripción) y etiquetas
X = df['title'].fillna('') + " " + df['description'].fillna('')
y = df['category']

# 3. Separación en entrenamiento y prueba (estratificada)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 4. Construir y entrenar el Pipeline
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', C=5.0))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("--- Resultados con Corpus Limpio ---")
print(f"Accuracy Global: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, zero_division=0))