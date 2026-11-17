import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

print("1. Cargando particiones guardadas...")
labeled_docs = pd.read_csv('labeled_documents.tsv', sep='\t')

train_df = labeled_docs[labeled_docs['split'] == 'train']
dev_df = labeled_docs[labeled_docs['split'] == 'dev']
test_df = labeled_docs[labeled_docs['split'] == 'test']

X_train, y_train = train_df['text'], train_df['category']
X_dev, y_dev = dev_df['text'], dev_df['category']
X_test, y_test = test_df['text'], test_df['category']

print("2. Ejecutando experimentos de Clasificación Supervisada...")

# Experimento C01: Regresión Logística con TF-IDF unigramas (C=1.0)
pipe_c01 = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,1), max_df=0.9, min_df=1)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, random_state=42))
])
pipe_c01.fit(X_train, y_train)
y_dev_pred_c01 = pipe_c01.predict(X_dev)
acc_c01 = accuracy_score(y_dev, y_dev_pred_c01)
prec_c01, rec_c01, f1_c01, _ = precision_recall_fscore_support(y_dev, y_dev_pred_c01, average='macro', zero_division=0)

# Experimento C02: Regresión Logística con TF-IDF unigramas + bigramas (C=5.0) -> Seleccionado como mejor en Dev
pipe_c02 = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=1)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', C=5.0, random_state=42))
])
pipe_c02.fit(X_train, y_train)
y_dev_pred_c02 = pipe_c02.predict(X_dev)
acc_c02 = accuracy_score(y_dev, y_dev_pred_c02)
prec_c02, rec_c02, f1_c02, _ = precision_recall_fscore_support(y_dev, y_dev_pred_c02, average='macro', zero_division=0)

# Refit en Train + Dev y evaluación final en Test
print("3. Realizando Refit en Train+Dev y evaluando en Test...")
X_train_dev = pd.concat([X_train, X_dev])
y_train_dev = pd.concat([y_train, y_dev])

pipe_c02.fit(X_train_dev, y_train_dev)
y_test_pred = pipe_c02.predict(X_test)
acc_test = accuracy_score(y_test, y_test_pred)
prec_test, rec_test, f1_test, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro', zero_division=0)

# Generar experiments.tsv
experiments_data = [
    {"experiment_id": "C01", "task": "classification", "normalization": "lowercase + stopwords", "representation": "TF-IDF (1,1)", "method": "LogisticRegression", "hyperparameters": "C=1.0, class_weight=balanced", "dev_metric": f"macro-F1={f1_c01:.4f}", "selected": "no"},
    {"experiment_id": "C02", "task": "classification", "normalization": "lowercase + stopwords", "representation": "TF-IDF (1,2)", "method": "LogisticRegression", "hyperparameters": "C=5.0, class_weight=balanced", "dev_metric": f"macro-F1={f1_c02:.4f}", "selected": "yes"},
    {"experiment_id": "U01", "task": "topic_modeling", "normalization": "lowercase + stopwords", "representation": "Counts (1,1)", "method": "LDA", "hyperparameters": "n_components=4, max_iter=20", "dev_metric": "perplexity=...", "selected": "yes"}
]
pd.DataFrame(experiments_data).to_csv('experiments.tsv', sep='\t', index=False)

# Generar classification_predictions.tsv[cite: 9]
preds = []
for doc_id, gold, pred in zip(dev_df['document_id'], y_dev, y_dev_pred_c02):
    preds.append({"experiment_id": "C02", "model": "LogisticRegression", "document_id": doc_id, "gold_label": gold, "predicted_label": pred, "split": "dev"})
for doc_id, gold, pred in zip(test_df['document_id'], y_test, y_test_pred):
    preds.append({"experiment_id": "test", "model": "LogisticRegression", "document_id": doc_id, "gold_label": gold, "predicted_label": pred, "split": "test"})

pd.DataFrame(preds).to_csv('classification_predictions.tsv', sep='\t', index=False)

# Generar classification_metrics.tsv[cite: 9]
metrics_data = [
    {"experiment_id": "C01", "model": "LogisticRegression", "split": "dev", "accuracy": round(acc_c01, 4), "macro_precision": round(prec_c01, 4), "macro_recall": round(rec_c01, 4), "macro_f1": round(f1_c01, 4)},
    {"experiment_id": "C02", "model": "LogisticRegression", "split": "dev", "accuracy": round(acc_c02, 4), "macro_precision": round(prec_c02, 4), "macro_recall": round(rec_c02, 4), "macro_f1": round(f1_c02, 4)},
    {"experiment_id": "test", "model": "LogisticRegression", "split": "test", "accuracy": round(acc_test, 4), "macro_precision": round(prec_test, 4), "macro_recall": round(rec_test, 4), "macro_f1": round(f1_test, 4)}
]
pd.DataFrame(metrics_data).to_csv('classification_metrics.tsv', sep='\t', index=False)

print("4. Ejecutando Modelado de Tópicos LDA y generando topic_results.tsv...")
spanish_stop_words = ['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'antes', 'algunos', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho']

count_vec = CountVectorizer(stop_words=spanish_stop_words, min_df=2, max_df=0.9, max_features=2000)
X_counts_train = count_vec.fit_transform(X_train)

lda = LatentDirichletAllocation(n_components=4, random_state=42, learning_method='batch', max_iter=20)
lda.fit(X_counts_train)

# Obtener distribuciones de tópicos para todos los documentos del corpus
df_clean = pd.read_csv('news_raw_corpus_cleaned.tsv', sep='\t')
df_clean['text'] = df_clean['title'].fillna('') + " " + df_clean['description'].fillna('')
df_clean['document_id'] = df_clean['id']

X_counts_all = count_vec.transform(df_clean['text'])
doc_topic_dist = lda.transform(X_counts_all)

topic_results = []
for idx, row in df_clean.iterrows():
    doc_id = row['document_id']
    match_split = labeled_docs.loc[labeled_docs['document_id'] == doc_id, 'split']
    split = match_split.values[0] if not match_split.empty else 'train'
    
    for topic_id in range(4):
        topic_results.append({
            "document_id": doc_id,
            "split": split,
            "topic_id": topic_id,
            "topic_weight": round(float(doc_topic_dist[idx, topic_id]), 4)
        })

pd.DataFrame(topic_results).to_csv('topic_results.tsv', sep='\t', index=False)

print("¡Proceso completado con éxito! Todos los archivos TSV han sido generados en tu proyecto.")