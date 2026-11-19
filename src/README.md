Librerias
pip install feedparser pandas spacy scikit-learn joblib
Modelo de lenguaje en español para spacy
-m spacy download es_core_news_sm

Etapa II reproducible
---------------------

La ejecución integrada de la Etapa II se realiza desde la raíz del proyecto:

python src\18_stage2_complete.py

El script utiliza data\processed\labeled_documents_split.tsv y genera los
entregables en results\stage2:

- classification_metrics.tsv: todos los experimentos de dev y la evaluación final.
- classification_predictions.tsv: predicciones de dev y test.
- classification_dev_reports.tsv: resultados por clase.
- classification_dev_confusion_matrices.tsv: matrices de confusión.
- classification_dev_errors_top10.tsv: diez errores de dev del mejor modelo.
- lda_metrics.tsv: ensayos LDA y selección por menor perplejidad en dev.
- lda_topic_results.tsv: proporciones documento-tópico.
- lda_topic_words.tsv: diez palabras principales por tópico.
- experiments.tsv: registro completo de experimentos y decisiones.

El pipeline compara tres normalizaciones (tokenización, limpieza y limpieza con
stopwords), tres representaciones (binaria, frecuencia y TF-IDF) y tres
clasificadores (MultinomialNB, LogisticRegression y LinearSVC). LDA es la única
tarea no supervisada seleccionada. El vocabulario y los modelos se ajustan con
train durante la selección; el modelo final se reentrena con train+dev y test
se utiliza solo para la evaluación final.

Para ampliar el corpus, ejecutar primero:

python src\collect_corpus.py --target-total 750 --target-per-category 250
python src\prepare_corpus.py --max-per-category 140
python src\prepare_corpus.py

El recolector conserva el esquema de Fase I, asigna las categorías Ciencia,
Cultura y Negocios (Economía se fusiona en Negocios), elimina duplicados por
URL/título y puede ejecutarse nuevamente para agregar noticias futuras sin
duplicar registros. La matriz de cumplimiento completa está en
../ETAPA_II_CUMPLIMIENTO.md.

Flujo de ejecución
1 rss_feedparser_v2
2 limpiar_duplicados
3 normalizacion
4 vsm
5 buscador (se puede omitir)
6 generar_benchmark
7 evalucion
