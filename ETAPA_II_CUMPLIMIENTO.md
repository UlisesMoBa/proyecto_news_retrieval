# Etapa II — Matriz de cumplimiento y uso

## Ejecución reproducible

Desde la raíz del proyecto:

```powershell
python -m pip install -r requirements-stage2.txt
python src\18_stage2_complete.py
```

La ejecución usa `data\processed\labeled_documents_split.tsv` y genera la
evidencia en `results\stage2\`. La semilla reproducible es `random_state=0`.
Las versiones fijadas son NumPy 2.4.3, pandas 3.0.1 y scikit-learn 1.9.1.

## Datos y protocolo

| Requisito | Evidencia | Estado |
|---|---|---|
| Reutilizar la colección de noticias e IDs estables | `data/processed/labeled_documents_split.tsv` | Cumplido |
| Texto como título concatenado con descripción | `data/processed/labeled_documents_split.tsv`, columna `text` | Cumplido |
| Etiqueta tomada de `category` | Columna `category` del corpus | Cumplido |
| Particiones disjuntas | `split=train/dev/test` | Cumplido |
| Train/dev/test reproducibles | `src/create_splits.py`, `random_state=0`, estratificación | Cumplido |
| No ajustar transformaciones con test | `src/18_stage2_complete.py` | Cumplido |
| 420 documentos procesados y 3 categorías balanceadas | `results/stage2/README.md` | Cumplido |

Distribución: train=294, dev=63 y test=63. Categorías: Ciencia=140,
Cultura=140 y Negocios=140. Economía se fusiona en Negocios. El corpus crudo
conserva 746 registros para futuras ampliaciones.

## Normalización y representación

| Requisito | Configuraciones implementadas | Estado |
|---|---|---|
| Comparar al menos tres preprocesamientos | tokenización; limpieza; limpieza + stopwords | Cumplido |
| Tokenización y minúsculas | `PREPROCESSORS` en `18_stage2_complete.py` | Cumplido |
| Limpieza de URLs y caracteres | función `clean` | Cumplido |
| Stopwords | función `remove_stopwords` | Cumplido |
| Lematización | Disponible en `src/04_normalizacion.py`, no usada en la selección final | Limitación documentada |
| Comparar binario, frecuencia y TF-IDF | Tres configuraciones en clasificación | Cumplido |
| Comparar n-gramas | Unigramas y bigramas en clasificación; unigramas y bigramas en LDA | Cumplido |
| Reducción dimensional | No se usó SVD/PCA; `max_features` solo limita vocabulario | No requerida |
| Pipelines contra leakage | `Pipeline` en todas las variantes de clasificación | Cumplido |

## Tarea A — clasificación supervisada

| Requisito | Evidencia |
|---|---|
| Clasificar la categoría de la noticia | `classification_predictions.tsv` |
| Comparar al menos tres algoritmos | MultinomialNB, LogisticRegression y LinearSVC |
| Comparar hiperparámetros/configuraciones | `classification_metrics.tsv` |
| Seleccionar por macro-F1 en dev | Fila `selected=True` |
| Accuracy, precisión, recall y F1 | `classification_metrics.tsv` |
| Resultados por clase | `classification_dev_reports.tsv` |
| Matrices de confusión | `classification_dev_confusion_matrices.tsv` |
| Revisar al menos diez errores | `classification_dev_errors_top10.tsv` |
| Reentrenar con train+dev | `C_FINAL_TEST` |
| Evaluar test una sola vez después de seleccionar | `classification_metrics.tsv` |

La configuración seleccionada es LogisticRegression balanceada con tokenización,
TF-IDF y unigramas+bigrama. Macro-F1 en dev: 0.5979; macro-F1 final en test:
0.5525.

## Tarea no supervisada seleccionada — LDA

Se selecciona LDA como la única tarea no supervisada de esta entrega. Las
categorías se conservan únicamente para metadatos y análisis externo; no se
usan para ajustar el modelo.

| Requisito | Evidencia |
|---|---|
| LDA como único modelo de tópicos | `src/18_stage2_complete.py` |
| CountVectorizer | `lda_metrics.tsv` |
| Variar preprocesamiento | U01–U04 |
| Variar n-gramas y vocabulario | U01–U04 |
| Variar número de tópicos | U01–U04 |
| Variar iteraciones y método de aprendizaje | U01–U04 |
| Diez palabras por tópico | `lda_topic_words.tsv` |
| Distribución documento-tópico | `lda_topic_results.tsv` |
| Perplejidad train/dev/test | `lda_metrics.tsv` |
| Manejar documentos sin términos | Vectorizadores y matrices válidas en el pipeline |
| No confundir IDs de tópicos con clases | Declarado en el reporte |

La configuración LDA seleccionada es U01, con cuatro tópicos, unigramas,
`max_iter=30` y aprendizaje batch. Perplejidad: train=145.9618,
dev=383.9403 y test=3761.6787.

## Registro, entregables e interpretación

| Requisito | Archivo |
|---|---|
| Registro completo de ensayos | `results/stage2/experiments.tsv` |
| Predicciones de clasificación | `results/stage2/classification_predictions.tsv` |
| Métricas | `results/stage2/classification_metrics.tsv` |
| Tópicos y pesos | `results/stage2/lda_topic_results.tsv` |
| Palabras de tópicos | `results/stage2/lda_topic_words.tsv` |
| Errores supervisados | `results/stage2/classification_dev_errors_top10.tsv` |
| Reporte analítico | `STAGE2_REPORT.md` |
| Dependencias y versiones | `requirements-stage2.txt` |
| Instrucciones de ejecución | `src/README.md` |

## Limitaciones declaradas

La lematización está implementada como herramienta de normalización heredada,
pero no se incluyó en la selección final. Tampoco se utilizó SVD/PCA. La
desigualdad entre categorías afecta especialmente la macro-F1 y explica parte
de la diferencia entre dev y test. La perplejidad solo es comparable entre
configuraciones con vocabularios equivalentes. La revisión de los diez errores
debe interpretarse junto con el texto almacenado en el archivo de errores.
