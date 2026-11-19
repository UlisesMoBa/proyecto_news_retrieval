# Etapa II: aprendizaje tradicional aplicado a noticias

## 1. Corpus y protocolo

Se utilizó `data/processed/labeled_documents_split.tsv`, generado a partir de
746 registros RSS crudos. Después de quitar duplicados y aplicar una cuota
balanceada, quedaron 420 noticias procesadas: Ciencia (140), Cultura (140) y
Negocios (140). Economía se fusiona en Negocios. El texto de cada documento es
la concatenación de título y descripción. Las particiones son train=294, dev=63
y test=63, generadas con estratificación,
`shuffle=True` y `random_state=0`.

Los vectorizadores y modelos se ajustan con train durante la selección. Dev se
utiliza para comparar configuraciones. Después de congelar la configuración
ganadora, el clasificador se vuelve a ajustar con train+dev y test se evalúa
una sola vez.

## 2. Normalización y representaciones

El ejecutor `src/18_stage2_complete.py` compara tres configuraciones:

1. Tokenización, minúsculas y eliminación de acentos.
2. Limpieza de URLs y caracteres no alfabéticos.
3. Limpieza más eliminación de stopwords españolas.

La lematización con spaCy existe en `src/04_normalizacion.py`, pero no se
incluye en la selección oficial de esta etapa. Tampoco se utiliza SVD, PCA u
otra reducción de dimensionalidad; `min_df`, `max_df` y `max_features` solo
controlan el vocabulario.

Para clasificación se comparan representaciones binarias, de frecuencia y
TF-IDF, todas con unigramas y bigramas. LDA utiliza CountVectorizer, como exige
el documento de la etapa.

## 3. Tarea A: clasificación supervisada

Se comparan MultinomialNB, LogisticRegression y LinearSVC. La métrica primaria
de selección es macro-F1, acompañada por accuracy, precision macro y recall
macro. También se generan reportes por clase, matrices de confusión,
predicciones y los diez primeros errores de dev.

La configuración seleccionada fue:

- Preprocesamiento: tokenización.
- Representación: TF-IDF.
- N-gramas: `(1, 2)`.
- Modelo: LogisticRegression balanceada.
- Macro-F1 en dev: `0.6484`.
- Macro-F1 final en test: `0.5525`.

El resultado menor en test muestra que las clases minoritarias siguen siendo
difíciles. En particular, Economía y Negocios tienen muchos menos documentos
que Ciencia y Cultura, por lo que la métrica macro es más sensible a esos
errores. La confusión léxica entre noticias de cultura, ciencia y economía
también puede producir errores cuando título y descripción comparten nombres
propios o vocabulario institucional.

## 4. Tarea no supervisada seleccionada: LDA

Se seleccionó LDA como la única tarea no supervisada de esta ejecución. No se
utilizan categorías para ajustar los tópicos. Se comparan cuatro
configuraciones variando:

- Tokenización y limpieza.
- Stopwords.
- Unigramas frente a unigramas+bigrama.
- `min_df` y `max_df`.
- Número de tópicos.
- `max_iter`.
- `learning_method`.

La configuración elegida por menor perplejidad en dev fue `U01`, con
tokenización, CountVectorizer de unigramas, cuatro tópicos y `max_iter=30`.
Sus resultados fueron:

- Perplejidad train: consultar `lda_metrics.tsv`.
- Perplejidad dev: `276.7439`.
- Perplejidad test: `2392.3880`.

Los tópicos se interpretan mediante sus diez palabras de mayor peso y las
proporciones documento-tópico. Los identificadores de tópico son arbitrarios y
no representan categorías supervisadas.

## 5. Entregables

La ejecución se realiza con:

```powershell
python src\18_stage2_complete.py
```

Los resultados se guardan en `results/stage2/`:

- `classification_metrics.tsv`
- `classification_predictions.tsv`
- `classification_dev_reports.tsv`
- `classification_dev_confusion_matrices.tsv`
- `classification_dev_errors_top10.tsv`
- `lda_metrics.tsv`
- `lda_topic_results.tsv`
- `lda_topic_words.tsv`
- `experiments.tsv`

## 6. Limitaciones

- La lematización no forma parte de la comparación oficial.
- No se aplicó reducción de dimensionalidad.
- El desbalance de clases limita la interpretación de accuracy y explica la
  diferencia entre F1 ponderado y macro-F1.
- La perplejidad de LDA no debe compararse entre vocabularios distintos sin
  considerar que las configuraciones usan espacios de características
  diferentes.
- La explicación final de errores debe complementarse con la revisión manual
  de los diez documentos almacenados en `classification_dev_errors_top10.tsv`.
