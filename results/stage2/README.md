# Stage II reproducible results

- Documents: 420; categories: 3
- Categories: Ciencia=140, Cultura=140, Negocios=140
- Partitions: train=294, dev=63, test=63
- Best classification configuration: C_cleaning_stopwords_binary_LogisticRegression
- Best dev macro-F1: 0.6484
- Final test macro-F1: 0.5525
- LDA selected configuration: U01
- LDA test perplexity: 2392.3880

## Implemented requirements

- Three preprocessing configurations: tokenization, cleaning, cleaning plus Spanish stopwords.
- Three representations for classification: binary, frequency and TF-IDF; unigram plus bigram features.
- Three traditional classifiers: MultinomialNB, LogisticRegression and LinearSVC.
- Macro-F1 is the selection metric on dev.
- The final classifier is refit on train plus dev and evaluated once on test.
- LDA is the single selected unsupervised task; its vectorizer and model are fitted on train during selection.
- LDA trials vary preprocessing, n-grams, vocabulary thresholds, number of topics, iterations and learning method.
- Outputs include metrics, predictions, per-class reports, confusion matrices, top ten topic words, topic proportions and ten dev errors.

The test partition is evaluated only after selection.
