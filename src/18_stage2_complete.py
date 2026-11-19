"""Reproducible Stage II pipeline.

Runs the mandatory supervised classification task and selects LDA as the
single unsupervised task.  All vectorizers and models are fitted on train
only during selection; test is used once for final reporting.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "labeled_documents_split.tsv"
OUT = ROOT / "results" / "stage2"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0

SPANISH_STOPWORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como",
    "con", "contra", "cual", "cuando", "de", "del", "desde", "donde",
    "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre", "era",
    "es", "esa", "esas", "ese", "eso", "esos", "esta", "estas", "este",
    "esto", "estos", "fue", "fueron", "ha", "han", "hasta", "hay", "la",
    "las", "le", "les", "lo", "los", "mas", "me", "mi", "mis", "mientras",
    "muy", "nada", "ni", "no", "nos", "nosotros", "nuestra", "nuestro",
    "o", "otra", "otras", "otro", "otros", "para", "pero", "poco", "por",
    "porque", "que", "quien", "quienes", "se", "sea", "ser", "si", "sin",
    "sobre", "son", "su", "sus", "tambien", "te", "tiene", "tienen",
    "todo", "todos", "tu", "tus", "un", "una", "uno", "unos", "unas", "y",
    "ya",
}


def strip_accents(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    return "".join(char for char in value if not unicodedata.combining(char))


def tokenize(value: str) -> str:
    value = strip_accents(str(value).lower())
    return " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?", value))


def clean(value: str) -> str:
    value = re.sub(r"https?://\S+", " ", str(value).lower())
    value = strip_accents(value)
    return " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?", value))


def remove_stopwords(value: str) -> str:
    return " ".join(word for word in clean(value).split() if word not in SPANISH_STOPWORDS)


PREPROCESSORS: dict[str, Callable[[str], str]] = {
    "tokenization": tokenize,
    "cleaning": clean,
    "cleaning_stopwords": remove_stopwords,
}


def load_data() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"Corpus not found: {INPUT}")
    df = pd.read_csv(INPUT, sep="\t")
    required = {"document_id", "text", "category", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df["document_id"].duplicated().any():
        raise ValueError("document_id values must be unique.")
    if df["category"].isna().any() or df["text"].isna().any():
        raise ValueError("The corpus contains missing text or category values.")
    if set(df["split"]) != {"train", "dev", "test"}:
        raise ValueError("Expected train, dev and test partitions.")
    return df.assign(text=lambda frame: frame["text"].astype(str))


def make_classifier(name: str):
    if name == "MultinomialNB":
        return MultinomialNB(alpha=1.0)
    if name == "LogisticRegression":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        )
    if name == "LinearSVC":
        return LinearSVC(class_weight="balanced", random_state=SEED)
    raise ValueError(f"Unknown classifier: {name}")


def make_vectorizer(representation: str):
    common = {"ngram_range": (1, 2), "min_df": 1, "max_df": 0.95}
    if representation == "binary":
        return TfidfVectorizer(binary=True, use_idf=False, norm=None, **common)
    if representation == "frequency":
        return TfidfVectorizer(use_idf=False, norm=None, **common)
    if representation == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, norm="l2", **common)
    raise ValueError(f"Unknown representation: {representation}")


def classification() -> tuple[pd.DataFrame, dict[str, object]]:
    df = load_data()
    train, dev, test = (df[df["split"] == part].copy() for part in ("train", "dev", "test"))
    records: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    reports: list[pd.DataFrame] = []
    matrices: list[pd.DataFrame] = []

    for prep_name, prep in PREPROCESSORS.items():
        for representation in ("binary", "frequency", "tfidf"):
            for model_name in ("MultinomialNB", "LogisticRegression", "LinearSVC"):
                experiment_id = f"C_{prep_name}_{representation}_{model_name}"
                pipe = Pipeline([
                    ("preprocess", FunctionTransformer(np.vectorize(prep), validate=False)),
                    ("vectorizer", make_vectorizer(representation)),
                    ("classifier", make_classifier(model_name)),
                ])
                pipe.fit(train["text"], train["category"])
                pred = pipe.predict(dev["text"])
                records.append({
                    "experiment_id": experiment_id,
                    "task": "classification",
                    "preprocessing": prep_name,
                    "representation": representation,
                    "ngram_range": "(1,2)",
                    "algorithm": model_name,
                    "hyperparameters": "class_weight=balanced; random_state=0"
                    if model_name != "MultinomialNB" else "alpha=1.0",
                    "split": "dev",
                    "accuracy": accuracy_score(dev["category"], pred),
                    "macro_precision": precision_score(dev["category"], pred, average="macro", zero_division=0),
                    "macro_recall": recall_score(dev["category"], pred, average="macro", zero_division=0),
                    "macro_f1": f1_score(dev["category"], pred, average="macro", zero_division=0),
                    "selected": False,
                })
                predictions.append(pd.DataFrame({
                    "experiment_id": experiment_id,
                    "model": model_name,
                    "document_id": dev["document_id"],
                    "gold_label": dev["category"],
                    "predicted_label": pred,
                    "split": "dev",
                }))
                report = pd.DataFrame(classification_report(
                    dev["category"], pred, output_dict=True, zero_division=0
                )).T.reset_index().rename(columns={"index": "label"})
                report.insert(0, "experiment_id", experiment_id)
                reports.append(report)
                matrices.append(pd.DataFrame(
                    confusion_matrix(dev["category"], pred),
                    index=sorted(dev["category"].unique()),
                    columns=sorted(dev["category"].unique()),
                ).reset_index(names="true_label").assign(experiment_id=experiment_id))

    metrics = pd.DataFrame(records)
    best_idx = metrics["macro_f1"].idxmax()
    best = metrics.loc[best_idx].copy()
    metrics.loc[best_idx, "selected"] = True
    prep = PREPROCESSORS[str(best["preprocessing"])]
    final_pipe = Pipeline([
        ("preprocess", FunctionTransformer(np.vectorize(prep), validate=False)),
        ("vectorizer", make_vectorizer(str(best["representation"]))),
        ("classifier", make_classifier(str(best["algorithm"]))),
    ])
    train_dev = pd.concat([train, dev], ignore_index=True)
    final_pipe.fit(train_dev["text"], train_dev["category"])
    test_pred = final_pipe.predict(test["text"])
    final_id = "C_FINAL_TEST"
    metrics = pd.concat([metrics, pd.DataFrame([{
        **best.to_dict(),
        "experiment_id": final_id,
        "split": "test",
        "accuracy": accuracy_score(test["category"], test_pred),
        "macro_precision": precision_score(test["category"], test_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(test["category"], test_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(test["category"], test_pred, average="macro", zero_division=0),
        "selected": True,
    }])], ignore_index=True)
    predictions.append(pd.DataFrame({
        "experiment_id": final_id,
        "model": best["algorithm"],
        "document_id": test["document_id"],
        "gold_label": test["category"],
        "predicted_label": test_pred,
        "split": "test",
    }))
    final_report = pd.DataFrame(classification_report(
        test["category"], test_pred, output_dict=True, zero_division=0
    )).T.reset_index().rename(columns={"index": "label"})
    final_report.insert(0, "experiment_id", final_id)
    reports.append(final_report)
    matrices.append(pd.DataFrame(
        confusion_matrix(test["category"], test_pred),
        index=sorted(test["category"].unique()),
        columns=sorted(test["category"].unique()),
    ).reset_index(names="true_label").assign(experiment_id=final_id))
    errors = predictions[0].iloc[0:0].copy()
    dev_best_pred = next(
        prediction
        for prediction in predictions
        if prediction["experiment_id"].iloc[0] == best["experiment_id"]
    )
    errors = dev_best_pred[dev_best_pred["gold_label"] != dev_best_pred["predicted_label"]].copy()
    errors["error_rank"] = range(1, len(errors) + 1)
    errors = errors.merge(dev[["document_id", "text"]], on="document_id", how="left")
    metrics.to_csv(OUT / "classification_metrics.tsv", sep="\t", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "classification_predictions.tsv", sep="\t", index=False)
    pd.concat(reports, ignore_index=True).to_csv(OUT / "classification_dev_reports.tsv", sep="\t", index=False)
    pd.concat(matrices, ignore_index=True).to_csv(OUT / "classification_dev_confusion_matrices.tsv", sep="\t", index=False)
    errors.head(10).to_csv(OUT / "classification_dev_errors_top10.tsv", sep="\t", index=False)
    return metrics, {"best": best, "errors": errors}


def lda() -> pd.DataFrame:
    df = load_data()
    train, dev, test = (df[df["split"] == part].copy() for part in ("train", "dev", "test"))
    configs = [
        ("U01", "tokenization", (1, 1), 3, 0.95, 4, 30, "batch"),
        ("U02", "cleaning", (1, 1), 2, 0.85, 4, 50, "batch"),
        ("U03", "cleaning_stopwords", (1, 1), 2, 0.85, 4, 50, "batch"),
        ("U04", "cleaning_stopwords", (1, 2), 2, 0.85, 5, 50, "online"),
    ]
    records: list[dict[str, object]] = []
    fitted: dict[str, tuple[object, object, dict[str, pd.DataFrame]]] = {}
    for exp_id, prep_name, ngrams, min_df, max_df, topics, max_iter, learning in configs:
        prep = PREPROCESSORS[prep_name]
        vectorizer = CountVectorizer(
            preprocessor=prep, tokenizer=str.split, token_pattern=None,
            ngram_range=ngrams, min_df=min_df, max_df=max_df, max_features=3000,
        )
        x_train = vectorizer.fit_transform(train["text"])
        x_dev = vectorizer.transform(dev["text"])
        x_test = vectorizer.transform(test["text"])
        model = LatentDirichletAllocation(
            n_components=topics, max_iter=max_iter,
            learning_method=learning, random_state=SEED,
        ).fit(x_train)
        records.append({
            "experiment_id": exp_id, "task": "LDA", "preprocessing": prep_name,
            "representation": "CountVectorizer", "ngram_range": str(ngrams),
            "algorithm": "LatentDirichletAllocation",
            "n_components": topics, "max_iter": max_iter,
            "learning_method": learning, "vocabulary_size": len(vectorizer.vocabulary_),
            "train_perplexity": model.perplexity(x_train),
            "dev_perplexity": model.perplexity(x_dev),
            "test_perplexity": np.nan, "selected": False,
        })
        fitted[exp_id] = (vectorizer, model, {"train": train, "dev": dev, "test": test})
    lda_metrics = pd.DataFrame(records)
    selected_idx = lda_metrics["dev_perplexity"].idxmin()
    lda_metrics.loc[selected_idx, "selected"] = True
    selected_id = str(lda_metrics.loc[selected_idx, "experiment_id"])
    vectorizer, model, parts = fitted[selected_id]
    train_dev = pd.concat([train, dev], ignore_index=True)
    final_vectorizer = CountVectorizer(
        preprocessor=PREPROCESSORS[str(lda_metrics.loc[selected_idx, "preprocessing"])],
        tokenizer=str.split, token_pattern=None,
        ngram_range=eval(str(lda_metrics.loc[selected_idx, "ngram_range"])),
        min_df=2, max_df=0.85, max_features=3000,
    )
    x_train_dev = final_vectorizer.fit_transform(train_dev["text"])
    x_test = final_vectorizer.transform(test["text"])
    final_model = LatentDirichletAllocation(
        n_components=int(lda_metrics.loc[selected_idx, "n_components"]),
        max_iter=int(lda_metrics.loc[selected_idx, "max_iter"]),
        learning_method=str(lda_metrics.loc[selected_idx, "learning_method"]),
        random_state=SEED,
    ).fit(x_train_dev)
    lda_metrics.loc[selected_idx, "test_perplexity"] = final_model.perplexity(x_test)
    topic_rows = []
    for split_name, frame, matrix in (
        ("train", train_dev, x_train_dev), ("test", test, x_test)
    ):
        distributions = final_model.transform(matrix)
        for row, distribution in zip(frame.itertuples(), distributions):
            for topic_id, weight in enumerate(distribution, start=1):
                topic_rows.append({
                    "experiment_id": selected_id, "document_id": row.document_id,
                    "split": split_name, "topic_id": topic_id,
                    "topic_weight": float(weight),
                })
    words = []
    names = final_vectorizer.get_feature_names_out()
    for topic_id, weights in enumerate(final_model.components_, start=1):
        for rank, index in enumerate(weights.argsort()[::-1][:10], start=1):
            words.append({"experiment_id": selected_id, "topic_id": topic_id,
                          "rank": rank, "word": names[index],
                          "weight": float(weights[index])})
    pd.DataFrame(topic_rows).to_csv(OUT / "lda_topic_results.tsv", sep="\t", index=False)
    pd.DataFrame(words).to_csv(OUT / "lda_topic_words.tsv", sep="\t", index=False)
    lda_metrics.to_csv(OUT / "lda_metrics.tsv", sep="\t", index=False)
    return lda_metrics


def main() -> None:
    metrics, classification_info = classification()
    lda_metrics = lda()
    selected = metrics[metrics["selected"] == True].copy()
    log = pd.concat([
        metrics.assign(selection_stage="dev_or_final"),
        lda_metrics.assign(selection_stage="dev_or_final"),
    ], ignore_index=True, sort=False)
    log.to_csv(OUT / "experiments.tsv", sep="\t", index=False)
    data = load_data()
    partition_counts = data["split"].value_counts().to_dict()
    category_counts = data["category"].value_counts().sort_index()
    best_test = metrics[metrics["experiment_id"] == "C_FINAL_TEST"].iloc[0]
    summary = [
        "# Stage II reproducible results",
        "",
        f"- Documents: {len(data)}; categories: {data['category'].nunique()}",
        f"- Categories: {', '.join(f'{name}={count}' for name, count in category_counts.items())}",
        f"- Partitions: train={partition_counts.get('train', 0)}, dev={partition_counts.get('dev', 0)}, test={partition_counts.get('test', 0)}",
        f"- Best classification configuration: {classification_info['best']['experiment_id']}",
        f"- Best dev macro-F1: {classification_info['best']['macro_f1']:.4f}",
        f"- Final test macro-F1: {best_test['macro_f1']:.4f}",
        f"- LDA selected configuration: {lda_metrics.loc[lda_metrics['selected'], 'experiment_id'].iloc[0]}",
        f"- LDA test perplexity: {lda_metrics.loc[lda_metrics['selected'], 'test_perplexity'].iloc[0]:.4f}",
        "",
        "## Implemented requirements",
        "",
        "- Three preprocessing configurations: tokenization, cleaning, cleaning plus Spanish stopwords.",
        "- Three representations for classification: binary, frequency and TF-IDF; unigram plus bigram features.",
        "- Three traditional classifiers: MultinomialNB, LogisticRegression and LinearSVC.",
        "- Macro-F1 is the selection metric on dev.",
        "- The final classifier is refit on train plus dev and evaluated once on test.",
        "- LDA is the single selected unsupervised task; its vectorizer and model are fitted on train during selection.",
        "- LDA trials vary preprocessing, n-grams, vocabulary thresholds, number of topics, iterations and learning method.",
        "- Outputs include metrics, predictions, per-class reports, confusion matrices, top ten topic words, topic proportions and ten dev errors.",
        "",
        "The test partition is evaluated only after selection.",
    ]
    (OUT / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("Stage II completed.")
    print(selected[["experiment_id", "split", "macro_f1"]].to_string(index=False))
    print(lda_metrics[["experiment_id", "dev_perplexity", "selected"]].to_string(index=False))


if __name__ == "__main__":
    main()
