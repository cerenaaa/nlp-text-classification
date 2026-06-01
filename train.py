"""Train and evaluate text classification models."""
import argparse
from data.synthetic_corpus import generate_corpus, train_test_split
from models.tfidf_classifier import TFIDFMultiLabelClassifier
from evaluation.multilabel_metrics import evaluate
from sklearn.preprocessing import MultiLabelBinarizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tfidf")
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()

    print("Generating corpus...")
    samples = generate_corpus(n=args.n_samples)
    train, test = train_test_split(samples)

    if args.model == "tfidf":
        print("
Training TF-IDF classifier...")
        clf = TFIDFMultiLabelClassifier()
        clf.fit(train)

        print("
Evaluating...")
        test_texts = [s["text"] for s in test]
        preds = clf.predict(test_texts)
        mlb = clf.mlb
        y_true = mlb.transform([s["labels"] for s in test])
        y_pred = mlb.transform(preds)
        y_score = clf.predict_proba(test_texts)
        evaluate(y_true, y_pred, y_score)

if __name__ == "__main__":
    main()
