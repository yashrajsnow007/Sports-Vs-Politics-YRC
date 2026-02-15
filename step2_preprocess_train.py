"""
Sports vs Politics Classifier
Complete implementation with:
- Feature Extraction: Bag of Words, TF-IDF, N-grams
- ML Classifiers: Naive Bayes, Logistic Regression, SVM
- Evaluation: Accuracy, Precision, Recall, F1-score
"""

import os
import csv
import math
import random
import urllib.request
from collections import Counter

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

def download_ag_news():
    """Download AG News dataset."""
    train_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
    test_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists("data/ag_news_train.csv"):
        print("Downloading training data...")
        urllib.request.urlretrieve(train_url, "data/ag_news_train.csv")
    
    if not os.path.exists("data/ag_news_test.csv"):
        print("Downloading test data...")
        urllib.request.urlretrieve(test_url, "data/ag_news_test.csv")
    
    return "data/ag_news_train.csv", "data/ag_news_test.csv"

def load_ag_news(filepath, categories={1: 0, 2: 1}, max_per_class=500):
    """
    Load AG News data. 
    Categories: 1=World/Politics->0, 2=Sports->1
    """
    data = {0: [], 1: []}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                class_idx = int(row[0])
                if class_idx in categories:
                    text = f"{row[1]} {row[2]}"
                    label = categories[class_idx]
                    if len(data[label]) < max_per_class:
                        data[label].append(text)
    
    # Combine into X, y
    X = data[0] + data[1]
    y = [0] * len(data[0]) + [1] * len(data[1])
    
    # Shuffle together
    combined = list(zip(X, y))
    random.seed(42)
    random.shuffle(combined)
    X, y = zip(*combined)
    
    return list(X), list(y)

# ============================================================================
# STEP 2: TEXT PREPROCESSING
# ============================================================================

def preprocess(text):
    """Lowercase and basic cleaning."""
    text = text.lower()
    # Keep only alphanumeric and spaces
    cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    return cleaned.split()

def get_ngrams(tokens, n):
    """Generate n-grams from token list."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = '_'.join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

# ============================================================================
# STEP 3: FEATURE EXTRACTION
# ============================================================================

class BagOfWords:
    """Bag of Words vectorizer."""
    
    def __init__(self, max_features=5000, ngram_range=(1, 1)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary = {}
    
    def _get_all_ngrams(self, tokens):
        """Get all n-grams in the specified range."""
        all_ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if n == 1:
                all_ngrams.extend(tokens)
            else:
                all_ngrams.extend(get_ngrams(tokens, n))
        return all_ngrams
    
    def fit(self, documents):
        """Build vocabulary from documents."""
        word_counts = Counter()
        
        for doc in documents:
            tokens = preprocess(doc)
            ngrams = self._get_all_ngrams(tokens)
            word_counts.update(set(ngrams))  # Count document frequency
        
        # Select top features
        most_common = word_counts.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
        
        return self
    
    def transform(self, documents):
        """Convert documents to BoW vectors."""
        vectors = []
        
        for doc in documents:
            tokens = preprocess(doc)
            ngrams = self._get_all_ngrams(tokens)
            ngram_counts = Counter(ngrams)
            
            vector = [0] * len(self.vocabulary)
            for word, idx in self.vocabulary.items():
                if word in ngram_counts:
                    vector[idx] = ngram_counts[word]
            
            vectors.append(vector)
        
        return vectors
    
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


class TfIdfVectorizer:
    """TF-IDF vectorizer."""
    
    def __init__(self, max_features=5000, ngram_range=(1, 1)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary = {}
        self.idf = {}
    
    def _get_all_ngrams(self, tokens):
        all_ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if n == 1:
                all_ngrams.extend(tokens)
            else:
                all_ngrams.extend(get_ngrams(tokens, n))
        return all_ngrams
    
    def fit(self, documents):
        """Build vocabulary and compute IDF."""
        doc_freq = Counter()
        n_docs = len(documents)
        
        for doc in documents:
            tokens = preprocess(doc)
            ngrams = self._get_all_ngrams(tokens)
            doc_freq.update(set(ngrams))
        
        # Select top features
        most_common = doc_freq.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
        
        # Compute IDF: log(N / df)
        for word, idx in self.vocabulary.items():
            df = doc_freq[word]
            self.idf[word] = math.log(n_docs / (df + 1)) + 1  # Smoothed IDF
        
        return self
    
    def transform(self, documents):
        """Convert documents to TF-IDF vectors."""
        vectors = []
        
        for doc in documents:
            tokens = preprocess(doc)
            ngrams = self._get_all_ngrams(tokens)
            ngram_counts = Counter(ngrams)
            total = len(ngrams) if ngrams else 1
            
            vector = [0.0] * len(self.vocabulary)
            for word, idx in self.vocabulary.items():
                if word in ngram_counts:
                    tf = ngram_counts[word] / total
                    vector[idx] = tf * self.idf[word]
            
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vector)) or 1
            vector = [v / norm for v in vector]
            
            vectors.append(vector)
        
        return vectors
    
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

# ============================================================================
# STEP 4: ML CLASSIFIERS
# ============================================================================

class NaiveBayesClassifier:
    """Multinomial Naive Bayes with Laplace smoothing."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = []
    
    def fit(self, X, y):
        self.classes = list(set(y))
        n_samples = len(y)
        n_features = len(X[0])
        
        for c in self.classes:
            # Get samples for this class
            class_samples = [X[i] for i in range(n_samples) if y[i] == c]
            n_class = len(class_samples)
            
            # Prior probability
            self.class_priors[c] = n_class / n_samples
            
            # Sum features for this class
            feature_sums = [0] * n_features
            for sample in class_samples:
                for i in range(n_features):
                    feature_sums[i] += sample[i]
            
            total = sum(feature_sums)
            
            # Feature probabilities with Laplace smoothing
            self.feature_probs[c] = [
                (feature_sums[i] + self.alpha) / (total + self.alpha * n_features)
                for i in range(n_features)
            ]
        
        return self
    
    def predict(self, X):
        predictions = []
        
        for sample in X:
            best_class = None
            best_score = float('-inf')
            
            for c in self.classes:
                # Log probability to avoid underflow
                score = math.log(self.class_priors[c])
                
                for i, val in enumerate(sample):
                    if val > 0:
                        score += val * math.log(self.feature_probs[c][i])
                
                if score > best_score:
                    best_score = score
                    best_class = c
            
            predictions.append(best_class)
        
        return predictions


class LogisticRegression:
    """Logistic Regression with gradient descent."""
    
    def __init__(self, lr=0.1, epochs=100, regularization=0.01):
        self.lr = lr
        self.epochs = epochs
        self.reg = regularization
        self.weights = None
        self.bias = 0
    
    def _sigmoid(self, z):
        # Clip to prevent overflow
        z = max(-500, min(500, z))
        return 1 / (1 + math.exp(-z))
    
    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        
        # Initialize weights
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        for epoch in range(self.epochs):
            # Gradient descent
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                # Forward pass
                z = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
                pred = self._sigmoid(z)
                
                # Gradient
                error = pred - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            
            # Update weights with regularization
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n_samples + self.reg * self.weights[j])
            self.bias -= self.lr * db / n_samples
        
        return self
    
    def predict(self, X):
        predictions = []
        n_features = len(X[0])
        
        for sample in X:
            z = sum(sample[j] * self.weights[j] for j in range(n_features)) + self.bias
            prob = self._sigmoid(z)
            predictions.append(1 if prob >= 0.5 else 0)
        
        return predictions


class SVM:
    """Support Vector Machine with SGD (linear kernel)."""
    
    def __init__(self, lr=0.001, epochs=100, C=1.0):
        self.lr = lr
        self.epochs = epochs
        self.C = C  # Regularization parameter
        self.weights = None
        self.bias = 0
    
    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        
        # Convert labels to -1, 1
        y_svm = [1 if label == 1 else -1 for label in y]
        
        # Initialize weights
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        for epoch in range(self.epochs):
            for i in range(n_samples):
                # Compute margin
                z = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
                
                if y_svm[i] * z < 1:
                    # Misclassified or within margin
                    for j in range(n_features):
                        self.weights[j] -= self.lr * (2 * (1/self.epochs) * self.weights[j] - self.C * y_svm[i] * X[i][j])
                    self.bias -= self.lr * (-self.C * y_svm[i])
                else:
                    # Correctly classified
                    for j in range(n_features):
                        self.weights[j] -= self.lr * (2 * (1/self.epochs) * self.weights[j])
        
        return self
    
    def predict(self, X):
        predictions = []
        n_features = len(X[0])
        
        for sample in X:
            z = sum(sample[j] * self.weights[j] for j in range(n_features)) + self.bias
            predictions.append(1 if z >= 0 else 0)
        
        return predictions

# ============================================================================
# STEP 5: EVALUATION METRICS
# ============================================================================

def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, F1."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    accuracy = (tp + tn) / len(y_true) if y_true else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    }

def print_results(name, metrics):
    """Pretty print results."""
    print(f"\n{name}")
    print("-" * 40)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    cm = metrics['confusion_matrix']
    print(f"  Confusion Matrix:")
    print(f"    TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}")

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    print("=" * 70)
    print("   SPORTS vs POLITICS CLASSIFIER")
    print("   Comparing: Naive Bayes, Logistic Regression, SVM")
    print("   Features: Bag of Words, TF-IDF, N-grams")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading AG News dataset...")
    train_path, test_path = download_ag_news()
    
    X_train, y_train = load_ag_news(train_path, max_per_class=800)
    X_test, y_test = load_ag_news(test_path, max_per_class=200)
    
    print(f"    Training samples: {len(X_train)} (Politics: {y_train.count(0)}, Sports: {y_train.count(1)})")
    print(f"    Test samples: {len(X_test)} (Politics: {y_test.count(0)}, Sports: {y_test.count(1)})")
    
    # Store all results for comparison
    results = []
    
    # ============ EXPERIMENT 1: Bag of Words ============
    print("\n[2] Feature Extraction: Bag of Words (Unigrams)")
    bow = BagOfWords(max_features=3000, ngram_range=(1, 1))
    X_train_bow = bow.fit_transform(X_train)
    X_test_bow = bow.transform(X_test)
    print(f"    Vocabulary size: {len(bow.vocabulary)}")
    
    # Naive Bayes + BoW
    print("\n[3] Training classifiers on BoW features...")
    
    nb = NaiveBayesClassifier(alpha=1.0)
    nb.fit(X_train_bow, y_train)
    y_pred = nb.predict(X_test_bow)
    metrics = compute_metrics(y_test, y_pred)
    print_results("Naive Bayes + Bag of Words", metrics)
    results.append(("Naive Bayes", "BoW", metrics))
    
    # Logistic Regression + BoW
    lr = LogisticRegression(lr=0.5, epochs=100)
    lr.fit(X_train_bow, y_train)
    y_pred = lr.predict(X_test_bow)
    metrics = compute_metrics(y_test, y_pred)
    print_results("Logistic Regression + Bag of Words", metrics)
    results.append(("Logistic Regression", "BoW", metrics))
    
    # SVM + BoW
    svm = SVM(lr=0.001, epochs=100, C=1.0)
    svm.fit(X_train_bow, y_train)
    y_pred = svm.predict(X_test_bow)
    metrics = compute_metrics(y_test, y_pred)
    print_results("SVM + Bag of Words", metrics)
    results.append(("SVM", "BoW", metrics))
    
    # ============ EXPERIMENT 2: TF-IDF ============
    print("\n[4] Feature Extraction: TF-IDF")
    tfidf = TfIdfVectorizer(max_features=3000, ngram_range=(1, 1))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Naive Bayes + TF-IDF
    nb2 = NaiveBayesClassifier(alpha=1.0)
    nb2.fit(X_train_tfidf, y_train)
    y_pred = nb2.predict(X_test_tfidf)
    metrics = compute_metrics(y_test, y_pred)
    print_results("Naive Bayes + TF-IDF", metrics)
    results.append(("Naive Bayes", "TF-IDF", metrics))
    
    # Logistic Regression + TF-IDF
    lr2 = LogisticRegression(lr=1.0, epochs=100)
    lr2.fit(X_train_tfidf, y_train)
    y_pred = lr2.predict(X_test_tfidf)
    metrics = compute_metrics(y_test, y_pred)
    print_results("Logistic Regression + TF-IDF", metrics)
    results.append(("Logistic Regression", "TF-IDF", metrics))
    
    # SVM + TF-IDF
    svm2 = SVM(lr=0.01, epochs=100, C=1.0)
    svm2.fit(X_train_tfidf, y_train)
    y_pred = svm2.predict(X_test_tfidf)
    metrics = compute_metrics(y_test, y_pred)
    print_results("SVM + TF-IDF", metrics)
    results.append(("SVM", "TF-IDF", metrics))
    
    # ============ EXPERIMENT 3: Bigrams ============
    print("\n[5] Feature Extraction: Bigrams (TF-IDF)")
    tfidf_bi = TfIdfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_bi = tfidf_bi.fit_transform(X_train)
    X_test_bi = tfidf_bi.transform(X_test)
    print(f"    Vocabulary size (with bigrams): {len(tfidf_bi.vocabulary)}")
    
    # Naive Bayes + Bigrams
    nb3 = NaiveBayesClassifier(alpha=1.0)
    nb3.fit(X_train_bi, y_train)
    y_pred = nb3.predict(X_test_bi)
    metrics = compute_metrics(y_test, y_pred)
    print_results("Naive Bayes + Bigrams TF-IDF", metrics)
    results.append(("Naive Bayes", "Bigrams", metrics))
    
    # Logistic Regression + Bigrams
    lr3 = LogisticRegression(lr=1.0, epochs=100)
    lr3.fit(X_train_bi, y_train)
    y_pred = lr3.predict(X_test_bi)
    metrics = compute_metrics(y_test, y_pred)
    print_results("Logistic Regression + Bigrams TF-IDF", metrics)
    results.append(("Logistic Regression", "Bigrams", metrics))
    
    # SVM + Bigrams
    svm3 = SVM(lr=0.01, epochs=100, C=1.0)
    svm3.fit(X_train_bi, y_train)
    y_pred = svm3.predict(X_test_bi)
    metrics = compute_metrics(y_test, y_pred)
    print_results("SVM + Bigrams TF-IDF", metrics)
    results.append(("SVM", "Bigrams", metrics))
    
    # ============ SUMMARY TABLE ============
    print("\n" + "=" * 70)
    print("   RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Classifier':<22} {'Features':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 70)
    
    for clf, feat, m in results:
        print(f"{clf:<22} {feat:<12} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f}")
    
    # Find best
    best = max(results, key=lambda x: x[2]['f1'])
    print(f"\nBest Model: {best[0]} + {best[1]} (F1: {best[2]['f1']:.4f})")
    
    # Return best model for interactive use
    return tfidf, lr2  # Return TF-IDF vectorizer and best model


def interactive_mode(vectorizer, model):
    """Interactive prediction mode."""
    print("\n" + "=" * 70)
    print("   INTERACTIVE CLASSIFICATION")
    print("   Enter text to classify as Sports or Politics")
    print("   Type 'quit' to exit")
    print("=" * 70)
    
    labels = {0: "POLITICS", 1: "SPORTS"}
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not text:
            continue
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # Transform and predict
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        
        print(f"Prediction: {labels[prediction]}")


def main():
    vectorizer, model = run_experiment()
    interactive_mode(vectorizer, model)


if __name__ == "__main__":
    main()
