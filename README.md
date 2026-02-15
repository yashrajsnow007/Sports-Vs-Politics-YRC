# Sports-Vs-Politics-YRC
This is a repository created for evaluation of Natural Language Understanding Assignment 1 problem 4 sports-vs-politics-classifier 

A machine learning text classification system that categorizes news articles as **Sports** or **Politics** using traditional ML techniques implemented from scratch.

## Overview

This project implements and compares three machine learning classifiers with three different feature extraction techniques for binary text classification, without using any external ML libraries.

### Classifiers Implemented
- **Naive Bayes** (Multinomial with Laplace smoothing)
- **Logistic Regression** (with gradient descent)
- **Support Vector Machine** (linear kernel with SGD)

### Feature Extraction Methods
- **Bag of Words (BoW)**
- **TF-IDF** (Term Frequency-Inverse Document Frequency)
- **N-grams** (Unigrams + Bigrams)

## Results

| Classifier | Features | Accuracy | Precision | Recall | F1-Score |
|------------|----------|----------|-----------|--------|----------|
| **Naive Bayes** | **BoW** | **0.9400** | **0.9112** | **0.9750** | **0.9420** |
| Logistic Regression | BoW | 0.9100 | 0.8981 | 0.9250 | 0.9113 |
| SVM | BoW | 0.9275 | 0.9091 | 0.9500 | 0.9291 |
| Naive Bayes | TF-IDF | 0.9325 | 0.9023 | 0.9700 | 0.9349 |
| Logistic Regression | TF-IDF | 0.9200 | 0.9000 | 0.9450 | 0.9220 |
| SVM | TF-IDF | 0.9175 | 0.8711 | 0.9800 | 0.9224 |
| Naive Bayes | Bigrams | 0.9275 | 0.8904 | 0.9750 | 0.9308 |
| Logistic Regression | Bigrams | 0.9175 | 0.8957 | 0.9450 | 0.9197 |
| SVM | Bigrams | 0.9175 | 0.8957 | 0.9450 | 0.9197 |

**Best Model:** Naive Bayes + Bag of Words (F1: 0.9420)

## Dataset

- **Source:** AG News Dataset
- **Classes:** Sports (Class 2), Politics/World (Class 1)
- **Training samples:** 1600 (800 per class)
- **Test samples:** 400 (200 per class)

### Dataset Statistics
| Category | Articles | Avg Words | Min Words | Max Words |
|----------|----------|-----------|-----------|-----------|
| Sports | 600 | 37.6 | 8 | 78 |
| Politics | 600 | 38.7 | 11 | 87 |

## Installation & Usage

### Requirements
- Python 3.x
- No external dependencies (uses only standard library)

### Run the Classifier

```bash
python B22AI059_prob4.py
