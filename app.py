from flask import Flask, render_template, request
import numpy as np
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer

# Flask app initialization
app = Flask(__name__)

# Preprocessing setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

def run_textrank(weighted_edge, vocab_len, MAX_ITERATIONS=50, d=0.85, threshold=0.0001):
    score = np.ones(vocab_len, dtype=np.float32)
    for _ in range(MAX_ITERATIONS):
        prev_score = np.copy(score)
        for i in range(vocab_len):
            summation = 0
            for j in range(vocab_len):
                if weighted_edge[i][j] != 0:
                    summation += (weighted_edge[i][j] / np.sum(weighted_edge[:, j])) * score[j]
            score[i] = (1 - d) + d * summation
        if np.sum(np.abs(prev_score - score)) <= threshold:
            break
    return score

def extract_keywords(scores, vocabulary, top_n=10):
    sorted_indices = np.argsort(scores)[::-1]
    keywords = [vocabulary[i] for i in sorted_indices[:top_n]]
    return keywords

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['input_text']
        method = request.form['method']
        
        # Preprocess the text
        processed_texts = preprocess(text)
        vocabulary = list(set(processed_texts))
        vocab_len = len(vocabulary)

        # Convert preprocessed text to TF-IDF matrix
        vectorizer = TfidfVectorizer(vocabulary=vocabulary)
        X = vectorizer.fit_transform([' '.join(processed_texts)])
        tfidf_matrix = X.toarray()

        # Initialize weighted edge matrix
        weighted_edge = np.zeros((vocab_len, vocab_len), dtype=np.float32)

        if method == 'tfidf':
            for i in range(vocab_len):
                for j in range(vocab_len):
                    if i != j:
                        weighted_edge[i][j] = tfidf_matrix[0][i] * tfidf_matrix[0][j]
        
        elif method == 'cosine':
            for i in range(vocab_len):
                for j in range(vocab_len):
                    if i != j:
                        dot_product = np.dot(tfidf_matrix[:, i], tfidf_matrix[:, j])
                        norm_i = np.linalg.norm(tfidf_matrix[:, i])
                        norm_j = np.linalg.norm(tfidf_matrix[:, j])
                        if norm_i != 0 and norm_j != 0:
                            weighted_edge[i][j] = dot_product / (norm_i * norm_j)
        
        elif method == 'jaccard':
            def jaccard_similarity(set1, set2):
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                return intersection / union

            for i in range(vocab_len):
                for j in range(vocab_len):
                    if i != j:
                        set_i = set([idx for idx, text in enumerate([processed_texts]) if vocabulary[i] in text])
                        set_j = set([idx for idx, text in enumerate([processed_texts]) if vocabulary[j] in text])
                        weighted_edge[i][j] = jaccard_similarity(set_i, set_j)
        
        elif method == 'cooccurrence':
            window_size = 3
            for window_start in range(len(processed_texts) - window_size + 1):
                window = processed_texts[window_start:window_start + window_size]
                for i in range(window_size):
                    for j in range(window_size):
                        if i != j:
                            index_i = vocabulary.index(window[i])
                            index_j = vocabulary.index(window[j])
                            weighted_edge[index_i][index_j] += 1 / abs(i - j)

        # Apply TextRank
        scores = run_textrank(weighted_edge, vocab_len)
        keywords = extract_keywords(scores, vocabulary)

        return render_template('result.html', keywords=keywords)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
