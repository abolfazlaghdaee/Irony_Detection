from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_embeddings(X_train, X_test, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf
