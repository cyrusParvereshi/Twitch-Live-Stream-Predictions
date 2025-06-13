import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression


def get_feature_scores(X, y, k=10, threshold=1.0):
    selector = SelectKBest(score_func=f_regression, k="all")
    selector.fit(X, y)
    feature_scores = pd.Series(selector.scores_, index=X.columns)
    sorted_scores = feature_scores.sort_values(ascending=False)
    if threshold is not None:
        sorted_scores = sorted_scores[sorted_scores > threshold]
    if k is not None:
        sorted_scores = sorted_scores[:k]
    return sorted_scores
