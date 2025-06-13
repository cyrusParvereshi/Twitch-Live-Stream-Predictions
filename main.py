from data_cleaning import data_clean
from feature_selection import get_feature_scores
import pandas as pd


def main():
    X_train_pre, X_test_pre, y_train, y_test = data_clean()
    feature_scores = get_feature_scores(X_train_pre, y_train)
    print(feature_scores)


# choose log(viewer_count) as y target var
# after predictions, convert them back to non-log: predicted_viewers = np.expm1(predicted_log_viewers)


if __name__ == "__main__":
    main()
