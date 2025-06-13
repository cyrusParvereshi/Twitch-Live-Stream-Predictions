from data_cleaning import data_clean
from feature_selection import get_feature_scores
import pandas as pd
import matplotlib.pyplot as plt
import lin_reg
import rand_forest
import simple_dnn
import predictions

def main():
    X_train_pre, X_test_pre, y_train, y_test, feature_names, full_pipeline,  X_train, X_test = data_clean()
#for the regression models, will log-transform the output variable since it's extremely skewed
    y_train_log = np.log1p(y_train)
    feature_scores = get_feature_scores(X_train_pre, y_train)
    print(feature_scores)
    X_train_df = pd.DataFrame(X_train_pre.toarray(), columns=feature_names) #these are the selected down ones
    X_test_df = pd.DataFrame(X_test_pre.toarray(), columns=feature_names)
    feature_scores_linear_regression = get_feature_scores(X_train_df, y_train, k=10, threshold=2.0)
    print(feature_scores_linear_regression)
    feature_scores_linear_regression.sort_values(ascending=False).head(10).plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.xlabel("Feature Score")
    plt.title("Top 10 Features")
    plt.show()
    selected_features = feature_scores_linear_regression.sort_values(ascending=False).index[:10]
    X_train_selected = X_train_df[selected_features]
    X_test_selected = X_test_df[selected_features]


# choose log(viewer_count) as y target var
# after predictions, convert them back to non-log: predicted_viewers = np.expm1(predicted_log_viewers)


if __name__ == "__main__":
    main()
