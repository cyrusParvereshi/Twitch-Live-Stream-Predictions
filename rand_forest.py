from sklearn.metrics import mean_absolute_error,  r2_score

def rand_forest():

    param_grid = {
        'regressor__n_estimators': [10, 20],
        'regressor__max_depth': [5, 10],
    }

    rf_model = Pipeline([
        ('preprocessing', full_pipeline), #preprocessing pipeline returned from data_clean()
        ('regressor', RandomForestRegressor())
    ])
    # Fit the model
    grid = GridSearchCV(rf_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=4)
    # Fit on log-transformed target, will use original dfs with all features (not just top k)
    grid.fit(X_train, y_train_log)
    y_pred_log = grid.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    # Evaluate and print results
    mae = mean_absolute_error(y_test, y_pred)
    print("Best parameters:", grid.best_params_)
    print("Random Forest MAE:", mae)