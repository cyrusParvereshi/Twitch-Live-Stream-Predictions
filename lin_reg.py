
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,  r2_score

def do_lin_reg():
	lr = LinearRegression()
	lr.fit(X_train_selected, y_train_log)
	y_pred_log = lr.predict(X_test_selected)
	y_pred = np.expm1(y_pred_log) #undo the log so we compare the raw values. 
	r2 = r2_score(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	print("R^2:", r2)
	print("MAE:", mae)
