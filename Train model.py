import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pickle
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV
import joblib

def scale_frame(frame):
    df = frame.copy()
    X,y = df.drop(columns = ['Price(euro)']), df['Price(euro)']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1,1))
    return X_scale, Y_scale, power_trans


if __name__ == "__main__":
    df = pd.read_csv("/var/lib/jenkins/workspace/download/cars.csv")
    X,Y, power_trans = scale_frame(df)
# разбиваем на тестовую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                    test_size=0.3,
                                                    random_state=42)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1 ],
      'l1_ratio': [0.001, 0.05, 0.01, 0.2]
 }
with mlflow.start_run():

    lr = SGDRegressor(random_state=42)
    clf = GridSearchCV(lr, params, cv = 5)
    clf.fit(X_train, y_train.reshape(-1))
    best = clf.best_estimator_
    y_pred = best.predict(X_val)
    y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1,1))
    (rmse, mae, r2)  = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)
    alpha = best.alpha
    l1_ratio = best.l1_ratio
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    
    predictions = best.predict(X_train)
    signature = infer_signature(X_train, predictions)
    mlflow.sklearn.log_model(lr, "model", signature=signature)
    with open("lr_cars.pkl", "wb") as file:
        joblib.dump(lr, file)
        
dfruns = mlflow.search_runs()
path2model = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://","") + '/model' #путь до эксперимента с лучшей моделью
print(path2model)
