import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape dos dados:", X.shape)
print("Primeiras linhas:")
print(pd.DataFrame(X, columns=data.feature_names).head())

mlflow.set_tracking_uri("file:./mlruns")   # precisa ter para não dar o erro de não encontrar o experimento: mlflow.exceptions.MlflowException: Could not find experiment with ID 0
mlflow.set_experiment("experiment_mlops_ead_Main.py") # procura por esse, se não existir, ele cria

# mlflow.start_run()
#
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
#
# print("Coeficientes:", model.coef_)
# print("Intercepto:", model.intercept_)
#
#
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
#
# print("Previsões:", y_pred)
# print("Reais:", y_test)
# print("Acurácia:", accuracy)
#
#
# mlflow.log_param("max_iter", model.max_iter)
# mlflow.log_metric("accuracy", accuracy)
# mlflow.sklearn.log_model(model, "model")#2025/08/22 16:15:18 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
# # mlflow.sklearn.log_model(model, name="model") # versão mais atual do mlflow
#
# mlflow.end_run()

# (.venv) PS C:\Users\Raphael\IdeaProjects\DesafioML> mlflow ui
# INFO:     Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)


print("-------------------------------")

# Inicializa o mlflow
mlflow.start_run()

# Define o modelo
clf = RandomForestClassifier(n_estimators=100) #uma árvore de decisão aleatória com 100 galhos

# Treina o modelo
clf.fit(X_train, y_train)

# Loga as métricas do modelo
mlflow.log_metric("accuracy", clf.score(X_test, y_test))

# Salva o modelo com o mlflow
mlflow.sklearn.log_model(clf, "random-forest-model")

# Finaliza o modelo
mlflow.end_run()

# (.venv) PS C:\Users\Raphael\IdeaProjects\DesafioML> mlflow ui
# INFO:     Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)