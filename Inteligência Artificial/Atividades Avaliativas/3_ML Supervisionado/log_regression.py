# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time

path_parquet = r"C:\Users\vinicius_vieira\OneDrive - Sicredi\Residência IA\Códigos\Inteligência Artificial\Atividades Avaliativas\3_ML Supervisionado\data\dataset.parquet"
df = pd.read_parquet(path_parquet)

# Separando as features (X) e o rótulo (y)
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Normalizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=62
)

# Dicionário de modelos
modelos = {
    "Regressão Logística Default": LogisticRegression(),
    "Regressão Logística Mode 1": LogisticRegression(max_iter=200),
    "Regressão Logística Mode 2": LogisticRegression(random_state=42, max_iter=800),
    "Regressão Logística Mode 3": LogisticRegression(
        random_state=42, max_iter=800, tol=1e-3
    ),
    "Regressão Logística Mode 4": LogisticRegression(
        random_state=42, max_iter=1000, tol=1e-2
    ),
}


# Tabela de resultados
resultados = []
for nome, modelo in modelos.items():
    # medindo o tempo de treinamento
    training_start = time.time()
    modelo.fit(X_train, y_train)
    training_end = time.time()
    training_time = training_end - training_start

    # medindo o tempo de teste
    test_start = time.time()
    y_pred = modelo.predict(X_test)
    test_end = time.time()
    test_time = test_end - test_start

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")

    resultados.append(
        {
            "Modelo": nome,
            "Acurácia": round(acc, 4),
            "F1-Score": round(f1, 4),
            "Revocação": round(recall, 4),
            "Precisão": round(precision, 4),
            "Tempo de treinamento": round(training_time, 4),
            "Tempo de teste": round(test_time, 4),
        }
    )
# Criar tabela dos resultados
df_resultados = pd.DataFrame(resultados)

df_resultados.to_excel(
    r"C:\Users\vinicius_vieira\OneDrive - Sicredi\Residência IA\Códigos\Inteligência Artificial\Atividades Avaliativas\3_ML Supervisionado\results\log_regression_result.xlsx",
    index=False,
)

# Resultado do desempenho
df_resultados.head()
