# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import time

# path = r"C:\Users\vinicius_vieira\OneDrive - Sicredi\Residência IA\Códigos\Inteligência Artificial\Atividades Avaliativas\3_ML Supervisionado\data\dataset.csv"
# df = pd.read_csv(path)

# path_to_save = r"C:\Users\vinicius_vieira\OneDrive - Sicredi\Residência IA\Códigos\Inteligência Artificial\Atividades Avaliativas\3_ML Supervisionado\data\dataset.parquet"
# df.to_parquet(path_to_save)

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
    "Árvore de Decisão": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "Regressão Logística": LogisticRegression(),
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
# Criar tabela comparativa
df_resultados = pd.DataFrame(resultados)

df_resultados.to_excel(
    r"C:\Users\vinicius_vieira\OneDrive - Sicredi\Residência IA\Códigos\Inteligência Artificial\Atividades Avaliativas\3_ML Supervisionado\results\models_results.xlsx",
    index=False,
)

# Visualização da comparação do desempenho dos modelos
df_resultados.set_index("Modelo")[
    ["Acurácia", "F1-Score", "Revocação", "Precisão"]
].plot(kind="bar", figsize=(12, 6))
plt.title("Comparação de Desempenho entre Modelos")
plt.ylabel("Pontuação")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualização da comparação do tempo entre os modelos
df_resultados.set_index("Modelo")[
    [
        "Tempo de treinamento",
        "Tempo de teste",
    ]
].plot(kind="bar", figsize=(12, 6))
plt.title("Comparação de Tempos entre Modelos")
plt.ylabel("Tempo")
plt.ylim(-5, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Resultado do desempenho dos modelos
df_resultados.head()
