# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Leitura dos dados
# Carregando dataset a partir de um arquivo CSV

path = r"C:\Users\vinicius_vieira\OneDrive - Sicredi\Residência IA\Códigos\Inteligência Artificial\Aulas\data\diabetes.csv"
df = pd.read_csv(path)
# Separando as features (X) e o rótulo (y)
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
# Convertendo os rótulos para valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Normalizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=62
)
# Dicionário de modelos
modelos = {
    "KNN": KNeighborsClassifier(),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "MLP": MLPClassifier(random_state=42, max_iter=1000),
    "SVM": SVC(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
}
# Tabela de resultados
resultados = []
for nome, modelo in modelos.items():
    # inicio do tempo treinamento
    modelo.fit(X_train, y_train)
    # fim do tempo treinamento
    # Apresenta o tempo ?? Quanto tempo demorou?

    # inicio do tempo teste
    y_pred = modelo.predict(X_test)
    # fim do tempo teste
    # Apresenta o tempo ?? Quanto tempo demorou?

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
        }
    )
# Criar tabela comparativa
df_resultados = pd.DataFrame(resultados)
print(df_resultados)

# ========================
# Visualização Opcional (Bar Plot)
# ========================
df_resultados.set_index("Modelo")[
    ["Acurácia", "F1-Score", "Revocação", "Precisão"]
].plot(kind="bar", figsize=(12, 6))
plt.title("Comparação de Desempenho entre Modelos")
plt.ylabel("Pontuação")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
