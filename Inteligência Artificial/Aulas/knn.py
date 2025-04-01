import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# Carregar os dados
df = pd.read_csv("iris.csv")

# Separando as features (X) e o rótulo (y)
X = df.iloc[:, 1:-1]  
y = df.iloc[:, -1]  


# Convertendo os rótulos para valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# Normalizando os dados
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=62)

# Criando e treinando o modelo K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Fazendo previsões
y_pred = knn.predict(X_test)


# Avaliação do modelo
print("Acurácia do K-NN:", accuracy_score(y_test, y_pred))

# Criando e exibindo a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:\n", conf_matrix)

# Plotando a matriz de confusão
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("K-NN - Matriz de Confusão")
plt.show()




