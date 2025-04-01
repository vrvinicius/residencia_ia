import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando dataset a partir de um arquivo CSV
df = pd.read_csv("iris.csv")

# Separando as features (X) e o rótulo (y)
X = df.iloc[:, 1:-1]  # Remove a primeira coluna (IND) e a última (Y)
y = df.iloc[:, -1]  # Última coluna contém os rótulos

# Convertendo os rótulos para valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalizando os dados
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo de rede neural Backpropagation usando MLPClassifier
# hidden_layer_sizes=(10, 8) --> número de camadas ocultas e a quantidade de neurônios em cada camada.
# Camada Oculta 1: 10 neurônios.
# Camada Oculta 2: 8 neurônios.
# Entrada →  Camada Oculta 1 → Camada Oculta 2 → Saída
#    4    →        10        →       8         →   3 (Iris)
# Exemplos:
# hidden_layer_sizes=(10)  # Apenas 1 camada oculta com 10 neurônios
# hidden_layer_sizes=(20, 10, 5)  # 3 camadas ocultas com 20, 10 e 5 neurônios
# hidden_layer_sizes=() estiver vazio, o modelo se comporta como um Perceptron Simples, ou seja, sem camadas ocultas.
#
# activation='relu' --> Define a função de ativação usada nos neurônios das camadas ocultas. A ativação introduz não linearidade, permitindo que a rede aprenda padrões complexos.
# 'identity' --> f(x)=x, não altera, útil para regressão linear.
# 'logistic' --> Função Sigmóide, usada para classificação binária
# 'relu'     --> Melhor para redes profundas, evita saturação.
# 'tanh'     --> Similar à sigmóide, mas com saída entre -1 e 1.
#
# solver='adam' --> Define o algoritmo de otimização usado para atualizar os pesos da rede neural durante o treinamento (Backpropagation).
# 
# 'lbfgs'	 --> Método de otimização baseado em segunda derivada. Funciona bem para problemas pequenos.
# 'sgd'	     --> Gradiente Descendente Estocástico. Pode ser mais lento e instável sem ajuste fino.
# 'adam'	 --> Algoritmo adaptativo eficiente, funciona bem na maioria dos casos.
#
# max_iter=500 --> Define quantas épocas (iterações) a rede neural será treinada. Cada época representa uma passagem completa pelo conjunto de treinamento.
#
# random_state=42 --> Define um valor fixo para o gerador de números aleatórios do Scikit-learn. Garante reprodutibilidade, ou seja, os mesmos resultados serão obtidos em cada execução.

mlp = MLPClassifier(hidden_layer_sizes=(20, 12), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Treinando o modelo
mlp.fit(X_train, y_train)

# Fazendo previsões
y_pred = mlp.predict(X_test)

# Avaliação do modelo
print("Acurácia da Rede Neural MLP:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Criando e exibindo a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:\n", conf_matrix)

# Plotando a matriz de confusão
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Rede Neural MLP - Matriz de Confusão")
plt.show()
