import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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

'''
# Testando diferentes profundidades
for depth in [2, 3, 5, 10, 11, 12, 13, 14, 15, None]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test)) 
    print(f"Profundidade: {depth}, Acurácia Treino: {train_acc:.3f}, Acurácia Teste: {test_acc:.3f}")

'''
# Criando e treinando o modelo de Árvore de Decisão
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Fazendo previsões
y_pred = clf.predict(X_test)

# Avaliação do modelo
print("Acurácia da Árvore de Decisão:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Criando e exibindo a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:\n", conf_matrix)

# Plotando a matriz de confusão
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Árvore de Decisão - Matriz de Confusão")
plt.show()

# Visualizando a Árvore de Decisão
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=df.columns[1:-1], class_names=label_encoder.classes_, filled=True)
plt.title("Árvore de Decisão")
plt.show()



