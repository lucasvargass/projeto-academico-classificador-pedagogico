import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('educacao_especial_sao_goncalo_registros - educacao_especial_sao_goncalo_registros.csv')

# exclusão de dados ineficientes
del df['estudante_id']
del df['estudante_nome']
del df['idade']
del df['sexo']
del df['escola']
del df['categoria_diagnostico']

#limpeza de dados
df = pd.get_dummies(df, columns=['plano_individualizado'], dtype=int)

df.info()

# definição dos dados de entrada X, de saída y, de treinamento e de teste
X = df.drop('nivel_suporte', axis=1)
y = df['nivel_suporte']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 1)

# aprendizado
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# mostrar desempenho
y_prediction = clf.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_prediction))