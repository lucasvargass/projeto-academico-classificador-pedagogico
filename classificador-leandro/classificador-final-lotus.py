import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# leitura dos dados
df = pd.read_csv('Pasta Atualizada.csv')


# reformatação de colunas
df['TOMADOR'].unique()


# execução do get_dummies
df = pd.get_dummies(df, columns = ['TOMADOR'],dtype=int)
# df['TOMADOR_TRANSHIP'] = df['TOMADOR_TRANSHIP'].apply(lambda x: 1 if x == True else 0)
# df['TOMADOR_CAMORIM'] = df['TOMADOR_CAMORIM'].apply(lambda x: 1 if x == True else 0)
# df['TOMADOR_BLUE AMAZON'] = df['TOMADOR_BLUE AMAZON'].apply(lambda x: 1 if x == True else 0)


# exclusão de dados ineficientes
del df['DATA DE RECEBIMENTO/ PREVISÃO']
del df['NAVIO']
del df['AGENCIAMENTO']


df.info()


# definição dos dados de entrada X, de saída y, de treinamento e de teste
X = df.drop('EMBARCAÇÃO', axis=1)
y = df['EMBARCAÇÃO']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 1)


# aprendizado
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# mostrar desempenho
y_prediction = clf.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_prediction))


# mostrar classificação
entrada_PAGAMENTO = int(input('Houve pagamento? (1=Pago / 0=Não pago): '))


entrada_ACOPLADO = int(input('É acoplado? (1=Sim / 0=Não): '))


entrada_DIAS_ATRACADO = int(input('Dias_atracado (5 ou 7): '))


# variáveis dummy
entrada_TOMADOR_TRANSHIP = 0
entrada_TOMADOR_CAMORIM = 0
entrada_TOMADOR_BLUE_AMAZON = 0


entrada_Tomador = int(input('Tomador (1=TRANSHIP, 2=CAMORIM, 3=BLUE AMAZON): '))


if entrada_Tomador == 1:
    entrada_TOMADOR_TRANSHIP = 1


elif entrada_Tomador == 2:
    entrada_TOMADOR_CAMORIM = 1


elif entrada_Tomador == 3:
    entrada_TOMADOR_BLUE_AMAZON = 1


# dataframe para previsão
df_para_classificar = pd.DataFrame([[ entrada_PAGAMENTO, entrada_ACOPLADO, entrada_DIAS_ATRACADO, entrada_TOMADOR_TRANSHIP, entrada_TOMADOR_CAMORIM, entrada_TOMADOR_BLUE_AMAZON ]],
                                   columns=['PAGAMENTO', 'ACOPLADO', 'DIAS ATRACADO', 'TOMADOR_BLUE AMAZON', 'TOMADOR_CAMORIM', 'TOMADOR_TRANSHIP'])


# previsão
y_prediction = clf.predict(df_para_classificar)
print("Prediction for Decision Tree: ", y_prediction)


# mostrar árvore de decisão
data_feature_names = X.columns
data_class_names = y.unique()
tree.plot_tree(clf, feature_names = data_feature_names,
               class_names= data_class_names, filled = True)
plt.show()