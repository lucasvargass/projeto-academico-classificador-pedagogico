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

# mostrar classificação
entrada_horas_suporte_por_semana = int(input('Digite horas de suporte por semana (20h semanais ou 40h semanais): '))

entrada_desempenho_academico = int(input('Digite desempenho academico (45 a 85): '))

entrada_turno_academico = int(input('Digite o turno academico (1 = Manhã ou 2 = Manhã e Tarde): '))

entrada_plano_individualizado = int(input('Possui plano individualizado (Sim = 0 / Não = 1): '))
if entrada_plano_individualizado == 1:
    plano_individualizado_Sim = 1
    plano_individualizado_Nao = 0
else:
    plano_individualizado_Sim = 0
    plano_individualizado_Nao = 1

df_para_classificar = pd.DataFrame(
      [[entrada_horas_suporte_por_semana, entrada_desempenho_academico, entrada_turno_academico, plano_individualizado_Nao, plano_individualizado_Sim]], columns=
       ['horas_suporte_por_semana', 'desempenho_academico', 'turno_academico', 'plano_individualizado_Não', 'plano_individualizado_Sim'])
y_prediction = clf.predict(df_para_classificar)
print("O Tratamento do aluno vai ser: ", y_prediction)

# arvore de decisão
data_feature_names = X.columns
data_class_names = y.unique()
tree.plot_tree(clf, feature_names = data_feature_names,
               class_names = data_class_names, filled = True)
plt.show()