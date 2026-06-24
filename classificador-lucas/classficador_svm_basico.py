import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('educacao_especial_sao_goncalo_registros - educacao_especial_sao_goncalo_registros.csv')

# exclusão de dados ineficientes
del df['estudante_id']
del df['estudante_nome']
del df['idade']


# reformatação e limpeza
df['sexo'].unique()

df = pd.get_dummies(df, columns = ['sexo'])
df['sexo_Feminino'] = df['sexo_Feminino'].apply(lambda x: 1 if x == True else 0)
df['sexo_Masculino'] = df['sexo_Masculino'].apply(lambda x: 1 if x == True else 0)

df['escola'].unique()

df = pd.get_dummies(df, columns = ['escola'])
df['escola_Escola Municipal Darcy Ribeiro'] = df['escola_Escola Municipal Darcy Ribeiro'].apply(lambda x: 1 if x == True else 0)
df['escola_Escola Municipal JOAO CABRAL DE MELO NETO'] = df['escola_Escola Municipal JOAO CABRAL DE MELO NETO'].apply(lambda x: 1 if x == True else 0)
df['escola_Colégio Municipal Estephânia de Carvalho'] = df['escola_Colégio Municipal Estephânia de Carvalho'].apply(lambda x: 1 if x == True else 0)

df['categoria_diagnostico'].unique()

df = pd.get_dummies(df, columns = ['categoria_diagnostico'])
df['categoria_diagnostico_Deficiência Física'] = df['categoria_diagnostico_Deficiência Física'].apply(lambda x: 1 if x == True else 0)
df['categoria_diagnostico_Transtorno de Ansiedade'] = df['categoria_diagnostico_Transtorno de Ansiedade'].apply(lambda x: 1 if x == True else 0)
df['categoria_diagnostico_Dislexia'] = df['categoria_diagnostico_Dislexia'].apply(lambda x: 1 if x == True else 0)
df['categoria_diagnostico_TDAH'] = df['categoria_diagnostico_TDAH'].apply(lambda x: 1 if x == True else 0)
df['categoria_diagnostico_TEA (Transtorno do Espectro Autista)'] = df['categoria_diagnostico_TEA (Transtorno do Espectro Autista)'].apply(lambda x: 1 if x == True else 0)
df['categoria_diagnostico_Deficiência Intelectual'] = df['categoria_diagnostico_Deficiência Intelectual'].apply(lambda x: 1 if x == True else 0)

df['plano_individualizado'].unique()

df = pd.get_dummies(df, columns = ['plano_individualizado'])
df['plano_individualizado_Sim'] = df['plano_individualizado_Sim'].apply(lambda x: 1 if x == True else 0)
df['plano_individualizado_Não'] = df['plano_individualizado_Não'].apply(lambda x: 1 if x == True else 0)

df.head()

# definição dos dados de entrada X, de saída y, de treinamento e de teste
X = df.drop('nivel_suporte', axis=1)
y = df['nivel_suporte']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 1)

# aprendizado COM SVM BASICO
clf = svm.SVC()
clf.fit(X_train,y_train)

# mostrar desempenho
y_prediction = clf.predict(X_test)
print("Assertividade:", accuracy_score(y_test, y_prediction))