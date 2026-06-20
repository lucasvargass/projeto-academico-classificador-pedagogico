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