import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# Carregar os dados
file_path = 'missao.xlsx'
data = pd.read_excel(file_path, sheet_name='Aba 1')  # Ajuste o nome da aba conforme necessário

# Limpeza e preparação de dados
data['Valor'] = pd.to_numeric(data['Valor'], errors='coerce')
data.dropna(subset=['Valor'], inplace=True)
data['Hora'] = pd.to_datetime(data['Hora'], format='%H:%M:%S').dt.hour
data['CBK'] = data['CBK'].map({'Sim': 1, 'Não': 0})
data.drop_duplicates(inplace=True)

# Verificar se há pelo menos duas classes distintas
if data['CBK'].nunique() < 2:
    raise ValueError("Data must contain at least two classes for classification.")

# Mostrar a proporção das classes
class_counts = data['CBK'].value_counts()
class_counts.plot(kind='bar')
plt.title('Proporção das Classes')
plt.xlabel('Classe')
plt.ylabel('Contagem')
plt.xticks(ticks=[0, 1], labels=['Não (0)', 'Sim (1)'])
plt.show()

# Separar os dados em características (X) e alvo (y)
X = data[['Valor', 'Hora']]
y = data['CBK']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Aplicar Random Under Sampling nos dados de treino
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

# Mostrar a proporção das classes após o undersampling
undersampled_class_counts = pd.Series(y_train_res).value_counts()
undersampled_class_counts.plot(kind='bar')
plt.title('Proporção das Classes Após Undersampling')
plt.xlabel('Classe')
plt.ylabel('Contagem')
plt.xticks(ticks=[0, 1], labels=['Não (0)', 'Sim (1)'])
plt.show()

# Treinar o modelo XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_res, y_train_res)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Calculando métricas
auc_score = roc_auc_score(y_test, probabilities)
conf_matrix = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions)

# Exibindo os resultados
print(f"Model: XGBoost")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)
print(f"ROC AUC Score: {auc_score}\n")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, probabilities)
plt.plot(fpr, tpr, linestyle='--', label=f"XGBoost (AUC = {auc_score:.2f})")

# Adicionando detalhes ao gráfico ROC
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()

# Prever CBK para Aba 2
data_aba2 = pd.read_excel(file_path, sheet_name='Aba 2')  # Ajuste o nome da aba conforme necessário
data_aba2['Valor'] = pd.to_numeric(data_aba2['Valor'], errors='coerce')
data_aba2.dropna(subset=['Valor'], inplace=True)
data_aba2['Hora'] = pd.to_datetime(data_aba2['Hora'], format='%H:%M:%S').dt.hour
data_aba2['CBK_Predicted'] = model.predict(data_aba2[['Valor', 'Hora']])
data_aba2['CBK_Predicted'] = data_aba2['CBK_Predicted'].map({1: 'Sim', 0: 'Não'})  # Converter predições para 'Sim' ou 'Não'

# Contagem de 'Sim' e 'Não'
count_sim = data_aba2['CBK_Predicted'].value_counts().get('Sim', 0)
count_nao = data_aba2['CBK_Predicted'].value_counts().get('Não', 0)
print(f"Contagem de 'Sim': {count_sim}")
print(f"Contagem de 'Não': {count_nao}")

# Salvar os resultados em um novo arquivo Excel
data_aba2.to_excel('missao_resultados.xlsx', index=False)
