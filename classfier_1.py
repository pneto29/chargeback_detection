import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

# Carrega os dados de uma planilha Excel
file_path = 'missao.xlsx'
data = pd.read_excel(file_path, sheet_name='Aba 1')

# Converte e limpa dados específicos
data['Valor'] = pd.to_numeric(data['Valor'], errors='coerce')
data.dropna(subset=['Valor'], inplace=True)
data['Hora'] = pd.to_datetime(data['Hora'], format='%H:%M:%S').dt.hour
data['CBK'] = data['CBK'].map({'Sim': 1, 'Não': 0})
data.drop_duplicates(inplace=True)

# Certifica-se de que existem duas classes para a classificação
if data['CBK'].nunique() < 2:
    raise ValueError("Os dados devem conter pelo menos duas classes para classificação.")

# Exibe a proporção inicial das classes
plt.figure(figsize=(8, 4))
sns.countplot(x='CBK', data=data, palette='coolwarm')
plt.title('Distribuição das Classes Antes do Undersampling')
plt.xlabel('Classe')
plt.ylabel('Contagem')
plt.xticks(ticks=[0, 1], labels=['Não (0)', 'Sim (1)'])
plt.show()

# Prepara dados para modelos
X = data[['Valor', 'Hora']]
y = data['CBK']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Split 70/30

# Aplica undersampling para equilibrar as classes
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

# Imprime os nomes das features finais
print("Nomes das features após o tratamento de dados:")
print(X_train_res.columns.tolist())

# Define e configura modelos de classificação
models = {
    'Regressão Logística': LogisticRegression(),
    'Floresta Aleatória': RandomForestClassifier(),
    'Classificador MLP': MLPClassifier(max_iter=500),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0)
}

results = []

# Treina e avalia cada modelo
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, probabilities)
    conf_matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    # Armazena as métricas
    result = {
        'Modelo': name,
        'ROC AUC Score': round(auc_score, 4),
        'Accuracy': round(report['accuracy'], 4),
        'Precision (No)': round(report['0']['precision'], 4),
        'Recall (No)': round(report['0']['recall'], 4),
        'F1-Score (No)': round(report['0']['f1-score'], 4),
        'Precision (Yes)': round(report['1']['precision'], 4),
        'Recall (Yes)': round(report['1']['recall'], 4),
        'F1-Score (Yes)': round(report['1']['f1-score'], 4)
    }
    results.append(result)

    # Exibe a matriz de confusão com métricas em uma figura separada
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 20})
    plt.title(f'Matriz de Confusão: {name}')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.xticks(ticks=[0.5, 1.5], labels=['Não (0)', 'Sim (1)'])
    plt.yticks(ticks=[0.5, 1.5], labels=['Não (0)', 'Sim (1)'], rotation=0)
    plt.show()

    print(f"Modelo: {name}")
    print("Relatório de Classificação:")
    print(classification_report(y_test, predictions, digits=4))
    print(f"Pontuação ROC AUC: {auc_score:.4f}\n")

    # Curva ROC na figura configurada
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.plot(fpr, tpr, linestyle='--', label=f"{name} (AUC = {auc_score:.4f})")

# Adicionando detalhes ao gráfico ROC
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curva ROC')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.legend(loc='best')
plt.show()

# Salva as métricas em um arquivo CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_performance_metrics01.csv', index=False)

# Identifica o classificador com melhor recall para a classe positiva
best_model = max(results, key=lambda x: x['Recall (Yes)'])
print(f"O classificador com melhor recall para a classe positiva é: {best_model['Modelo']} com recall de {best_model['Recall (Yes)']:.4f}")

# Exibe os nomes das variáveis
print("Nomes das variáveis do dataset:")
print(X.columns.tolist())

plt.figure(figsize=(10, 8))

# Treina e avalia cada modelo
for name, model in models.items():
    probabilities = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, probabilities)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.plot(fpr, tpr, linestyle='--', label=f"{name} (AUC = {auc_score:.4f})")

# Adicionando detalhes ao gráfico ROC
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curva ROC')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.legend(loc='best')
plt.show()

# Calcular e imprimir o impacto financeiro para os modelos XGBoost, LightGBM e RandomForest

for name, model in models.items():
    if name in ['XGBoost', 'LightGBM', 'Floresta Aleatória']:
        predictions = model.predict(X_test)
        chargeback_indices = X_test[predictions == 1].index
        financial_impact = data.loc[chargeback_indices, 'Valor'].sum()
        print(f"Impacto financeiro associado aos chargebacks para o modelo {name}: R$ {financial_impact:.2f}")
