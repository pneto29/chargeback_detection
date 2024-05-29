import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
file_path = 'missao.xlsx'
data = pd.read_excel(file_path, sheet_name='Aba 1')  # Ajuste o nome da aba conforme necessário

# Converter a coluna 'Valor' para numérico, ignorando erros
data['Valor'] = pd.to_numeric(data['Valor'], errors='coerce')
data.dropna(subset=['Valor'], inplace=True)  # Remover linhas com valores nulos

# Converter 'Hora' para tipo adequado
data['Hora'] = pd.to_datetime(data['Hora'], format='%H:%M:%S').dt.hour

# Distribuição dos valores das transações
plt.figure(figsize=(10, 6))
sns.histplot(data['Valor'], bins=50, kde=True, color='skyblue')
plt.title('Distribuição dos Valores das Transações')
plt.xlabel('Valor da Transação (R$)')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()

# Frequência de transações por hora do dia
hourly_counts = data['Hora'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.barplot(x=hourly_counts.index, y=hourly_counts, palette='spring')
plt.title('Frequência de Transações por Hora do Dia')
plt.xlabel('Hora do Dia')
plt.ylabel('Número de Transações')
plt.xticks(hourly_counts.index)
plt.grid(True)
plt.show()

# Incidência de estornos (chargebacks)
if 'CBK' in data.columns:
    chargeback_counts = data['CBK'].value_counts(normalize=True) * 100
    plt.figure(figsize=(7, 7))
    chargeback_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=['salmon', 'lightblue'])
    plt.title('Incidência de Estornos (Chargebacks)')
    plt.ylabel('')
    plt.show()

# Análise adicional para responder às questões específicas
# 1. Caracterização das transações deste cliente
print("Estatísticas descritivas do valor das transações:")
print(data['Valor'].describe())

# 2. Perfil das transações que retornaram chargeback
if 'CBK' in data.columns:
    chargeback_data = data[data['CBK'] == 'Sim']
    print("\nPerfil das transações com chargeback:")
    print(chargeback_data.describe())
    plt.figure(figsize=(10, 6))
    sns.histplot(chargeback_data['Valor'], bins=30, kde=True, color='red')
    plt.title('Distribuição dos Valores das Transações com Chargeback')
    plt.xlabel('Valor da Transação (R$)')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()
