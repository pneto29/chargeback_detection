import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def carregar_dados(filepath):
    # Carregar os dados da primeira aba do arquivo Excel
    data = pd.read_excel(filepath, sheet_name=0)
    # Converter 'Valor' para numérico, tratando não-numéricos como NaN
    data['Valor'] = pd.to_numeric(data['Valor'], errors='coerce')
    # Converter 'Hora' considerando erros de formatação
    data['Hora'] = pd.to_datetime(data['Hora'], format='%H:%M:%S', errors='coerce').dt.time
    return data

def analise_univariada(data):
    # Estatísticas descritivas para 'Valor'
    print(data['Valor'].describe())
    # Frequência de valores para 'CBK'
    print(data['CBK'].value_counts())
    # Contagem de cartões únicos
    print('Número de cartões únicos:', data['Cartão'].nunique())

def analise_bivariada(data):
    # Relacionamento entre 'Valor' e 'CBK'
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='CBK', y='Valor', data=data)
    plt.title('Distribuição dos Valores de Transação por Estorno (CBK)')
    plt.xlabel('Estorno')
    plt.ylabel('Valor da Transação')
    plt.grid(True)
    plt.show()

    # Adicionar uma coluna com o dia da semana
    data['Dia da Semana'] = data['Dia'].dt.dayofweek

    # Agregar transações por dia da semana e por hora
    transacoes_por_dia_semana = data.groupby('Dia da Semana')['Valor'].count()
    transacoes_por_hora = data.groupby(data['Hora'].apply(lambda x: x.hour if pd.notna(x) else None))['Valor'].count()

    # Transações por dia da semana
    plt.figure(figsize=(10, 6))
    sns.barplot(x=transacoes_por_dia_semana.index, y=transacoes_por_dia_semana.values)
    plt.title('Transações por Dia da Semana')
    plt.xlabel('Dia da Semana')
    plt.ylabel('Número de Transações')
    plt.xticks(range(7), ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'])
    plt.grid(True)
    plt.show()

    # Transações por hora
    plt.figure(figsize=(10, 6))
    sns.barplot(x=transacoes_por_hora.index, y=transacoes_por_hora.values)
    plt.title('Transações por Hora do Dia')
    plt.xlabel('Hora do Dia')
    plt.ylabel('Número de Transações')
    plt.grid(True)
    plt.show()

def main():
    filepath = 'missao.xlsx'
    data = carregar_dados(filepath)
    print("Análise Univariada:")
    analise_univariada(data)
    print("Análise Bivariada:")
    analise_bivariada(data)

if __name__ == "__main__":
    main()
