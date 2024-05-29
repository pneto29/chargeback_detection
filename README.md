Este repositório contém diversos scripts Python para análise e classificação de transações com o objetivo de detectar chargebacks. Abaixo está uma descrição sucinta dos arquivos incluídos e suas funcionalidades.

Arquivos
1. cbk_eda_all.py
Este script realiza uma Análise Exploratória de Dados (EDA) detalhada no conjunto de dados de transações. A análise inclui a limpeza e preparação dos dados, visualizações estatísticas e cálculo de métricas descritivas para entender a distribuição e os padrões dos chargebacks.

Principais Funcionalidades:

Carregamento e limpeza dos dados.
Conversão de valores e extração de características temporais.
Visualização da distribuição dos valores e horários das transações.
Cálculo de estatísticas descritivas (média, mediana, desvio padrão, quartis).
2. client_transaction.py
Este script foca na análise das transações dos clientes, identificando padrões e outliers. Inclui técnicas para a detecção de anomalias e análise dos comportamentos de compra dos clientes.

Principais Funcionalidades:

Carregamento dos dados de transações de clientes.
Análise de padrões de compra.
Detecção de anomalias nas transações.
Visualização das transações por cliente.
3. classfier_1.py
Este script implementa diversos modelos de classificação para a detecção de chargebacks, comparando seu desempenho através de métricas como AUC-ROC e matrizes de confusão. Modelos incluem Regressão Logística, Floresta Aleatória, MLP, XGBoost, LightGBM e CatBoost.

Principais Funcionalidades:

Carregamento e preparação dos dados.
Divisão dos dados em conjuntos de treino e teste.
Balanceamento das classes usando undersampling.
Treinamento de múltiplos modelos de classificação.
Avaliação do desempenho dos modelos usando métricas diversas.
Visualização das matrizes de confusão e curvas ROC.
4. column_save_CBK_with_ensemble.py
Este script aplica modelos de ensemble para prever a ocorrência de chargebacks e salva os resultados em um novo arquivo Excel. Também calcula o impacto financeiro dos chargebacks para diferentes modelos.
