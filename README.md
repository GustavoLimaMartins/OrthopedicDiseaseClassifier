# 🦴 Classificação de Problemas Ortopédicos da Coluna Vertebral (`pipeline.py`)

## 🎯 Objetivo
Este projeto visa classificar problemas ortopédicos da coluna vertebral (Hérnia de Disco, Normal e Espondilolistese) usando um modelo de **K-Nearest Neighbors (KNN)**. O *script* implementa um pipeline completo de Machine Learning, desde a aquisição de dados até a otimização e avaliação do modelo.

## 💾 Dataset: Coluna Vertebral (Vertebra Column)

### Descrição
O *dataset* contém dados sobre problemas ortopédicos na coluna vertebral, diagnosticados no *Centre Médico-Chirurgical de Réadaptation des Massues*, em Lyon, França.
* **ID OpenML**: 1523
* **Entradas**: 310 instâncias anonimizadas.
* **Atributos Preditivos**: 6 atributos biomecânicos.

### Classes
| Diagnóstico | Código | Instâncias Originais |
| :--- | :--- | :--- |
| **Normal** (NO) | 2 | 100 |
| **Hérnia de Disco** (Disk Hernia - DH) | 1 | 60 |
| **Espondilolistese** (Spondylolisthesis - SL) | 3 | 150 |

---

## 🛠️ Detalhamento do Pipeline

### 1. Preparação Inicial de Dados
O *script* carrega o *dataset* do **OpenML** (`data_id=1523`) e o converte para um `pandas.DataFrame`.
* O atributo-alvo (`target`) é mapeado para as *strings* de diagnóstico (`Disk Hernia`, `Normal`, `Spondylolisthesis`).
* **Dimensionalidade**: `(310, 7)` (310 linhas, 6 atributos preditivos + 1 coluna `diagnostic`).
* **Estatísticas**: É calculada a média dos atributos por classe, permitindo uma primeira visualização de como os atributos diferem entre os diagnósticos.

### 2. Análise Exploratória de Dados (EDA)

O *script* gera visualizações importantes para a compreensão dos dados:

* **Comportamento de Pares Ordenados (`sns.pairplot`)**:
    * Visualiza a distribuição de cada atributo e a relação de pares de atributos, coloridos pela variável-alvo (`diagnostic`).
    * **Recurso Visual**: Gráfico de dispersão com histogramas diagonais.
* **Correlações entre Atributos (`sns.heatmap`)**:
    * Identifica a relação linear entre os atributos preditivos. O *script* nota as maiores correlações:
        * `('V1', 'V4')` **(0.81)**
        * `('V1', 'V3')` **(0.72)**
        * `('V1', 'V6')` **(0.64)**
        * `('V1', 'V2')` **(0.63)**
    * **Recurso Visual**: Mapa de calor da matriz de correlação.
* **Histograma dos Dados (`df.hist`)**:
    * Mostra a distribuição de frequência de cada atributo.
    * **Recurso Visual**: 6 histogramas individuais.

### 3. Pré-processamento e Tratamento de Outliers

#### **Padronização da Escala (`StandardScaler`)**
Os dados são padronizados (média 0 e desvio-padrão 1) para garantir que todos os atributos contribuam igualmente para a distância no KNN.

* **Boxplot Antes da Remoção de Outliers**: Visualiza a dispersão dos dados padronizados, mostrando a presença de *outliers*.
    * **Recurso Visual**: Boxplot.

#### **Remoção Coletiva de Outliers**
O *script* implementa uma remoção de *outliers* baseada no **escore Z**.
* **Critério**: Instâncias com valores **abaixo de -2.5** ou **acima de 3** são identificadas e removidas.
* Os índices dos *outliers* são coletados e removidos do *DataFrame* original, resultando em um conjunto de dados mais limpo.

* **Boxplot Após a Remoção de Outliers**: Visualiza a nova distribuição após a limpeza.
    * **Recurso Visual**: Boxplot.

### 4. Divisão e Balanceamento de Dados

#### **Divisão em Treino e Teste (`train_test_split`)**
* Os dados limpos são divididos em conjuntos de **Treino (80%)** e **Teste (20%)**, usando `stratify=y` para garantir que a proporção das classes seja mantida em ambos os conjuntos.

#### **Oversampling com SMOTE**
O *dataset* original é desbalanceado (150 SL, 100 NO, 60 DH). A técnica **SMOTE (Synthetic Minority Over-sampling Technique)** é aplicada *apenas* ao conjunto de treinamento (`X_train_scaled`, `y_train`) para criar instâncias sintéticas das classes minoritárias.
* **Objetivo**: Balancear as classes no treinamento, melhorando a capacidade do modelo de aprender com as classes menos representadas.

### 5. Treinamento e Avaliação do Modelo Base (KNN)

Um modelo **KNeighborsClassifier** inicial é treinado com `n_neighbors=3`.

* **Relatório de Classificação (`classification_report`)**: Fornece métricas detalhadas (Precisão, Recall, F1-Score) para cada classe no conjunto de teste.
* **Matriz de Confusão (`ConfusionMatrixDisplay`)**:
    * **Recurso Visual**: Exibe o desempenho do classificador, mostrando acertos e erros (falsos positivos/negativos) para cada classe.

### 6. Otimização do Modelo (Afinando o $n$)

#### **Busca Manual pelo Melhor $n$**
O *script* calcula e plota a **Taxa de Erro Média** para valores de $K$ (número de vizinhos) entre 1 e 15.
* **Recurso Visual**: Gráfico de linha que ajuda a identificar visualmente o valor de $K$ que minimiza o erro.

#### **Otimização de Hiperparâmetros com Grid Search**

O `GridSearchCV` é usado em conjunto com **Validação Cruzada (K-Fold com $k=5$)** para encontrar a melhor combinação de hiperparâmetros.

* **Parâmetros Avaliados (`param_grid`)**:
    * `n_neighbors`: `[3, 4, 8, 10, 12, 14]`
    * `weights`: `['distance']`
    * `metric`: `['euclidean', 'manhattan']`
    * `best_accuracy_history`: `86.44%`

O arquivo `pipeline.py` implementa um fluxo de trabalho completo para a **classificação de problemas ortopédicos da coluna vertebral** usando o algoritmo **K-Nearest Neighbors (KNN)**.

O processo inclui:
1.  **Obtenção e Preparação de Dados**: Carregamento do dataset, engenharia de *features* e tratamento inicial.
2.  **Análise Exploratória de Dados (EDA)**: Visualizações e estatísticas para entender a estrutura dos dados.
3.  **Pré-processamento e Remoção de Outliers**: Padronização da escala e limpeza de dados.
4.  **Treinamento e Avaliação do Modelo Base**: Treinamento de um modelo KNN inicial, incluindo o uso da técnica **SMOTE** para balanceamento de classes e avaliação com métricas e matriz de confusão.
5.  **Otimização de Hiperparâmetros (Grid Search)**: Busca pelo melhor conjunto de parâmetros para o modelo KNN.* **Métrica de Avaliação**: `accuracy_score`
* **Resultado**: O *script* imprime os `grid.best_params_` e treina o modelo KNN final com esses parâmetros otimizados.
* **Resultado Final**: A **Acurácia** do modelo otimizado é impressa.
