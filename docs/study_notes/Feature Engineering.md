# Guia Abrangente de Feature Engineering (Engenharia de Atributos)

A **Feature Engineering** (Engenharia de Atributos ou Recursos) é a arte e a ciência de transformar dados brutos em representações numéricas e contextuais que maximizam a capacidade preditiva de algoritmos de Machine Learning (ML).

Frequentemente citada como uma das etapas mais determinantes para o sucesso de projetos de dados, ela resume a máxima do setor: *"melhores features superam algoritmos mais complexos"*.

---

## 1. O que é e por que é essencial?

Modelos de aprendizado de máquina não compreendem o mundo como humanos; eles compreendem números, distribuições, distâncias e matrizes. Se os dados de entrada forem ruidosos, mal escalonados ou desconectados do problema de negócio, o modelo sofrerá do princípio **GIGO** (*Garbage In, Garbage Out*).

### Objetivos Principais:
- **Aumentar o poder preditivo:** Destacar padrões e relações latentes que o modelo teria dificuldade em aprender sozinho.
- **Reduzir a complexidade do modelo:** Um modelo mais simples (ex.: regressão linear ou árvore rasa) com excelentes variáveis frequentemente supera redes neurais complexas treinadas sobre dados brutos.
- **Melhorar a interpretabilidade:** Recursos alinhados a regras de negócio tornam as decisões do modelo auditáveis e explicáveis.

---

## 2. Tipos de Variáveis e Técnicas por Modalidade

Cada tipo de dado exige transformações específicas para se adequar às hipóteses dos algoritmos.

---

### A. Variáveis Numéricas

Variáveis numéricas contínuas ou discretas frequentemente apresentam distribuições assimétricas, ordens de grandeza muito distintas ou limites desiguais.

#### 1. Escalonamento e Normalização
Algoritmos baseados em distância (KNN, SVM, K-Means) ou gradiente descendente (Regressão Linear/Logística, Redes Neurais) são sensíveis à escala:

*   **Padronização (StandardScaler / Z-Score):**
    Centraliza a média em 0 e ajusta o desvio padrão para 1.
    $$z = \frac{x - \mu}{\sigma}$$
    *Ideal quando os dados seguem aproximadamente uma distribuição normal.*

*   **Min-Max Scaling:**
    Comprime os valores para um intervalo delimitado (usualmente $[0, 1]$).
    $$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
    *Muito utilizado em redes neurais e processamento de imagens.*

*   **Robust Scaling:**
    Utiliza a mediana e o intervalo interquartil ($IQR = Q_3 - Q_1$), sendo resistente à influência de outliers severos.
    $$x_{robust} = \frac{x - Q_2}{IQR}$$

#### 2. Transformações de Distribuição
Muitos modelos lineares pressupõem normalidade dos resíduos e homoscedasticidade. Para corrigir distribuições com cauda longa (assimetria positiva):
*   **Transformação Logarítmica:** $y = \log(x + 1)$, comprime valores extremos.
*   **Transformações de Potência:** Box-Cox (apenas para valores $x > 0$) e Yeo-Johnson (suporta valores negativos e zero).

#### 3. Discretização (Binning)
Transforma variáveis contínuas em intervalos categóricos ou ordinais (ex.: faixas etárias: *18–25*, *26–40*, *40+*). Ajuda a capturar relações não-lineares em modelos lineares e a mitigar o impacto de outliers.

---

### B. Variáveis Categóricas

Modelos matemáticos exigem valores numéricos. A conversão de categorias exige cautela para não introduzir ordenações espúrias.

| Técnica | Descrição | Vantagens | Desvantagens |
| :--- | :--- | :--- | :--- |
| **One-Hot Encoding** | Cria uma coluna binária (0/1) para cada categoria. | Não assume ordenação; simples de interpretar. | Causa a *maldição da dimensionalidade* se a cardinalidade for alta. |
| **Ordinal / Label Encoding** | Atribui um inteiro para cada categoria ($1, 2, 3\dots$). | Compacto; não expande o número de colunas. | Induz relação de grandeza que pode não existir (ex.: Verde=1, Azul=2, Vermelho=3). |
| **Target Encoding (Mean Encoding)** | Substitui cada categoria pela média da variável alvo correspondente. | Mantém 1 dimensão mesmo em alta cardinalidade. | Alto risco de **Data Leakage** e overfitting; exige suavização/regularização. |
| **Frequency / Count Encoding** | Substitui a categoria pela contagem de ocorrências dela no dataset. | Informa a raridade da categoria sem expandir colunas. | Categorias distintas com mesma contagem tornam-se indistinguíveis. |

---

### C. Dados Temporais (Data e Hora)

Informações temporais brutas (ex.: `2026-11-03 14:30:00`) raramente devem ser passadas diretamente para o modelo. Em vez disso, extrai-se contexto:

1.  **Componentes Calendáricos:**
    *   Ano, mês, dia do mês, dia da semana, hora do dia.
    *   Flags booleanas: `eh_fim_de_semana`, `eh_feriado`, `horario_comercial`.
2.  **Transformações Cíclicas (Seno e Cosseno):**
    *   O dia 31 de dezembro está próximo do dia 1 de janeiro, assim como 23h59 está próxima de 00h00. Para representar essa continuidade circular:
        $$x_{\sin} = \sin\left(\frac{2\pi \cdot t}{T}\right), \quad x_{\cos} = \cos\left(\frac{2\pi \cdot t}{T}\right)$$
        *(onde $t$ é o valor atual e $T$ é o período total, ex.: 24 para horas ou 7 para dias).*
3.  **Lags e Janelas Móveis (Rolling Windows):**
    *   **Lags:** Valores passados ($t-1, t-7, t-30$).
    *   **Agregações em Janela:** Média, desvio padrão, mínimo e máximo das últimas $N$ horas/dias.
    *   **Diferenças:** Variação absoluta ou percentual em relação ao período anterior.

---

### D. Interação e Conhecimento de Domínio

Muitas vezes, a relação entre duas variáveis é mais informativa do que cada uma isoladamente:

*   **Relações e Razões (Ratios):**
    *   Exemplo no setor financeiro: $\text{Dívida} / \text{Renda}$ ou $\text{Gasto com Cartão} / \text{Limite Total}$.
    *   Exemplo no e-commerce: $\text{Itens Comprados} / \text{Sessões no App}$.
*   **Agregações Cruzadas (Group-by Features):**
    *   Calcular estatísticas em múltiplos níveis hierárquicos: "Gasto médio do cliente comparado ao gasto médio dos clientes da mesma região".
    *   $$x_{relativo} = x_i - \mu_{\text{categoria}}$$

---

## 3. Tratamento de Dados Ausentes e Outliers

A imputação de dados faz parte do pipeline de feature engineering:

1.  **Imputação Estatística Simples:** Média ou mediana (numéricas) e moda (categóricas).
2.  **Imputação Avançada:** KNN Imputer, MICE (*Multivariate Imputation by Chained Equations*) ou modelos preditivos dedicados.
3.  **Flag de Ausência:** Criar uma coluna booleana `is_missing_colunaX`. O fato de um dado estar ausente pode ser, por si só, um forte preditor de risco ou comportamento.
4.  **Tratamento de Outliers:**
    *   **Winsorization / Trimming:** Cortar os percentis extremos (ex.: 1º e 99º percentis).
    *   **Capping por IQR:** Substituir valores fora de $[Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR]$.

---

## 4. Seleção de Features (Feature Selection)

Criar centenas de atributos pode introduzir ruído, aumentar o custo computacional e degradar o desempenho do modelo (maldição da dimensionalidade). Por isso, a seleção é um complemento indispensável:

1.  **Métodos de Filtro (Filter Methods):**
    *   Análises estatísticas independentes do modelo (correlação de Pearson/Spearman, Teste Qui-Quadrado, ANOVA, Informação Mútua).
2.  **Métodos Wrapper:**
    *   Avaliam combinações de features treinando modelos iterativamente (RFE - *Recursive Feature Elimination*, Forward/Backward Selection). São mais custosos computacionalmente.
3.  **Métodos Embutidos (Embedded Methods):**
    *   Seleção durante o próprio treinamento do modelo (Penalização L1/Lasso, Feature Importance de Random Forest ou XGBoost / LightGBM, valores de SHAP).

---

## 5. Cuidados Críticos e Boas Práticas

### A. Prevenção de Data Leakage (Vazamento de Dados)
O erro mais comum e custoso em feature engineering:
*   **Regra de Ouro:** Todas as transformações (médias para imputação, escalas min-max, target encoding, etc.) devem ser aprendidas **apenas no conjunto de treino** (`fit`) e apenas aplicadas no conjunto de teste/validação (`transform`).
*   Em dados temporais, a validação cruzada deve sempre respeitar o tempo (*Time Series Split*), nunca selecionando dados futuros para prever eventos passados.

### B. Reprodutibilidade com Pipelines
Evite criar scripts com variáveis manuais soltas. Utilize abstrações robustas, como:
*   `scikit-learn` Pipelines e `ColumnTransformer`.
*   Bibliotecas dedicadas como `Feature-engine` ou `Category Encoders`.
*   Feature Stores (ex.: Feast, Hopsworks) para ambientes corporativos de produção em larga escala.

---

## Resumo em Diagrama de Fluxo

```
[Dados Brutos] 
       │
       ▼
[Limpeza & Imputação] ───► Gera Flags de Ausência
       │
       ▼
[Engenharia por Tipo]
  ├── Numéricas  ──► Escalonamento, Log, Ratios, Binning
  ├── Categóricas──► One-Hot, Target, Frequency Encoding
  └── Temporais  ──► Componentes, Seno/Cosseno, Janelas Móveis
       │
       ▼
[Seleção de Features] ──► Filtros, RFE, Importância via SHAP
       │
       ▼
[Modelagem (ML)]
```