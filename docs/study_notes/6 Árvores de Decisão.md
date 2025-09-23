# Árvores de Decisão

As **Árvores de Decisão** são um dos algoritmos mais intuitivos e fundamentais do Machine Learning. Elas pertencem à categoria de aprendizagem supervisionada e podem ser usadas tanto para tarefas de **classificação** quanto de **regressão**.

> Pense nelas como um fluxograma, onde cada nó interno representa uma "pergunta" sobre uma característica (feature), cada ramo representa a resposta a essa pergunta, e cada nó folha (terminal) representa o resultado final.

## Como Funciona uma Árvore de Decisão?

O processo de construção de uma árvore é chamado de **particionamento recursivo**. A ideia é dividir o conjunto de dados em subconjuntos cada vez menores e mais "puros" até que um critério de parada seja atingido.

1.  **Começa no Nó Raiz (Root Node):** A árvore começa com um único nó que representa todo o conjunto de dados.
2.  **Encontra a Melhor Divisão (Split):** O algoritmo procura a melhor característica e o melhor valor de corte para dividir os dados. A "melhor" divisão é aquela que resulta nos subconjuntos mais homogêneos possíveis.
3.  **Divide os Dados:** O nó é dividido em nós filhos com base na pergunta e nas respostas (ex: "Idade > 30?"). Um ramo é criado para cada resposta (Sim/Não).
4.  **Repete Recursivamente:** O processo dos passos 2 e 3 é repetido para cada nó filho.
5.  **Critério de Parada:** A recursão para quando:
    * Todos os dados em um nó pertencem à mesma classe (o nó é "puro").
    * A árvore atinge uma profundidade máxima pré-definida.
    * O número de amostras em um nó é muito pequeno para continuar a divisão.
    * Quando um nó não é mais dividido, ele se torna um **nó folha (leaf node)**.

### Critérios de Divisão (Como a "Melhor Pergunta" é Escolhida?)

Para encontrar a melhor divisão, o algoritmo busca maximizar a "pureza" dos nós filhos. As duas métricas mais comuns para isso em tarefas de classificação são:

#### 1. Impureza Gini (Gini Impurity)

Usada pelo algoritmo **CART (Classification and Regression Trees)**. Mede a probabilidade de um elemento aleatório do conjunto ser classificado incorretamente. A impureza é 0 quando o nó é perfeitamente puro.
A fórmula para um dado nó é:

$$
Gini = \sum_{k=1}^{K} p_k (1 - p_k)
$$

Onde $p_k$ é a proporção de amostras da classe $k$ no nó.

#### 2. Entropia e Ganho de Informação (Entropy and Information Gain)

Usada pelos algoritmos **ID3** e **C4.5**. A entropia é uma medida de desordem ou incerteza. O algoritmo escolhe a divisão que proporciona o maior **Ganho de Informação**, que é a redução na entropia após a divisão.
A fórmula da entropia para um nó é:

$$
H = - \sum_{k=1}^{K} p_k \log_2(p_k)
$$


## Tipos de Árvores de Decisão

* **Árvores de Classificação:** O resultado nos nós folha é uma classe discreta (ex: "Aprovado", "Reprovado", "Spam").
* **Árvores de Regressão:** O resultado nos nós folha é um valor contínuo (ex: preço de um imóvel). Nesses casos, o critério de divisão geralmente é a **redução da variância** ou a minimização do **Erro Quadrático Médio (MSE)**.

## Vantagens e Desvantagens

### Vantagens

* **Fácil de Interpretar e Visualizar:** São modelos "white-box", cuja lógica pode ser facilmente explicada e desenhada.
* **Pouca Preparação de Dados:** Não exigem normalização ou escalonamento de features.
* **Lida com Dados Diversos:** Pode usar tanto características numéricas quanto categóricas.
* **Seleção de Features Implícita:** As características mais importantes aparecem naturalmente nos nós superiores da árvore.

### Desvantagens

* **Propensão a Overfitting:** Podem criar modelos excessivamente complexos que se ajustam aos dados de treino, mas não generalizam bem para novos dados.
* **Instabilidade:** Pequenas variações nos dados de treino podem gerar árvores completamente diferentes.
* **Viés:** Podem ser tendenciosas para características que possuem muitos níveis.
* **Fronteiras de Decisão Ortogonais:** As fronteiras de decisão são sempre paralelas aos eixos, o que pode não ser ideal para alguns problemas.

## Superando as Limitações

Na prática, raramente se usa uma única árvore de decisão. Em vez disso, elas servem como blocos de construção para algoritmos de **ensemble** muito mais poderosos:

* **Random Forest (Floresta Aleatória):** Constrói *muitas* árvores em diferentes subconjuntos de dados e de características. A previsão final é feita pela "votação" ou média das previsões de todas as árvores, reduzindo o overfitting.

* **Gradient Boosting (e variantes como XGBoost, LightGBM):** Constrói árvores de forma sequencial, onde cada nova árvore tenta corrigir os erros da árvore anterior. Estão entre os modelos de melhor desempenho para dados tabulares.

## Como Funciona a Árvore de Regressão?

#### 1. A Predição

Em uma árvore de regressão, a predição para qualquer ponto de dado que termina em um nó folha específico é simplesmente a **média (ou valor médio)** de todas as amostras de treinamento que caíram naquele mesmo nó folha.

Por exemplo, se um nó folha de uma árvore que prevê preços de imóveis contém 5 casas do conjunto de treino com os preços `[R$ 300k, R$ 320k, R$ 310k, R$ 330k, R$ 305k]`, a predição para qualquer nova casa que chegue a este nó será a média desses valores: `R$ 313k`.

#### 2. A Construção da Árvore: O Critério de Divisão

O objetivo ao construir a árvore é criar nós que sejam os mais homogêneos possíveis. Em regressão, "homogêneo" significa que os valores-alvo (`y`) dentro de um nó têm baixa variância.

Em vez de usar a Impureza Gini ou a Entropia, as árvores de regressão buscam a divisão que mais **reduz a variância** dos dados. O critério mais comum para isso é minimizar a **Soma dos Erros Quadráticos (Sum of Squared Errors - SSE)** ou, de forma equivalente, o **Erro Quadrático Médio (Mean Squared Error - MSE)**.

O algoritmo testa todas as divisões possíveis e escolhe aquela que resulta na menor soma de erros quadráticos nos nós filhos.

### Formulação Matemática Detalhada

Vamos formalizar o processo de divisão.

#### 1. Definição da Função de Custo para um Nó

Considere um nó de decisão `m` que representa uma região `R_m` do espaço de features, contendo `N_m` amostras de treinamento.

Primeiro, calculamos o valor previsto para este nó, que é a média dos valores-alvo das amostras nele contidas:

$$
\hat{y}_m = \frac{1}{N_m} \sum_{i \in R_m} y_i
$$

Onde:
- $\hat{y}_m$ é a predição para o nó `m`.
- $N_m$ é o número de amostras no nó `m`.
- $y_i$ é o valor-alvo real da amostra `i`.

A função de custo para este nó, que mede sua "impureza", é a **Soma dos Erros Quadráticos (SSE)** em relação a essa média:

$$
\text{Custo}(R_m) = \sum_{i \in R_m} (y_i - \hat{y}_m)^2
$$

#### 2. O Processo de Divisão (Splitting)

Uma divisão é definida por um par `(j, s)`, onde `j` é o índice de uma feature e `s` é um limiar (threshold) de valor. Essa divisão parte o conjunto de dados do nó pai `R_p` em dois subconjuntos (nós filhos):

- Região Esquerda: $R_{\text{left}}(j, s) = \{ (x, y) \in R_p \mid x_j \leq s \}$
- Região Direita: $R_{\text{right}}(j, s) = \{ (x, y) \in R_p \mid x_j > s \}$

O objetivo do algoritmo é encontrar o par `(j, s)` que **minimiza a soma dos custos dos nós filhos**:

$$
\min_{j, s} \left[ \text{Custo}(R_{\text{left}}(j, s)) + \text{Custo}(R_{\text{right}}(j, s)) \right]
$$

Substituindo a definição de custo, a expressão completa é:

$$
\min_{j, s} \left[ \sum_{i \in R_{\text{left}}} (y_i - \hat{y}_{\text{left}})^2 + \sum_{i \in R_{\text{right}}} (y_i - \hat{y}_{\text{right}})^2 \right]
$$

Onde $\hat{y}_{\text{left}}$ e $\hat{y}_{\text{right}}$ são as médias dos valores-alvo nos respectivos nós filhos.

### Exemplo Intuitivo

Imagine que você está prevendo o peso de uma pessoa com base na sua altura.

1. O algoritmo pega a feature "altura" e testa todos os valores únicos como possíveis limiares (ex: 1.65m, 1.70m, 1.75m...).
2. Para o teste "altura $\leq$ 1.70m":
   - Ele calcula a média dos pesos do grupo "$\leq$ 1.70m" e a soma dos erros quadráticos (SSE) desse grupo.
   - Ele calcula a média dos pesos do grupo "> 1.70m" e a soma dos erros quadráticos (SSE) desse grupo.
   - Ele soma esses dois valores de SSE.
3. O algoritmo repete isso para todos os outros limiares e para todas as outras features disponíveis.
4. A divisão que resultar na **menor soma total de SSE** é escolhida como a melhor divisão para aquele nó.

Este processo é então repetido recursivamente para os nós filhos até que um critério de parada (como profundidade máxima ou número mínimo de amostras por folha) seja atingido.

## Como Funciona a Árvore de Classificação?

#### 1. A Predição

A predição em uma árvore de classificação é feita por um sistema de "votação". Para qualquer novo ponto de dado que atravessa a árvore e termina em um nó folha, a predição é a **classe majoritária (a moda)** das amostras de treinamento que pertencem àquele nó.

Por exemplo, se um nó folha contém 10 amostras de treino, sendo 7 da classe "Aprovado" e 3 da classe "Reprovado", qualquer nova amostra que chegar a este nó será classificada como "Aprovado".

#### 2. A Construção da Árvore: O Critério de Divisão

O objetivo ao construir a árvore é criar nós que sejam os mais **"puros"** possíveis. Em classificação, um nó é 100% puro se todas as amostras que ele contém pertencem à mesma classe. Para medir a pureza e decidir qual a melhor divisão, o algoritmo utiliza métricas de impureza de classe, como:

1.  **Impureza Gini (Gini Impurity):** Mede a probabilidade de classificar incorretamente uma amostra escolhida aleatoriamente no nó.
2.  **Entropia e Ganho de Informação (Entropy and Information Gain):** A entropia mede o nível de "desordem" em um nó. O algoritmo busca a divisão que mais reduz essa desordem.

O algoritmo testa todas as divisões possíveis e escolhe aquela que resulta na maior "purificação" dos dados.

### Formulação Matemática Detalhada

#### Métrica 1: Impureza Gini (Gini Impurity)

Esta métrica é usada pelo popular algoritmo **CART (Classification and Regression Trees)**.

Para um nó `m` contendo amostras de `K` classes, a proporção de cada classe no nó é:

$$
p_{mk} = \frac{1}{N_m} \sum_{i \in R_m} I(y_i = k)
$$

Onde $N_m$ é o número de amostras no nó e $I(\cdot)$ é a função indicadora.

A **Impureza Gini** para o nó `m` é definida como:

$$
G_m = 1 - \sum_{k=1}^{K} (p_{mk})^2
$$

Um valor de $G_m=0$ indica um nó perfeitamente puro. Para uma divisão, o custo é a **média ponderada da impureza Gini dos filhos**:

$$
\text{Custo}_{\text{Gini}}(j, s) = \frac{N_{\text{left}}}{N_p} G_{\text{left}} + \frac{N_{\text{right}}}{N_p} G_{\text{right}}
$$

O algoritmo busca a feature `j` e o limiar `s` que **minimizam** esta função de custo.

#### Métrica 2: Entropia e Ganho de Informação (Entropy and Information Gain)

Esta métrica é a base dos algoritmos **ID3** e **C4.5**.

A **Entropia** de um nó `m`, que mede a incerteza, é calculada como:

$$
H_m = - \sum_{k=1}^{K} p_{mk} \log_2(p_{mk})
$$

Por convenção, $0 \log_2 0 = 0$. Uma entropia $H_m=0$ significa que o nó é perfeitamente puro.

O algoritmo busca **maximizar** o **Ganho de Informação (Information Gain - IG)**, que é a redução na entropia após a divisão:

$$
\text{IG}(j, s) = H_p - \left( \frac{N_{\text{left}}}{N_p} H_{\text{left}} + \frac{N_{\text{right}}}{N_p} H_{\text{right}} \right)
$$

Maximizar o Ganho de Informação é equivalente a minimizar a entropia ponderada dos filhos.

### Exemplo Intuitivo

Imagine que queremos prever se alguém vai jogar tênis (`Sim`/`Não`) com base no tempo (`Sol`, `Nublado`, `Chuva`).

1.  **Nó Raiz:** Contém todos os dados (ex: 9 `Sim` e 5 `Não`). O algoritmo calcula a impureza/entropia inicial.
2.  **Avalia a Divisão por "Tempo":**
    * O algoritmo cria três grupos: um para `Sol`, um para `Nublado` e um para `Chuva`.
    * Para cada grupo, ele conta quantos `Sim` e `Não` existem e calcula sua impureza/entropia.
    * Ele calcula o Ganho de Informação (ou a redução na impureza Gini) para esta divisão.
3.  **Avalia Outras Divisões:** O algoritmo faz o mesmo para outras features (umidade, vento, etc.).
4.  **Escolhe a Melhor:** A feature que proporcionar o maior Ganho de Informação (ou a menor impureza Gini ponderada) será escolhida como o primeiro critério de divisão da árvore.
5.  O processo se repete recursivamente para cada sub-nó criado.

## Quando Usar Árvores de Decisão? (Cenários Ideais)

#### Quando ela é Preferível

1.  **Quando a Interpretabilidade é Essencial:** Se você precisa explicar o "porquê" por trás de uma previsão, as árvores de decisão são uma das melhores escolhas.
2.  **Para um Rápido Modelo de Baseline:** São fáceis de treinar e não exigem pré-processamento complexo, sendo ótimas para obter uma primeira versão funcional de um modelo.
3.  **Quando os Dados Têm Relações Não-Lineares:** Elas capturam naturalmente interações complexas entre as features (ex: "SE idade > 40 E renda < 50k, ENTÃO...").

#### Quando ela Obtém o Melhor Resultado

O melhor desempenho não vem de uma única árvore, mas de seus ensembles.
1.  **Com Ensembles (Random Forest, Gradient Boosting):** O estado da arte do desempenho para a **maioria dos problemas com dados estruturados/tabulares** (planilhas, tabelas) é alcançado com algoritmos como **XGBoost e LightGBM**.

#### Quando ela Não é a Melhor Escolha

1.  **Para Dados Não Estruturados:** Para tarefas com imagens, áudio ou texto, as redes neurais (CNNs, Transformers) são muito superiores.
2.  **Quando a Relação é Claramente Linear:** Um modelo mais simples, como a Regressão Linear ou Logística, pode ter um desempenho similar ou até melhor, com uma interpretação ainda mais simples.

## Métodos de Interpretação

Existem três abordagens principais para interpretar uma árvore de decisão:

1.  Visualização da Árvore Completa.
2.  Análise da Importância das Features (Feature Importance).
3.  Extração de Regras de Decisão em Texto.

#### 1. Visualização da Árvore (Interpretação Global e Local)

Este é o método mais direto e intuitivo, ideal para árvores que não são excessivamente profundas.

**Como ler um nó da árvore visualizada?**

Cada caixa (nó) na visualização contém informações cruciais:

* **A Condição de Divisão** (ex: `petal length (cm) <= 2.45`): A "pergunta" que o nó faz. Se for verdadeira, segue-se pelo ramo da esquerda; se for falsa, pelo da direita. O nó raiz (o primeiro de todos) mostra a feature mais importante.
* **Métrica de Impureza** (ex: `gini = 0.667`): O valor da métrica de impureza (Gini ou Entropia). Um valor de `0.0` indica um nó perfeitamente puro.
* **Samples** (ex: `samples = 150`): O número total de amostras de treinamento que passaram por aquele nó.
* **Value** (ex: `value = [50, 50, 50]`): A distribuição das amostras entre as classes.
* **Class** (ex: `class = setosa`): A classe majoritária naquele nó. Em um nó folha, esta seria a predição final.

#### 2. Análise da Importância das Features (Interpretação Global)

A árvore de decisão calcula naturalmente a importância de cada feature. A lógica é simples: **features que resultam na maior redução de impureza são consideradas mais importantes.**

* **Como funciona?** A importância de uma feature é a soma total da redução de impureza que ela proporciona em todos os nós onde é usada, ponderada pelo número de amostras que passam por eles.
* **Na prática:** Bibliotecas como `scikit-learn` calculam isso automaticamente através do atributo `.feature_importances_`. O ideal é plotar um gráfico de barras para ranquear as features.

Isso responde à pergunta: **"Quais são os fatores mais decisivos que meu modelo aprendeu?"**

#### 3. Extração de Regras de Decisão (Interpretação Global e Local)

Quando a árvore é muito grande para ser visualizada, a melhor abordagem é extrair suas regras em formato de texto (`IF-THEN`).

* **Na prática:** O `scikit-learn` oferece a função `export_text` que gera um resumo legível.

**Exemplo de saída:**

```
| feature_2 <= 2.45
|   | class: 0
| feature_2 > 2.45
|   | feature_3 <= 1.75
|   |   | feature_2 <= 4.95
|   |   |   | class: 1
|   |   | feature_2 > 4.95
|   |   |   | class: 2
|   | feature_3 > 1.75
|   |   | class: 2
```

### Passo a Passo: Explicando uma Previsão Específica (Interpretação Local)

Esta é a aplicação mais comum no mundo dos negócios: "Por que o cliente X teve seu crédito negado?".

1.  **Selecione a Instância:** Pegue os dados do cliente X (idade, renda, histórico, etc.).
2.  **Trace o Caminho na Árvore:** Comece no nó raiz. Use os dados do cliente para responder à pergunta do nó e siga para o ramo correspondente.
3.  **Repita o Processo:** Continue descendo na árvore, respondendo à pergunta de cada nó, até chegar a um nó folha.
4.  **Construa a Narrativa:** O caminho que você percorreu é a explicação.

**Exemplo de Narrativa:**

> "A solicitação de crédito do cliente X foi **negada** porque:
> 1.  Seu `histórico de crédito` é 'ruim' (primeira divisão na árvore), E
> 2.  Sua `renda anual` é inferior a R$ 50.000 (segunda divisão).
>
> Esses fatores o colocaram em um grupo final onde 85% dos clientes com perfil semelhante também tiveram o crédito negado."

### Limitações a Considerar

* **Profundidade Excessiva:** Árvores muito profundas podem se tornar uma "selva de regras", difíceis de interpretar como um todo.
* **Instabilidade:** Pequenas mudanças nos dados de treino podem gerar uma árvore completamente diferente. Isso significa que a "explicação" para um mesmo caso pode mudar se o modelo for treinado novamente.

## Exemplo Completo da Árvore de Decisão: Da Construção à Classificação

Esta seção apresenta uma análise completa e detalhada de como um algoritmo de Árvore de Decisão funciona. O processo é dividido em duas fases principais:

1.  **Fase de Construção (Treinamento):** Onde o modelo aprende as regras a partir de um conjunto de dados.
2.  **Fase de Classificação (Predição):** Onde a árvore treinada é usada para classificar uma nova amostra.

### Fase 1: Construção da Árvore (Treinamento)

Nesta fase, o objetivo é criar a melhor estrutura de "perguntas" em cascata para separar os dados de forma eficiente.

#### Etapa 0: O Cenário (O Conjunto de Dados)

Utilizaremos um conjunto de dados com 10 amostras, 4 características e 3 classes de dispositivos eletrônicos: **Notebook (0)**, **Tablet (1)** e **Celular (2)**.

| Amostra | F0: Tela (pol) | F1: Peso (g) | F2: RAM (GB) | F3: Bateria (mAh) | **Classe** | **Dispositivo** |
| :--- | :--- | :--- | :--- | :--- | :--- |:--- |
| A0 | 15.6 | 1800 | 8 | 5000 | **0** | Notebook |
| A1 | 14.0 | 1200 | 16 | 6000 | **0** | Notebook |
| A2 | 13.3 | 950 | 8 | 4500 | **0** | Notebook |
| A3 | 11.0 | 700 | 4 | 8000 | **1** | Tablet |
| A4 | 10.1 | 650 | 6 | 7500 | **1** | Tablet |
| A5 | 12.9 | 720 | 8 | 9000 | **1** | Tablet |
| A6 | 6.7 | 210 | 8 | 5000 | **2** | Celular |
| A7 | 6.1 | 180 | 4 | 4000 | **2** | Celular |
| A8 | 6.9 | 240 | 12 | 5500 | **2** | Celular |
| A9 | 6.5 | 190 | 6 | 4500 | **2** | Celular |

**Distribuição Inicial:** {Notebook: 3, Tablet: 3, Celular: 4} -> Total de 10 amostras.

#### Etapa 1: Ponto de Partida - Entropia do Nó Raiz

Iniciamos com todas as 10 amostras no nó raiz. O primeiro passo é medir a "impureza" ou "desordem" inicial deste nó usando a Entropia de Shannon.

**Cálculo da Entropia do Nó Raiz (Pai):**
A fórmula da entropia é:
$$H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$$

1.  **Proporções (`p_i`):**
    * $p_{Notebook} = 3/10 = 0.3$
    * $p_{Tablet} = 3/10 = 0.3$
    * $p_{Celular} = 4/10 = 0.4$

2.  **Cálculo:**
    $$H(Raiz) = - [ (0.3 \cdot \log_2(0.3)) + (0.3 \cdot \log_2(0.3)) + (0.4 \cdot \log_2(0.4)) ]$$
    $$H(Raiz) = - [ (0.3 \cdot -1.737) + (0.3 \cdot -1.737) + (0.4 \cdot -1.322) ]$$
    $$H(Raiz) = - [ -0.521 - 0.521 - 0.529 ] \approx 1.571$$

> A entropia inicial do conjunto de dados é **1.571**. Este é o valor que o algoritmo tentará reduzir a cada divisão.

#### Etapa 2: A Busca Exaustiva pela Melhor Divisão

O algoritmo agora busca a divisão que maximiza o **Ganho de Informação**, definido como: `Ganho = H(Pai) - H_ponderada(Filhos)`.

* **Variáveis de controle inicializadas:**
    * `best_gain = -1.0`
    * `best_split_idx = None`
    * `best_split_thresh = None`

##### Análise da Característica F1: "Peso" (Exemplo Detalhado)

Após testar outras características com ganhos menores, vamos analisar a F1 ("Peso"). O algoritmo testa vários limiares. Vamos analisar o limiar **`Peso <= 300`**.

1.  **Divisão:**
    * **Filho Esquerda (Peso <= 300):** 4 amostras com rótulos `[2, 2, 2, 2]` -> {Celular: 4}. **(Nó Puro!)**
    * **Filho Direita (Peso > 300):** 6 amostras com rótulos `[0, 0, 0, 1, 1, 1]` -> {Notebook: 3, Tablet: 3}.

2.  **Entropia dos Filhos:**
    * $H(Esquerda) = 0$ (nós puros têm entropia zero).
    * $H(Direita) = -[(3/6 \cdot \log_2(3/6)) + (3/6 \cdot \log_2(3/6))] = 1.0$

3.  **Entropia Ponderada Final:**
    * $H_{pond} = (4/10 \cdot H(Esquerda)) + (6/10 \cdot H(Direita))$
    * $H_{pond} = (0.4 \cdot 0) + (0.6 \cdot 1.0) = 0.6$

4.  **Cálculo do Ganho de Informação:**
    * `Ganho = H(Raiz) - H_{pond} = 1.571 - 0.6 = 0.971`

> **Atualização:** Como `0.971` é o maior ganho encontrado até agora, as variáveis são atualizadas: `best_gain = 0.971`, `best_split_idx = 1`, `best_split_thresh = 300`.

#### Etapa 3: A Decisão Final e a Primeira Divisão

Após testar todas as combinações, a divisão com o maior ganho foi a da Característica "Peso" com limiar 300. A árvore começa a ser formada.

#### Etapa 4: Recursão - Construindo o Próximo Nível

O processo recomeça para o nó da direita, que agora é o novo "pai".

* **Novo Nó Pai:** Contém 6 amostras: {Notebook: 3, Tablet: 3}.
* **Entropia do Novo Nó:** $H = 1.0$.

O algoritmo buscará a melhor divisão para estas 6 amostras. Suponha que ele encontre que `Tela <= 12.0` é a melhor regra, com um `Ganho = 0.46`. A árvore se expande:

O processo continua até que todos os nós sejam puros ou outro critério de parada seja atingido.

### Fase 2: Classificação de uma Nova Amostra (Predição)

Com a árvore treinada, podemos classificar um novo dispositivo.

**Nova Amostra:** `{Tela: 14.5, Peso: 1300g, RAM: 8GB, Bateria: 5200mAh}`

O processo é uma simples travessia pela árvore:

1.  **Começar na Raiz:**
    * **Pergunta:** `Peso <= 300g?`
    * **Análise:** `1300 <= 300` é **FALSO**.
    * **Ação:** Seguir pelo ramo da **direita (NÃO)**.

2.  **Percorrer o Próximo Nó:**
    * **Pergunta:** `Tela <= 12.0 polegadas?`
    * **Análise:** `14.5 <= 12.0` é **FALSO**.
    * **Ação:** Seguir pelo ramo da **direita (NÃO)**.

3.  **Chegar a um Nó Folha:**
    * Seguindo este caminho, chegaríamos a um nó que, após mais divisões, conteria majoritariamente amostras da classe "Notebook".

> **Predição Final:** A classificação para a nova amostra é **Notebook (Classe 0)**.

## Variantes do Algoritmo de Árvore de Decisão

Existem diversas variantes e evoluções do algoritmo de Árvore de Decisão. Cada uma foi desenvolvida para resolver limitações da versão anterior, como a forma de escolher a melhor divisão, o tratamento de diferentes tipos de dados e o controle do superajuste (overfitting).

### 1. ID3 (Iterative Dichotomiser 3)

É um dos algoritmos mais antigos e fundamentais. Foi o precursor de muitas outras variantes.

* **Critério de Divisão:** Utiliza o **Ganho de Informação (Information Gain)**, baseado em Entropia. Ele escolhe o atributo que fornece a maior redução na incerteza (entropia) sobre as classes.
* **Características Principais:**
    * Funciona apenas com atributos **categóricos**. Ele não lida nativamente com dados numéricos contínuos.
    * Tende a favorecer atributos com um grande número de valores distintos, o que pode não levar à melhor árvore.
    * Não possui um mecanismo de poda (*pruning*), o que o torna muito suscetível ao overfitting.

### 2. C4.5

É a evolução direta do ID3, criada pelo mesmo autor (Ross Quinlan) para superar suas principais limitações.

* **Critério de Divisão:** Utiliza o **Ganho de Informação Normalizado (Gain Ratio)**. Essa métrica ajusta o Ganho de Informação para penalizar atributos com muitos valores, resolvendo uma das maiores falhas do ID3.
* **Características Principais:**
    * Lida tanto com atributos **categóricos quanto contínuos**. Para os contínuos, ele encontra um ponto de corte ideal para criar uma divisão binária.
    * Consegue tratar **dados faltantes**, atribuindo probabilidades a diferentes caminhos na árvore.
    * Implementa a **poda pós-construção (post-pruning)**, uma técnica que remove galhos da árvore após ela ter sido totalmente construída, a fim de reduzir o overfitting e melhorar a generalização.

### 3. CART (Classification and Regression Trees)

É talvez a variante mais popular e a mais implementada em bibliotecas modernas, como o scikit-learn. Como o nome sugere, é projetado tanto para classificação quanto para regressão.

* **Critério de Divisão:**
    * **Classificação:** Utiliza o **Índice de Gini (Gini Impurity)**, que é computacionalmente mais eficiente que a Entropia, pois não exige cálculos de logaritmo.
    * **Regressão:** Utiliza a **Redução de Variância**, buscando a divisão que minimiza a variância dos valores nos nós filhos.
* **Características Principais:**
    * Constrói exclusivamente **árvores binárias**, ou seja, cada nó se divide em exatamente dois filhos.
    * Lida com dados categóricos e contínuos.
    * Utiliza um método de poda sofisticado chamado **Cost-Complexity Pruning**, que poda a árvore com base em um parâmetro de complexidade.

### 4. CHAID (Chi-squared Automatic Interaction Detection)

Esta variante tem uma abordagem estatística mais formal para a criação de divisões.

* **Critério de Divisão:** Utiliza o **teste estatístico Qui-quadrado ($\chi^2$)** para determinar a melhor divisão. Ele mede a significância da associação entre os atributos e a variável alvo.
* **Características Principais:**
    * Pode criar divisões com **mais de dois galhos (árvores não binárias)**. Por exemplo, se um atributo categórico como "Região" tiver os valores "Norte", "Sul" e "Nordeste", o CHAID pode criar um nó com três galhos.
    * É muito utilizado em marketing e pesquisa de ciências sociais por sua interpretabilidade.
    * Não necessita de poda; a parada da árvore é determinada pela significância estatística.

### 5. MARS (Multivariate Adaptive Regression Splines)

Embora não seja uma "árvore" no sentido tradicional, o MARS é frequentemente visto como um parente próximo. Ele usa divisões semelhantes às das árvores para modelar não-linearidades nos dados.

* **Critério de Divisão:** Em vez de criar galhos, o MARS cria **funções de base lineares por partes (splines)**. Cada "nó" corresponde a um ponto de inflexão onde o comportamento da função muda.
* **Características Principais:**
    * É primariamente um modelo de **regressão**.
    * Excelente para capturar relações complexas e não-lineares nos dados, produzindo um modelo que é mais suave e contínuo que uma árvore de regressão padrão.
    * O processo de construção envolve uma fase de avanço (adicionando termos) e uma fase de retrocesso (removendo termos menos importantes), que é análoga à poda.

### Tabela Resumo

| Variante | Critério de Divisão (Classificação) | Tipo de Árvore | Lida com Dados Contínuos? | Poda (Pruning)? |
| :--- | :--- | :--- | :--- | :--- |
| **ID3** | Ganho de Informação | Multi-galho | Não | Não |
| **C4.5** | Ganho Normalizado | Multi-galho | Sim | Sim (Pós-poda) |
| **CART** | Índice de Gini | Binária | Sim | Sim (Cost-Complexity) |
| **CHAID** | Teste Qui-quadrado | Multi-galho | Sim | Não (usa parada estatística) |
| **MARS** | (Modelo de Regressão) | Funções Spline | Sim | Sim (Fase de retrocesso) |
