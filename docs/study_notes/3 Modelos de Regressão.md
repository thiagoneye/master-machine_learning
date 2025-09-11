# Sumário

- [[#Regressão Linear]]
- [[#Otimização com Gradiente Descendente]]
- [[#Regularização]]
- [[#Métricas de Avaliação]]
- [[#Regressão Multivariada]]
- [[#Regressão Não-Linear]]

--- 
# Regressão Linear

A regressão linear é uma das técnicas mais fundamentais e amplamente utilizadas no campo da estatística e do machine learning. Seu principal objetivo é modelar e analisar a relação entre uma variável dependente (também chamada de resposta ou alvo) e uma ou mais variáveis independentes (conhecidas como preditoras ou explicativas). A essência da técnica reside em encontrar a linha reta que melhor se ajusta aos dados, permitindo assim fazer previsões.

## Tipos de Regressão Linear

Existem duas categorias principais de regressão linear, que se distinguem pelo número de variáveis preditoras utilizadas:

1. **Regressão Linear Simples:** Utiliza apenas **uma** variável independente para prever a variável dependente. A relação é descrita pela equação de uma reta:  
$$Y = \beta_0 + \beta_1 X + \epsilon$$
2. **Regressão Linear Múltipla:** Emprega **duas ou mais** variáveis independentes para modelar a variável dependente. A equação se expande para representar um plano ou hiperplano:  
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon$$

### Componentes da Equação

- $Y$: A variável dependente que se deseja prever.
- $X$: As variáveis independentes que são usadas para fazer a previsão.
- $\beta_0$​ **(Intercepto)**: É o valor previsto de $Y$ quando todas as variáveis independentes são iguais a zero. Graficamente, é o ponto onde a linha de regressão cruza o eixo vertical.
- $\beta_1​,\beta_2​,\dots$ **(Coeficientes Angulares)**: Representam a mudança média na variável dependente $Y$ para cada aumento de uma unidade na variável independente correspondente, mantendo todas as outras variáveis constantes. Eles quantificam a força e a direção (positiva ou negativa) da relação.
- $\epsilon$ **(Termo de Erro)**: Representa a variação em $Y$ que não é explicada pelas variáveis independentes do modelo. É a diferença entre os valores observados e os valores previstos pela reta de regressão.

O método mais comum para encontrar os melhores valores para os coeficientes ($\beta$) é o **Método dos Mínimos Quadrados Ordinários (MQO)**, que minimiza a soma dos quadrados das distâncias verticais entre os pontos de dados reais e a linha de regressão.

### As Suposições do Modelo

Para que um modelo de regressão linear seja considerado válido e suas inferências confiáveis, certas condições, ou suposições, devem ser atendidas:

1. **Linearidade:** A relação entre as variáveis independentes e a variável dependente deve ser linear.
2. **Independência:** Os erros (resíduos) devem ser independentes uns dos outros.
3. **Homocedasticidade:** A variância dos erros deve ser constante em todos os níveis das variáveis independentes.
4. **Normalidade dos Erros:** Os erros devem seguir uma distribuição normal.
5. **Ausência de Multicolinearidade (para regressão múltipla):** As variáveis independentes não devem ser altamente correlacionadas entre si.

A violação dessas suposições pode levar a conclusões enganosas e previsões imprecisas.

### Avaliando o Desempenho do Modelo

Após o ajuste do modelo, é crucial avaliar sua qualidade e poder preditivo:

- **Coeficiente de Determinação ($R^2$):** Mede a proporção da variabilidade na variável dependente que é explicada pelo modelo. Varia de $0$ a $1$, com valores mais altos indicando um melhor ajuste.
- $R^2$ **Ajustado:** Uma versão modificada do $R^2$ que leva em conta o número de preditores no modelo, penalizando a inclusão de variáveis desnecessárias.
- **Teste F:** Avalia a significância estatística geral do modelo, testando se pelo menos um dos coeficientes das variáveis preditoras é diferente de zero.
- **Teste t:** Utilizado para determinar a significância estatística de cada coeficiente individualmente.    

## Aspectos Relevantes

- **O que "Linear" Realmente Significa** Uma das maiores fontes de confusão é o termo "linear". A regressão linear não exige que a relação entre a variável dependente ($Y$) e a independente ($X$) seja uma linha reta. Ela exige que o modelo seja **linear nos seus parâmetros** ($\beta$). Isso significa que você pode modelar relações curvas usando regressão linear. Por exemplo, a **regressão polinomial** é um tipo de regressão linear $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2^2 + \epsilon$. A equação descreve uma parábola (uma curva), mas o modelo é linear porque os coeficientes $\beta_0​,\beta_1​, \beta_2$​ não estão elevados a potências, multiplicados entre si ou dentro de funções não-lineares (como $sin(\beta)$).
- **Extremamente Sensível a Outliers** Como o Método dos Mínimos Quadrados Ordinários (MQO) minimiza a **soma dos erros ao quadrado**, ele penaliza desproporcionalmente os erros grandes. Um único ponto de dado muito distante da linha de tendência (um outlier) pode ter um enorme poder de "puxar" a linha de regressão em sua direção, distorcendo completamente os resultados.

## Alternativas ao Método dos Mínimos Quadrados Ordinários (MQO)

O MQO é o método padrão por ser matematicamente conveniente e eficiente sob as condições ideais. No entanto, quando essas condições não são atendidas (especialmente pela presença de outliers), outras técnicas são mais robustas e adequadas.

1. **Regressão Robusta (Robust Regression)** Este é um grupo de métodos projetados especificamente para serem menos sensíveis a outliers.
    - **Mínimos Desvios Absolutos (LAD - Least Absolute Deviations):** Em vez de minimizar a soma dos erros ao quadrado ($e^2$), este método minimiza a soma dos valores absolutos dos erros ($|e|$). Erros grandes têm menos influência, pois não são elevados ao quadrado.
    - **Regressão de Huber:** Uma abordagem híbrida inteligente. Para erros pequenos, ela se comporta como o MQO (quadrática), mas para erros grandes, ela se comporta como a LAD (linear). Isso a torna robusta a outliers, mas ainda eficiente quando não há outliers.
2. **Regressão Regularizada (Regularized Regression)** Esses métodos são usados principalmente para evitar o superajuste (overfitting) e lidar com multicolinearidade (quando as variáveis preditoras são altamente correlacionadas). Eles funcionam adicionando uma "penalidade" à equação que o modelo tenta minimizar.
    - **Regressão Ridge (Regularização L2):** Adiciona uma penalidade proporcional à soma dos quadrados dos coeficientes. Isso "encolhe" os coeficientes, reduzindo sua magnitude, o que é útil quando há multicolinearidade. Os coeficientes se aproximam de zero, mas nunca chegam a ser exatamente zero.
    - **Regressão Lasso (Regularização L1):** Adiciona uma penalidade proporcional à soma dos valores absolutos dos coeficientes. A grande vantagem do Lasso é que ele pode reduzir coeficientes de variáveis menos importantes a **exatamente zero**, efetivamente realizando uma seleção automática de variáveis.
3. **Regressão Bayesiana** Em vez de encontrar um único valor "ótimo" para cada coeficiente (β), a abordagem bayesiana trata os coeficientes como variáveis aleatórias. Ela usa os dados para atualizar uma crença prévia sobre os parâmetros, resultando em uma distribuição de probabilidade posterior para cada um. Isso permite uma quantificação mais rica da incerteza do modelo.
4. **Métodos de Otimização (Alternativa Computacional)** Para conjuntos de dados gigantescos, calcular a solução exata do MQO (através da "equação normal") pode ser computacionalmente inviável. Nesses casos, algoritmos iterativos como o **Gradiente Descendente (Gradient Descent)** são usados para encontrar os valores dos coeficientes. Embora o objetivo seja o mesmo (minimizar a soma dos erros ao quadrado), a abordagem para chegar lá é diferente.

---
# Otimização com Gradiente Descendente

| Característica                     | Método dos Mínimos Quadrados Ordinários (MQO)                                                                     | Gradiente Descendente                                                                                         |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Natureza**                       | **Analítico (Solução Direta)**                                                                                    | **Numérico (Solução Iterativa)**                                                                              |
| **Como Funciona**                  | Resolve uma equação matemática (a "equação normal") que encontra o ponto mínimo da função de custo de uma só vez. | Começa com uma estimativa e dá passos graduais na direção que mais reduz o erro, até convergir para o mínimo. |
| **Computação**                     | Envolve operações de álgebra linear, como inversão de matrizes.                                                   | Envolve o cálculo de derivadas (gradientes) e atualizações repetidas dos parâmetros.                          |
| **Velocidade**                     | Rápido para conjuntos de dados pequenos a médios.                                                                 | Pode ser lento para convergir com muitos dados (Batch), mas rápido por iteração (SGD).                        |
| **Uso de Memória**                 | Alto. Requer o carregamento de todos os dados na memória para calcular as matrizes.                               | Baixo a alto, dependendo da variação (SGD usa pouquíssima memória).                                           |
| **Escalabilidade**                 | **Não escala bem** para conjuntos de dados muito grandes (muitas amostras) ou com muitas características.         | **Escala muito bem**. É a abordagem padrão para Big Data e modelos complexos.                                 |
| **Necessidade de Hiperparâmetros** | Não precisa de hiperparâmetros como a taxa de aprendizado.                                                        | Requer o ajuste cuidadoso da **taxa de aprendizado (**α**)** e do número de iterações.                        |
| **Precisão**                       | Encontra a solução **exata** e ótima de uma só vez.                                                               | Encontra uma solução **aproximada**, que se aproxima do ótimo a cada iteração.                                |
| **Uso Online**                     | Não pode ser usado em cenários de _online learning_ (dados chegam em tempo real).                                 | Pode ser usado com _online learning_ (especialmente a versão Estocástica - SGD).                              |

---
# Regularização

A regularização é uma técnica usada para evitar o **overfitting**. O overfitting acontece quando um modelo de machine learning se ajusta _tão bem_ aos dados de treinamento que acaba "decorando" as informações, em vez de aprender os padrões gerais. Isso faz com que o modelo tenha um desempenho ruim com dados novos que ele nunca viu antes.

Na Regressão Linear padrão, o objetivo é encontrar a linha (ou plano, em mais dimensões) que melhor se ajusta aos seus dados. O "melhor ajuste" é definido como a linha que minimiza a soma dos erros quadrados – basicamente, a soma das distâncias ao quadrado entre cada ponto de dado real e a linha prevista pelo modelo.

O problema (o overfitting) acontece quando o modelo, na ânsia de minimizar esse erro, se contorce demais para chegar perto de todos os pontos de treinamento. Isso geralmente leva a **coeficientes com valores muito grandes**, o que torna o modelo super sensível a pequenas variações nos dados.

É aí que a regularização entra na Regressão Linear. Ela modifica a função de custo do modelo. Em vez de apenas minimizar o erro, o modelo agora precisa minimizar:

**Erro + Penalidade**

Essa **penalidade** está diretamente ligada ao tamanho dos coeficientes do modelo. Ou seja, o modelo é penalizado por ter coeficientes grandes.

Pense nisso como um balanço:

- O modelo ainda quer se ajustar bem aos dados (minimizando o **erro**).
- Mas agora ele também é incentivado a manter seus coeficientes pequenos (minimizando a **penalidade**).

Isso força o modelo a encontrar uma solução mais "simples" e menos extrema, o que o ajuda a generalizar melhor para dados que ele nunca viu.

## L2 (Ridge Regression)

A regularização L2 combate o sobreajuste ao adicionar um termo de penalidade à função de custo da regressão linear. Essa penalidade é proporcional à soma dos quadrados dos coeficientes do modelo. A nova função de custo a ser minimizada se torna:

$$
\text{Custo} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2
$$

Onde:
- $\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ é a soma dos erros quadrados (o custo original da regressão linear).
- $\lambda$ é o **parâmetro de regularização**, um hiperparâmetro que controla a intensidade da penalidade.
- $\sum_{j=1}^{p} \beta_j^2$ é a penalidade L2, a soma dos quadrados de todos os coeficientes do modelo (exceto o intercepto, $\beta_0$, que geralmente não é penalizado).

### O Papel do Parâmetro de Regularização (λ)

O parâmetro `λ` (em algumas bibliotecas, como Scikit-learn, é chamado de `alpha`) é crucial para o funcionamento da Regressão de Ridge. Ele determina o quão forte será a penalização sobre os coeficientes:

* **Se `λ = 0`**: A penalidade é nula, e a Regressão de Ridge se torna idêntica à regressão linear simples. O modelo terá alta variância e potencial para sobreajuste.
* **Se `λ` é pequeno**: A penalidade é leve, resultando em uma pequena "encolhida" dos coeficientes em direção a zero.
* **Se `λ` é grande**: A penalidade é severa, forçando os coeficientes a se aproximarem significativamente de zero. Isso resulta em um modelo mais simples (com menor variância), mas pode introduzir um viés (bias) se o valor for excessivamente alto, levando ao subajuste (underfitting).

A escolha do valor ideal de `λ` é tipicamente realizada através de técnicas como a **validação cruzada (cross-validation)**, que avalia o desempenho do modelo em diferentes subconjuntos dos dados para encontrar o valor que melhor generaliza.

### Consequências e Vantagens da Regularização L2

A principal consequência da adição da penalidade L2 é a **redução da magnitude dos coeficientes**. No entanto, é importante notar que a Regressão de Ridge **não zera completamente os coeficientes**, apenas os aproxima de zero. Isso significa que todas as variáveis preditoras são mantidas no modelo final, embora a sua influência seja atenuada.

As principais vantagens da utilização da regularização L2 são:

* **Prevenção do Sobreajuste:** Ao penalizar coeficientes grandes, a L2 impede que o modelo se ajuste excessivamente aos dados de treinamento.
* **Lida com a Multicolinearidade:** Em cenários com variáveis altamente correlacionadas, a regressão linear simples pode atribuir coeficientes muito grandes e instáveis. A Regressão de Ridge distribui o peso entre essas variáveis correlacionadas, tornando o modelo mais estável.
* **Melhora a Generalização:** Modelos regularizados com L2 tendem a ter um desempenho mais consistente em dados nunca antes vistos.

Em resumo, a regularização L2 é uma ferramenta poderosa e amplamente utilizada para tornar os modelos de regressão linear mais robustos e confiáveis, especialmente em cenários com alta dimensionalidade e multicolinearidade. Ao introduzir um "custo" para a complexidade, ela encontra um equilíbrio entre o ajuste aos dados de treinamento e a capacidade de fazer previsões precisas em novos dados.

## L1 (Lasso Regression)

A **regularização L1**, mais conhecida como **Regressão LASSO (Least Absolute Shrinkage and Selection Operator)**, é uma técnica poderosa utilizada para combater o sobreajuste (overfitting) em modelos de regressão linear. Sua principal característica, que a diferencia da regularização L2, é a capacidade de realizar **seleção de variáveis** de forma automática.

### O Princípio da Regularização L1

A Regressão LASSO funciona de maneira semelhante à Regressão de Ridge (L2), adicionando um termo de penalidade à função de custo da regressão linear. A diferença crucial está na forma como essa penalidade é calculada: a L1 utiliza a soma dos **valores absolutos** dos coeficientes.

A função de custo a ser minimizada na Regressão LASSO é:

$$
\text{Custo} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|
$$

Onde:
- $\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ é a soma dos erros quadrados (o termo de custo tradicional).
- $\lambda$ é o parâmetro de regularização, que controla a força da penalidade.
- $\sum_{j=1}^{p} |\beta_j|$ é a **penalidade L1**, a soma dos valores absolutos (ou magnitude) de todos os coeficientes do modelo.

### A Mágica da Seleção de Variáveis

A utilização do valor absoluto na penalidade tem uma consequência prática muito importante. À medida que o valor de `λ` aumenta, a Regressão LASSO não apenas "encolhe" os coeficientes em direção a zero, como a L2 faz, mas ela pode **reduzir alguns coeficientes a exatamente zero**.

Quando o coeficiente de uma variável preditora se torna zero, é como se a variável fosse completamente removida do modelo. Por isso, a LASSO é dita como uma técnica que produz **modelos esparsos** (sparse models), ou seja, modelos que utilizam apenas um subconjunto das variáveis originais.

### O Papel do Parâmetro de Regularização (λ)

O parâmetro `λ` (ou `alpha` em bibliotecas como Scikit-learn) é fundamental para controlar o comportamento do modelo:

* **Se `λ = 0`**: Não há penalidade. O modelo é uma regressão linear simples, com risco de sobreajuste.
* **Se `λ` aumenta**: A penalidade se torna mais forte. Mais coeficientes são forçados a se tornar exatamente zero. O modelo se torna mais simples.
* **Se `λ` é muito grande**: A penalidade é tão forte que todos os coeficientes podem se tornar zero, resultando em um modelo extremamente simples (e provavelmente inútil).

A escolha do `λ` ideal é um ato de equilíbrio, geralmente encontrado via **validação cruzada (cross-validation)**.

### Vantagens e Aplicações da Regularização L1

As principais vantagens da Regressão LASSO são:

1.  **Seleção Automática de Variáveis:** Em cenários com um grande número de variáveis, a LASSO pode identificar e descartar as menos importantes, simplificando o modelo.
2.  **Melhora a Interpretabilidade do Modelo:** Ao criar modelos esparsos, a LASSO nos entrega um resultado final mais simples e fácil de explicar, pois se concentra apenas nas variáveis mais influentes.
3.  **Redução do Sobreajuste:** Assim como a L2, ela efetivamente combate o overfitting ao restringir a magnitude dos coeficientes.

Devido a essas características, a LASSO é especialmente útil em campos como bioinformática (análise de genes), finanças e economia, onde se deseja não apenas prever, mas também entender quais fatores são os mais determinantes.

---
# Métricas de Avaliação

Após construir um modelo de regressão, a etapa crucial seguinte é avaliar sua performance. Para isso, utilizamos métricas de avaliação que quantificam o quão precisas são as previsões do modelo em comparação com os valores reais.

A escolha da métrica correta depende do contexto do problema e do que se considera um "bom" resultado. Abaixo estão as métricas mais importantes e utilizadas.
## 1. Erro Absoluto Médio (MAE - Mean Absolute Error)

É uma das métricas mais simples e intuitivas. O MAE calcula a média das diferenças absolutas entre os valores previstos e os valores reais.

**Fórmula:**
$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$
Onde:
- `n` é o número de observações.
- `yᵢ` é o valor real.
- `ŷᵢ` é o valor previsto.

- **Interpretação:** "Em média, as previsões do meu modelo erram por X unidades".
- **Unidade:** A mesma da variável alvo (ex: R$, kg, anos). Isso a torna muito fácil de interpretar.
- **Vantagens:** É robusta a outliers (erros muito grandes), pois não eleva os erros ao quadrado.
- **Desvantagens:** Não penaliza erros grandes com mais intensidade.

## 2. Erro Quadrático Médio (MSE - Mean Squared Error)

O MSE calcula a média dos erros ao quadrado. Ao elevar o erro ao quadrado, ele dá um peso muito maior para erros grandes.

**Fórmula:**
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

- **Interpretação:** É mais difícil de interpretar diretamente, pois sua unidade é o quadrado da unidade da variável alvo (ex: R$², kg²).
- **Vantagens:** Penaliza fortemente os outliers ou erros grandes. É a função de custo mais comum para otimização de modelos de regressão.
- **Desvantagens:** Sensível a outliers e a unidade ao quadrado dificulta a comunicação do resultado.

## 3. Raiz do Erro Quadrático Médio (RMSE - Root Mean Squared Error)

O RMSE é simplesmente a raiz quadrada do MSE, trazendo a unidade de medida de volta para a mesma da variável alvo.

**Fórmula:**
$$
RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

- **Interpretação:** Pode ser entendido como o desvio padrão dos resíduos (erros de previsão).
- **Unidade:** A mesma da variável alvo (ex: R$, kg, anos).
- **Vantagens:** É interpretável e, ao mesmo tempo, penaliza erros grandes com mais intensidade que o MAE.
- **Desvantagens:** Assim como o MSE, é sensível a outliers.

**Dica:** Se o RMSE for muito maior que o MAE, isso indica a presença de erros pontuais muito grandes.

## 4. Coeficiente de Determinação (R² ou R-squared)

O R² mede o **poder explicativo** do modelo. Ele representa a proporção da variância da variável alvo que é explicada pelas variáveis preditoras.

**Fórmula:**
$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$
Onde `ȳ` é a média dos valores reais.

- **Interpretação:** Varia de 0 a 1 (0% a 100%). Um R² de 0.82 significa que o modelo explica 82% da variabilidade dos dados.
- **Vantagens:** Fornece uma medida relativa da "qualidade do ajuste".
- **Desvantagens:** O valor de R² sempre aumenta ou se mantém ao adicionar novas variáveis, mesmo que sejam inúteis, podendo levar ao sobreajuste.

## 5. R² Ajustado (Adjusted R-squared)

O R² Ajustado corrige a principal falha do R², penalizando a adição de variáveis preditoras que não melhoram o modelo de forma significativa.

- **Interpretação:** Similar ao R², mas leva em conta o número de variáveis no modelo. Será sempre menor ou igual ao R².
- **Vantagens:** É uma métrica muito mais confiável para comparar modelos com um número diferente de variáveis.
- **Quando usar?** Sempre que estiver comparando modelos com diferentes conjuntos de preditores.

## Resumo e Qual Métrica Escolher?

| Métrica         | O que Mede                       | Unidades      | Sensibilidade a Outliers | Principal Vantagem                             |
| :-------------- | :------------------------------- | :------------ | :----------------------- | :--------------------------------------------- |
| **MAE**         | Erro absoluto médio              | Mesma do alvo | Baixa                    | Fácil de interpretar e comunicar               |
| **RMSE**        | Raiz do erro quadrático médio    | Mesma do alvo | Alta                     | Penaliza erros grandes; popular                |
| **R²**          | Proporção da variância explicada | Nenhuma       | Média                    | Mede o poder explicativo do modelo             |
| **R² Ajustado** | Proporção da variância explicada | Nenhuma       | Média                    | Compara modelos com nº de variáveis diferentes |

A melhor prática é nunca confiar em uma única métrica. Analise um conjunto delas para ter uma visão completa e robusta da performance do seu modelo.

---
# Regressão Multivariada

Na estatística formal, a Regressão Multivariada (ou *Multivariate Multiple Regression*) é um modelo mais avançado que lida com:

* **Múltiplas variáveis preditoras (`X`)**.
* **Múltiplas variáveis alvo (`Y`)**.

Neste caso, o modelo tenta prever um vetor de resultados simultaneamente, levando em conta que as variáveis alvo podem estar correlacionadas entre si.

O Modelo Em vez de uma única equação, temos um sistema de equações, uma para cada variável alvo: $$ Y_1 = \beta_{10} + \beta_{11} X_1 + \dots + \beta_{1p} X_p + \epsilon_1 $$ $$ Y_2 = \beta_{20} + \beta_{21} X_1 + \dots + \beta_{2p} X_p + \epsilon_2 $$ $$ \vdots $$ $$ Y_q = \beta_{q0} + \beta_{q1} X_1 + \dots + \beta_{qp} X_p + \epsilon_q $$ A principal característica aqui é que o modelo considera a correlação entre os termos de erro (`ε₁, ε₂, ..., εq`) para fazer previsões potencialmente melhores.

**Exemplo Prático:**
Prever simultaneamente a **taxa de inflação (`Y₁`)** e a **taxa de desemprego (`Y₂`)** com base no **crescimento do PIB (`X₁`)**. Como inflação e desemprego estão relacionados, modelá-los juntos pode ser mais eficaz.

---
# Regressão Não-Linear

A regressão não linear abrange uma vasta gama de modelos que são utilizados quando a relação entre as variáveis independentes ($X$) e a dependente ($Y$) não pode ser adequadamente descrita por uma linha reta.

É crucial fazer uma distinção importante logo de início:

1. **Modelos Não-Lineares nas Variáveis (mas lineares nos parâmetros):** A relação entre $X$ e $Y$ é uma curva, mas a equação do modelo ainda é uma combinação linear de seus coeficientes. Estes são os mais simples e muitas vezes são o primeiro passo além da regressão linear.
2. **Modelos Intrinsecamente Não-Lineares (não-lineares nos parâmetros):** A equação não pode ser escrita como uma soma ponderada dos coeficientes. Estes são os modelos não-lineares "verdadeiros" e geralmente requerem algoritmos de otimização mais complexos.

## Categoria 1: Modelos Lineares nos Parâmetros

Estes modelos usam o mesmo mecanismo de ajuste da regressão linear (como o MQO), mas aplicam-no a uma versão transformada das variáveis.

### 1. Regressão Polinomial

É a abordagem mais simples. Em vez de ajustar uma reta ($y = \beta_0 + \beta_1 x$), ajustamos uma curva polinomial, adicionando potências da variável independente como novos preditores.

- **Expressão (grau 2):** $y = \beta_0 + \beta_1 x + \beta_2 x^2 + \epsilon$
- **Como Funciona:** Embora a linha seja uma curva, a equação é linear em relação aos coeficientes ($\beta_0​,\beta_1​,\beta_2$​). Na prática, você cria novas colunas ($x^2,x^3$, etc.) e roda uma regressão linear múltipla padrão.
- **Uso:** Ótima para relações curvilíneas simples, como uma parábola.

### 2. Regressão com Splines

É uma versão mais flexível e poderosa da regressão polinomial. Em vez de ajustar uma única curva complexa a todos os dados, os splines dividem os dados em seções e ajustam uma curva mais simples (geralmente cúbica) a cada seção, conectando-as suavemente em pontos chamados "nós" (knots).

- **Como Funciona:** Permite que o modelo seja muito mais flexível, capturando tendências locais nos dados que uma única curva polinomial perderia.
- **Uso:** Muito utilizada em estatística para modelar relações complexas e desconhecidas sem sobreajustar.

### 3. Modelos Aditivos Generalizados (GAMs)

Os GAMs levam a ideia dos splines um passo adiante. Eles modelam a variável resposta como uma soma de funções suaves (não-lineares) de cada variável preditora.

- **Expressão:** $y = \beta_0 + f_1 (x_1) + f_2 (x_2) + \dots + f_p (x_p) + \epsilon$  
- **Como Funciona:** Cada função $f_j​(x_j​)$ é estimada de forma não-paramétrica (geralmente com splines). A grande vantagem é que você pode visualizar o efeito não-linear de cada variável individualmente, tornando o modelo muito interpretável.
- **Uso:** Excelente para quando você precisa de flexibilidade e interpretabilidade.

## Categoria 2: Modelos Intrinsecamente Não-Lineares

Estes modelos são definidos por equações que não são lineares nos seus parâmetros e requerem algoritmos de otimização iterativos (como o método de Newton ou Gradiente Descendente) para encontrar os melhores coeficientes.

## Tabela Resumo

| Modelo                | Como Cria a Não-Linearidade                            | Vantagens                                                      | Desvantagens                                                 |
| --------------------- | ------------------------------------------------------ | -------------------------------------------------------------- | ------------------------------------------------------------ |
| **Polinomial**        | Adiciona termos de potência ($x^2,x^3$)                | Simples de implementar, interpretável.                         | Pouco flexível, propenso a sobreajuste nas bordas.           |
| **Splines / GAMs**    | Conecta curvas simples em seções (nós).                | Muito flexível, interpretável, evita sobreajuste local.        | Mais complexo de implementar que a polinomial.               |
| **Árvores (RF, GBM)** | Particiona o espaço em regiões retangulares.           | Alta precisão, robusto, captura interações.                    | Menos interpretável ("caixa-preta"), pode sobreajustar.      |
| **Redes Neurais**     | Múltiplas camadas de funções de ativação não-lineares. | Extremamente flexível, estado da arte para dados complexos.    | Requer muitos dados, computacionalmente caro, "caixa-preta". |
| **SVR**               | "Truque do Kernel" para projetar em alta dimensão.     | Eficaz em alta dimensão, bom com diferentes tipos de relações. | Sensível à escolha do kernel e hiperparâmetros.              |
