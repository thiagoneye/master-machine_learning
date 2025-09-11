# Distribuições de Probabilidades: O que são e os principais tipos

Em estatística e probabilidade, uma **distribuição de probabilidade**, é um modelo matemático que descreve a frequência com que os diferentes valores possíveis de uma variável aleatória ocorrem. Em outras palavras, ela nos mostra o padrão de variação de um conjunto de dados, indicando quais resultados são mais ou menos prováveis.

A compreensão das distribuições é fundamental para a análise de dados, pois permite fazer inferências, previsões e tomar decisões baseadas na probabilidade de ocorrência de determinados eventos.

As distribuições de probabilidades são categorizadas principalmente em dois tipos, com base na natureza da variável em estudo: **distribuições discretas** e **distribuições contínuas**.

---
## Distribuições de Probabilidades Discretas

Uma variável discreta é aquela que pode assumir um número finito ou contável de valores distintos. Por exemplo, o número de caras ao lançar uma moeda três vezes (pode ser 0, 1, 2 ou 3) ou o número de e-mails que você recebe em uma hora. As principais distribuições discretas são:

### 1. Distribuição de Bernoulli
É a mais simples das distribuições e serve como base para muitas outras. Ela modela um único experimento com apenas dois resultados possíveis: "sucesso" (com probabilidade *p*) ou "fracasso" (com probabilidade 1-*p*). Um exemplo clássico é o lançamento de uma única moeda, onde "cara" pode ser o sucesso.

### 2. Distribuição Binomial
Esta distribuição descreve o número de sucessos em uma série de *n* ensaios de Bernoulli independentes e idênticos. Para que uma variável siga uma distribuição binomial, as seguintes condições devem ser atendidas:
* O número de ensaios é fixo.
* Cada ensaio tem apenas dois resultados possíveis (sucesso ou fracasso).
* A probabilidade de sucesso é constante para cada ensaio.
* Os ensaios são independentes.

**Exemplo:** A probabilidade de obter exatamente 7 caras em 10 lançamentos de uma moeda.

### 3. Distribuição de Poisson
A distribuição de Poisson é usada para modelar o número de vezes que um evento ocorre em um intervalo de tempo ou espaço específico, quando a taxa média de ocorrência é conhecida e os eventos são independentes.

**Exemplo:** O número de chamadas que um call center recebe por hora, ou o número de defeitos em um rolo de tecido.

### 4. Distribuição Geométrica
Esta distribuição modela o número de ensaios de Bernoulli necessários para se obter o primeiro sucesso.

**Exemplo:** O número de vezes que você precisa lançar um dado até obter o número 6 pela primeira vez.

## Distribuições de Probabilidades Contínuas

Uma variável contínua é aquela que pode assumir qualquer valor dentro de um determinado intervalo. Exemplos incluem a altura de uma pessoa, a temperatura de uma sala ou o tempo necessário para completar uma tarefa. Para variáveis contínuas, a probabilidade não é atribuída a um ponto específico, mas sim a um intervalo de valores.

### 1. Distribuição Normal (ou Gaussiana)
É a mais importante e amplamente utilizada das distribuições de probabilidade. A sua representação gráfica é a famosa "curva em forma de sino", que é simétrica em torno da sua média. Muitas variáveis naturais, como altura, peso e pressão arterial, tendem a seguir uma distribuição normal. O Teorema do Limite Central, um conceito fundamental em estatística, afirma que a soma (ou média) de um grande número de variáveis aleatórias independentes e identicamente distribuídas tenderá a ser normalmente distribuída, independentemente da distribuição original das variáveis.

### 2. Distribuição Uniforme
Na distribuição uniforme, todos os resultados em um determinado intervalo têm a mesma probabilidade de ocorrer. O gráfico desta distribuição é um retângulo.

**Exemplo:** A geração de um número aleatório entre 0 e 1, onde qualquer número tem a mesma chance de ser escolhido.

### 3. Distribuição Exponencial
Esta distribuição é usada para modelar o tempo até a ocorrência de um evento em um processo de Poisson. Ela está intimamente relacionada à distribuição de Poisson. Enquanto a de Poisson descreve o número de eventos em um intervalo, a exponencial descreve o tempo entre esses eventos.

**Exemplo:** O tempo de vida de um componente eletrônico ou o tempo entre a chegada de clientes em uma loja.

### 4. Distribuição Qui-Quadrado ($\chi^2$)
A distribuição Qui-Quadrado é derivada da distribuição normal e é amplamente utilizada em testes de hipóteses e na construção de intervalos de confiança. Ela é particularmente importante para testar a aderência de um conjunto de dados a uma distribuição esperada (teste de bondade de ajuste) e para testar a independência entre variáveis categóricas.

A escolha da distribuição correta para modelar um conjunto de dados é um passo crucial na análise estatística, pois as conclusões tiradas dependem fortemente do modelo de probabilidade assumido.

--- 
# A Transformação Box-Cox

A Transformação Box-Cox é uma poderosa ferramenta estatística utilizada para converter dados não-normais em um conjunto de dados com distribuição aproximadamente normal. Essa transformação é fundamental em diversas áreas da análise de dados, pois muitos modelos estatísticos, como a regressão linear e a análise de variância (ANOVA), pressupõem que os dados sigam uma distribuição normal. Ao estabilizar a variância e reduzir a assimetria dos dados, a transformação Box-Cox permite a aplicação correta e eficaz desses modelos, levando a conclusões mais robustas e precisas.

## O Que É e Para Que Serve?

Em sua essência, a Transformação Box-Cox é uma família de transformações de potência que busca o valor ideal de um parâmetro, denominado lambda ($\lambda$), para tornar um conjunto de dados o mais próximo possível de uma distribuição normal. A principal utilidade dessa transformação reside em corrigir duas violações comuns das premissas de modelos estatísticos:

* **Não-normalidade dos resíduos:** Em modelos de regressão, por exemplo, é crucial que os **resíduos** (as diferenças entre os valores observados e os previstos pelo modelo) sigam uma distribuição normal.
* **Heterocedasticidade:** Ocorre quando a variância dos resíduos não é constante ao longo dos diferentes níveis das variáveis independentes. A transformação Box-Cox pode ajudar a estabilizar essa variância, tornando-a mais homogênea (homocedasticidade).

## A Fórmula por Trás da Transformação

A transformação de Box-Cox é definida pela seguinte fórmula matemática:

$$y_i^{(\lambda)} = \begin{cases} \frac{y_i^\lambda - 1}{\lambda} & \text{se } \lambda \neq 0 \\ \ln(y_i) & \text{se } \lambda = 0 \end{cases}$$

Onde:

* $y_i$ é o dado original (que deve ser positivo).
* $\lambda$ é o parâmetro de transformação.

O objetivo é encontrar o valor de $\lambda$ que maximiza a normalidade dos dados transformados.

### Como Escolher o Parâmetro Lambda ($\lambda$)?

A seleção do valor ótimo de $\lambda$ geralmente é realizada por meio do método de máxima verossimilhança. Softwares estatísticos como R, Python, Minitab e outros, possuem funções que testam uma gama de valores para $\lambda$ (geralmente entre -5 e 5) e identificam aquele que resulta na melhor aproximação de uma distribuição normal.

Visualmente, a escolha do $\lambda$ pode ser auxiliada por um gráfico de log-verossimilhança, que mostra o valor da função de verossimilhança para diferentes valores de $\lambda$. O valor de $\lambda$ que corresponde ao pico da curva nesse gráfico é o valor ótimo.

### Interpretando os Valores de Lambda

Diferentes valores de $\lambda$ correspondem a diferentes transformações. Alguns dos valores mais comuns e suas interpretações são:

| Valor de Lambda ($\lambda$) | Transformação               | Efeito                             |
| :-------------------------- | :-------------------------- | :--------------------------------- |
| $\lambda = 2$               | Transformação Quadrática ($y^2$) | Corrige assimetria à esquerda moderada. |
| $\lambda = 1$               | Nenhuma transformação ($y$) | Os dados já são aproximadamente normais. |
| $\lambda = 0.5$             | Raiz Quadrada ($\sqrt{y}$)  | Corrige assimetria à direita moderada.  |
| $\lambda = 0$               | Logarítmica Natural ($\ln(y)$) | Corrige assimetria à direita forte.     |
| $\lambda = -0.5$            | Recíproca da Raiz Quadrada ($1/\sqrt{y}$) | Corrige assimetria à direita mais forte. |
| $\lambda = -1$              | Recíproca ($1/y$)           | Corrige assimetria à direita muito forte. |

É importante notar que, embora a análise estatística seja realizada com os dados transformados, a interpretação final dos resultados (como os coeficientes de uma regressão) deve ser feita na escala original dos dados. Para isso, é necessário aplicar a transformação inversa.

## Aplicações Práticas

A transformação Box-Cox é amplamente utilizada em diversas áreas:

* **Análise de Regressão:** Para garantir que os resíduos do modelo sigam uma distribuição normal e tenham variância constante, melhorando a validade dos testes de hipóteses e dos intervalos de confiança para os coeficientes do modelo.
* **Análise de Séries Temporais:** Para estabilizar a variância ao longo do tempo, um pré-requisito para a aplicação de modelos como o ARIMA (Autoregressive Integrated Moving Average).
* **Controle Estatístico de Processo (CEP):** Para normalizar dados de processos industriais, permitindo o uso de gráficos de controle que assumem normalidade.
* **Ciências Sociais e Econômicas:** Para analisar variáveis como renda, população e outros indicadores que frequentemente apresentam distribuições assimétricas.

## Pressupostos e Limitações

Apesar de sua utilidade, a transformação Box-Cox possui algumas premissas e limitações importantes:

* **Dados Positivos:** A transformação só pode ser aplicada a dados estritamente positivos (maiores que zero). Caso os dados contenham zeros ou valores negativos, é necessário adicionar uma constante a todos os dados antes de aplicar a transformação para torná-los positivos.
* **Dados Contínuos:** A transformação é mais adequada para variáveis contínuas.
* **Perda de Interpretabilidade:** A transformação altera a escala original dos dados, o que pode dificultar a interpretação direta dos resultados. É crucial reverter a transformação para comunicar as conclusões de forma clara.
* **Não Garante a Normalidade:** Em alguns casos, mesmo após a aplicação da transformação Box-Cox, os dados podem não atingir uma distribuição perfeitamente normal.

Para situações em que os dados contêm valores não-positivos, uma alternativa é a **transformação de Yeo-Johnson**, que é uma extensão da Box-Cox e pode ser aplicada a qualquer dado contínuo.

Em resumo, a transformação Box-Cox é uma técnica valiosa no arsenal de qualquer analista de dados, permitindo a adequação dos dados a pressupostos de importantes modelos estatísticos e, consequentemente, a obtenção de análises mais fidedignas e perspicazes.

## A Transformação de Box-Cox de Dois Parâmetros

Uma extensão direta do método original, a transformação de Box-Cox de dois parâmetros introduz um parâmetro de deslocamento, $\lambda_2$, além do parâmetro de potência original, $\lambda_1$​. A fórmula é aplicada a $y + \lambda_2$, o que permite que a transformação seja utilizada em dados que contenham valores negativos ou zero, desde que $y+\lambda_2​>0$.

A fórmula é definida como:

$$y_i^{(\lambda_1, \lambda_2)} = \begin{cases} \frac{(y_i + \lambda_2)^{\lambda_1} - 1}{\lambda} & \text{se } \lambda_1 \neq 0 \\ \ln(y_i + \lambda_2) & \text{se } \lambda_1 = 0 \end{cases}$$

Essa variante oferece mais flexibilidade, mas a estimação de dois parâmetros pode ser computacionalmente mais intensiva.

## A Transformação de Yeo-Johnson: Uma Solução Abrangente

Provavelmente a alternativa mais popular e amplamente utilizada, a transformação de Yeo-Johnson (2000) tem a vantagem de ser aplicável a dados que incluem valores positivos, negativos e zero, sem a necessidade de um parâmetro de deslocamento. Ela utiliza uma abordagem engenhosa, com fórmulas ligeiramente diferentes para dados não-negativos e negativos, garantindo que a transformação seja bem definida em todo o domínio dos números reais.

A transformação de Yeo-Johnson é definida da seguinte forma:

- **Para $y \ge 0$:**
$$y_i^{(\lambda)} = \begin{cases} \frac{(y_i + 1)^\lambda - 1}{\lambda} & \text{se } \lambda \neq 0 \\ \ln(y_i + 1) & \text{se } \lambda = 0 \end{cases}$$
- **Para $y < 0$:**
$$y_i^{(\lambda)} = \begin{cases} \frac{-((y_i + 1)^{2-\lambda} - 1)}{\lambda} & \text{se } \lambda \neq 2 \\ -\ln(-y_i + 1) & \text{se } \lambda = 2 \end{cases}$$

Essa flexibilidade torna a transformação de Yeo-Johnson uma escolha robusta em muitas situações práticas de análise de dados.

## A Transformação de Manly

Proposta por Manly (1976), esta transformação utiliza uma função exponencial e é particularmente eficaz para corrigir a assimetria em distribuições unimodais. Assim como a Yeo-Johnson, ela também pode ser aplicada a dados com valores negativos.

A fórmula da transformação de Manly é:

$$y_i^{(\lambda)} = \begin{cases} \frac{e^{\lambda y_i} - 1}{\lambda} & \text{se } \lambda \neq 0 \\ y_i & \text{se } \lambda = 0 \end{cases}$$

## Outras Extensões e Alternativas Relevantes

- **Transformação "Transform-Both-Sides" (TBS):** Em contextos de regressão, em vez de transformar apenas a variável dependente ($Y$), o modelo TBS aplica a mesma transformação de potência tanto em $Y$ quanto nas variáveis independentes ($X$). O objetivo é linearizar a relação entre as variáveis e, ao mesmo tempo, estabilizar a variância dos erros do modelo.
- **Transformação de Johnson:** Trata-se de uma família de transformações, e não de uma única. O sistema de Johnson é capaz de se ajustar a uma vasta gama de distribuições não normais, sendo uma alternativa poderosa quando as transformações de potência mais simples não alcançam a normalidade desejada.
- **Transformação de Box-Cox Espacial:** Utilizada no campo da geoestatística, esta variante permite que o parâmetro de transformação ($\lambda$) varie espacialmente. Isso é útil em cenários onde a relação entre a média e a variância dos dados muda em diferentes localizações geográficas.