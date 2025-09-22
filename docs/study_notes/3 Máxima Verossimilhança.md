# A Máxima Verossimilhança

A máxima verossimilhança (ou "maximum likelihood" em inglês) é um método amplamente utilizado na estatística para estimar os parâmetros de um modelo estatístico. A ideia central é encontrar os valores dos parâmetros que tornam a probabilidade de observar os dados que você realmente tem a maior possível.

Vamos pensar em um exemplo simples para entender a intuição por trás disso.

Imagine que você tem uma moeda e quer saber se ela é honesta (probabilidade de dar cara, $p$, igual a $0.5$) ou viciada. Você joga a moeda $10$ vezes e obtém $7$ caras e $3$ coroas.

Qual é o valor de $p$ (a probabilidade de sair cara) que melhor explica esse resultado?

- Se a moeda for honesta ($p=0.5$), a probabilidade de obter $7$ caras em $10$ lançamentos é uma.
- Se a moeda for viciada e $p=0.7$, a probabilidade de obter $7$ caras em $10$ lançamentos é outra, provavelmente maior.    
- Se a moeda for muito viciada e $p=0.9$, a probabilidade de obter $7$ caras é menor, porque é mais provável que saiam mais caras.

A **função de verossimilhança** quantifica essa probabilidade. Ela é uma função dos parâmetros do modelo (neste caso, $p$), dados os seus resultados observados.

O **método da máxima verossimilhança** busca o valor de $p$ que maximiza essa função de verossimilhança. Intuitivamente, para o nosso exemplo, o valor que torna o resultado de $7$ caras e $3$ coroas o "mais provável" de acontecer é $p=0.7$.

## Por que é tão importante?

A máxima verossimilhança é uma ferramenta poderosa e popular por várias razões:

- **Intuição:** A ideia por trás dela é bastante intuitiva: encontre os parâmetros que tornam seus dados observados mais prováveis.
- **Propriedades estatísticas:** Os estimadores de máxima verossimilhança (os valores dos parâmetros que você encontra) têm propriedades estatísticas muito desejáveis, como consistência (à medida que você coleta mais dados, o estimador se aproxima do valor real) e eficiência (eles têm a menor variância possível em comparação com outros estimadores não viciados).    
- **Ampla aplicabilidade:** Pode ser usada para uma vasta gama de modelos estatísticos, desde modelos de regressão linear e logística até modelos mais complexos.

## Como funciona?

O processo funciona em algumas etapas principais:

1.  **Escolha de um Modelo (Distribuição de Probabilidade):** Primeiro, assume-se que os dados observados foram gerados por uma distribuição de probabilidade específica (por exemplo, Normal, Binomial, Poisson), mas com parâmetros desconhecidos. Para uma distribuição Normal, por exemplo, os parâmetros são a média ($\mu$) e a variância ($\sigma^2$).

2.  **Construção da Função de Verossimilhança:** Constrói-se uma função, chamada **Função de Verossimilhança** ($L(\theta | \mathbf{x})$). Essa função representa a probabilidade conjunta de observar o conjunto de dados $\mathbf{x}$ para um determinado conjunto de parâmetros $\theta$. Se as observações são independentes e identicamente distribuídas (i.i.d.), essa função é o produto das probabilidades de cada observação individual.

3.  **Maximização da Função:** O passo crucial é encontrar o valor dos parâmetros $\theta$ que maximiza essa função de verossimilhança. Encontrar o pico dessa função significa encontrar os parâmetros que tornam os dados observados o mais provável possível.

4.  **Uso da Log-Verossimilhança (Log-Likelihood):** Na prática, lidar com produtos de probabilidades pode ser matematicamente complicado e numericamente instável. Por isso, é comum trabalhar com o **logaritmo da função de verossimilhança**, conhecido como *log-likelihood*. Como o logaritmo é uma função monotonicamente crescente, maximizar o logaritmo é equivalente a maximizar a própria função. O logaritmo transforma produtos em somas, o que simplifica muito os cálculos.

5.  **Otimização:** A maximização é geralmente feita usando cálculo diferencial. Deriva-se a função de log-verossimilhança em relação a cada parâmetro, iguala-se as derivadas a zero e resolve-se o sistema de equações para encontrar os valores ótimos.

## Fundamentos

Suponha-se que se tenha uma amostra $x_1, x_2, \dots, x_n$ de $n$ observações independentes e identicamente distribuídas extraídas de uma função de distribuição desconhecida com função de densidade (ou função de probabilidade) $f_0(·)$. Se sabe, porém, que $f_0$ pertence a uma família de distribuições $\{ f(·|\theta), \theta \in \Theta\}$, chamada modelo paramétrico, de maneira que $f_0$ corresponde a $\theta = \theta _0$, que é o **verdadeiro valor** do parâmetro. Se deseja encontrar o valor $\theta$ (ou _estimador_) que esteja o mais próximo possível ao verdadeiro valor $\theta_0$.

Tanto $x_i$ como $\theta$ podem ser vetores.

A ideia desse método é encontrar primeiro a função densidade de todas as observações, que sob condições de independência, é

$$f(x_1, x_2, \dots, x_n | \theta) = f(x_1 | \theta) \times f(x_2 | \theta) \times \dots \times f(x_n | \theta)$$

Observando esta função sob um ângulo ligeiramente distinto, pode-se supor que os valores observados $x_1, x_2, \dots, x_n$ são fixos enquanto que $\theta$ pode variar livremente. Esta é a função de verossimilhança:

$$L(\theta | x) = \prod _{x=1} ^n p(x_i | \theta)$$

Na prática, é geralmente usado o logaritmo dessa função:

$$l (\theta | x_1, x_2, \dots, x_n) = \ln L = \Sigma _{i=1} ^{n} \ln f(x_i | \theta)$$

O método da **máxima verossimilhança** estima $\theta_0$ buscando o valor de $\theta$ que maximiza $l(\theta | x).$ Este é o chamado **estimador de máxima verossimilhança** (**MLE**) de $\theta _0$:

$$\theta_{mle} = \arg \max _{\theta \in \Theta} l (\theta | x_1, x_2, \dots, x_n)$$
 
Às vezes, esse estimador é uma função explícita dos dados observados $x_1, x_2, \dots, x_n$, mas muitas vezes se precisa recorrer à otimizações numéricas. Também pode acontecer que o máximo não seja único ou não exista.

Para encontrar esse máximo, geralmente resolvemos a **equação de verossimilhança**: $$\frac{\partial \ell(\theta | \mathbf{x})}{\partial \theta} = 0$$

## Melhores Abordagens para Otimização da Verossimilhança

A otimização da função de verossimilhança é um problema central em estatística e machine learning. A "melhor" abordagem depende das características do problema, como a complexidade da função, o número de parâmetros, o tamanho do dataset e a disponibilidade das derivadas.

### 1. Métodos Baseados em Gradiente (Gradient-Based Methods)

Estes métodos usam a informação da derivada (gradiente) para encontrar a direção de "subida" mais íngreme e alcançar o pico da função de verossimilhança.

#### a) Métodos de Segunda Ordem (Usam a Hessiana)

Estes métodos usam tanto a primeira derivada (Gradiente/Score) quanto a segunda (Matriz Hessiana).

* **Método de Newton (ou Newton-Raphson):**
    * **Como funciona:** Aproxima a função por uma parábola a cada passo e salta diretamente para o seu topo.
    * **Vantagens:** Convergência extremamente rápida (quadrática) perto do máximo.
    * **Desvantagens:** Exige o cálculo e a inversão da Matriz Hessiana a cada iteração, o que é computacionalmente caro para muitos parâmetros.
    * **Quando usar:** Em problemas estatísticos clássicos com um número moderado de parâmetros.
* **Métodos Quasi-Newton:**
    * **Como funciona:** Constroem uma **aproximação** da Hessiana iterativamente, usando apenas o gradiente.
    * **Exemplos:** **BFGS** (Broyden–Fletcher–Goldfarb–Shanno) e sua variante de memória limitada, **L-BFGS**.
    * **Vantagens:** Excelente equilíbrio entre velocidade de convergência e custo computacional.
    * **Quando usar:** **Frequentemente a melhor escolha padrão para a maioria dos problemas de otimização estatística.** É o algoritmo padrão em muitos pacotes como `SciPy` e `R`.

#### b) Métodos de Primeira Ordem (Usam apenas o Gradiente)

* **Gradiente Ascendente Estocástico (Stochastic Gradient Ascent - SGA):**
    * **Como funciona:** Calcula o gradiente usando um pequeno lote de dados (mini-batch) em vez do dataset completo a cada iteração.
    * **Vantagens:** Extremamente eficiente para datasets massivos ("Big Data").
    * **Desvantagens:** As atualizações são "ruidosas", exigindo um ajuste cuidadoso da taxa de aprendizagem.
    * **Quando usar:** **Abordagem dominante em Deep Learning.** Variantes como **Adam**, **RMSprop** e **Adagrad** são os otimizadores padrão.

### 2. O Algoritmo de Expectation-Maximization (EM)

Esta é uma abordagem especializada para problemas de máxima verossimilhança com **variáveis latentes (ocultas)** ou **dados ausentes**.

* **Como funciona:** Alterna entre dois passos:
    1.  **Passo E (Expectation):** Estima as variáveis latentes com base nos parâmetros atuais.
    2.  **Passo M (Maximization):** Atualiza os parâmetros usando MLE, assumindo que as variáveis latentes estimadas são as verdadeiras.
* **Vantagens:** Transforma um problema intratável numa sequência de passos mais simples. Garante o aumento da verossimilhança a cada iteração.
* **Quando usar:** É a ferramenta certa para modelos como **Misturas de Gaussianas (GMMs)** e **Modelos Ocultos de Markov (HMMs)**.

### 3. Métodos Sem Derivadas (Derivative-Free Methods)

* **Como funciona:** Usados quando a derivada da função de verossimilhança é desconhecida ou intratável. Eles exploram o espaço de parâmetros de forma mais heurística.
* **Exemplos:** Nelder-Mead, Powell's Method.
* **Vantagens:** Podem ser aplicados a problemas de "caixa-preta" (black-box).
* **Desvantagens:** Geralmente muito mais lentos e menos eficientes.
* **Quando usar:** Como último recurso, quando as derivadas não estão disponíveis.

### Recomendações e Resumo

| Cenário                                              | Melhor Abordagem (geralmente)                             |
| :--------------------------------------------------- | :-------------------------------------------------------- |
| **Problemas estatísticos padrão** (parâmetros: baixo a médio) | **Quasi-Newton (L-BFGS)** |
| **Big Data / Deep Learning** (parâmetros: alto a massivo) | **Gradiente Estocástico e suas variantes (Adam, etc.)** |
| **Problemas com dados ausentes ou variáveis latentes** | **Algoritmo Expectation-Maximization (EM)** |
| **Verossimilhança complexa, sem derivadas fáceis** | **Métodos Sem Derivadas (ex: Nelder-Mead)** |

## Cenários Ideais para Usar MLE

1.  **Estimar Parâmetros de uma Distribuição de Probabilidade**
    * **Descrição:** Este é o caso de uso mais clássico. Se acredita que os seus dados seguem uma distribuição específica (Normal, Exponencial, Binomial, Poisson, etc.), a MLE é o método padrão para estimar os parâmetros dessa distribuição.
    * **Exemplo:** Mediu a altura de 1000 pessoas e assume que a altura segue uma distribuição Normal. A MLE encontrará os valores mais prováveis para a média (`μ`) e o desvio padrão (`σ`).
2.  **Como Base para Modelos de Regressão e Classificação**
    * **Descrição:** Muitos algoritmos de Machine Learning usam a MLE como o princípio para otimizar os seus parâmetros.
    * **Exemplos:**
        * **Regressão Linear:** Minimizar o Erro Quadrático Médio (MSE) é equivalente a maximizar a verossimilhança, assumindo que os erros são normalmente distribuídos.
        * **Regressão Logística:** O modelo é ajustado encontrando os coeficientes que maximizam a verossimilhança dos rótulos de classe observados.
3.  **Quando Propriedades Teóricas Sólidas são Importantes**
    * **Descrição:** A MLE é muito popular porque, sob certas condições e com amostras grandes (propriedades assintóticas), os seus estimadores são excelentes.
    * **Propriedades:**
        * **Consistência:** Com o aumento do tamanho da amostra, a estimativa da MLE converge para o valor verdadeiro do parâmetro.
        * **Eficiência Assintótica:** Para amostras grandes, nenhum outro estimador insesgado tem uma variância menor.
        * **Normalidade Assintótica:** As estimativas seguem aproximadamente uma distribuição Normal em amostras grandes, o que é útil para construir intervalos de confiança.
4.  **Quando se Necessita de um Framework Flexível e Geral**
    * **Descrição:** O princípio da MLE pode ser aplicado a uma vasta gama de problemas, desde modelos simples até sistemas muito complexos, como em análise de sobrevivência e modelos de séries temporais.

### Quando Ser Cauteloso ou Usar Alternativas

Existem situações em que a MLE pode não ser a melhor escolha:

1.  **Amostras Pequenas**
    * **Problema:** Com poucos dados, a MLE pode produzir estimativas enviesadas (biased) ou com alta variância. As garantias assintóticas ainda não se aplicam.
2.  **Risco de Overfitting**
    * **Problema:** Se o seu modelo tem muitos parâmetros em comparação com a quantidade de dados, a MLE pode ajustar-se excessivamente ao ruído.
    * **Alternativa:** Métodos de regularização (como Ridge ou Lasso), que podem ser vistos como uma **Estimativa de Máximo a Posteriori (MAP)** de uma perspetiva Bayesiana.
3.  **Quando se Tem Conhecimento Prévio Forte sobre os Parâmetros**
    * **Problema:** A MLE é uma abordagem *frequentista* e não incorpora crenças prévias sobre os parâmetros.
    * **Alternativa:** A **Inferência Bayesiana** é a abordagem ideal aqui, pois combina o conhecimento prévio com a verossimilhança dos dados.
4.  **Quando o Modelo Probabilístico é Desconhecido ou Intratável**
    * **Problema:** Se não se tem ideia de qual distribuição os seus dados seguem, ou se a função de verossimilhança é muito complexa para ser escrita.
    * **Alternativas:** Métodos não paramétricos ou o **Método dos Momentos**.

### Tabela Resumo

| Use MLE Quando...                                                   | Considere Alternativas Quando...                          |
| :------------------------------------------------------------------ | :-------------------------------------------------------- |
| ✅ Pode assumir um modelo probabilístico para os dados.             | ❌ O modelo probabilístico é desconhecido ou complexo.      |
| ✅ Precisa estimar parâmetros de distribuições conhecidas.          | ❌ A amostra de dados é muito pequena.                    |
| ✅ Está a implementar modelos como Regressão Logística.              | ❌ Há um alto risco de overfitting (muitos parâmetros).    |
| ✅ Tem uma amostra grande e deseja estimadores eficientes.          | ❌ Tem forte conhecimento prévio sobre os parâmetros.       |
| ✅ Propriedades teóricas (consistência, normalidade) são desejáveis. | ❌ A suposição de dados i.i.d. é violada gravemente.      |
