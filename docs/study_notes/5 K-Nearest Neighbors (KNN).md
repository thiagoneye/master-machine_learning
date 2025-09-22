# K-Nearest Neighbors (KNN)

O K-Nearest Neighbors (KNN), ou K-Vizinhos Mais Próximos, é um algoritmo utilizado tanto para tarefas de classificação quanto de regressão, sua simplicidade conceitual, aliada a uma eficácia surpreendente em diversos cenários, o torna um ponto de partida popular para muitos problemas de data science. Diferentemente de muitos outros algoritmos, o KNN é considerado um _"lazy learner"_ (aprendiz preguiçoso), pois não constrói um modelo explícito durante a fase de treinamento, mas sim armazena todo o conjunto de dados para utilizá-lo no momento da predição.

## O que é o K-Nearest Neighbors (KNN)?

O princípio fundamental do KNN reside na ideia de que pontos de dados semelhantes tendem a existir nas proximidades uns dos outros. Em outras palavras, um novo dado será classificado ou terá seu valor previsto com base na classe ou no valor de seus "vizinhos" mais próximos no conjunto de dados de treinamento.

O **"K"** no nome do algoritmo representa o número de vizinhos que serão considerados para a tomada de decisão. A escolha desse hiperparâmetro é crucial para o desempenho do modelo.

- **Para classificação:** um novo ponto de dados é atribuído à classe que é mais comum entre seus K vizinhos mais próximos (votação majoritária).
- **Para regressão:** o valor previsto para um novo ponto de dados é a média (ou mediana) dos valores de seus K vizinhos mais próximos.

## Formulação Matemática

A "vizinhança" no KNN é determinada por meio de métricas de distância. A escolha da métrica de [[3 Distâncias (Dissimilaridades)|distância]] adequada depende da natureza dos dados.

O processo de predição, matematicamente, pode ser resumido da seguinte forma:

1. **Cálculo das Distâncias:** Para um novo ponto de dado $x_{new}$, calcule a distância para todos os pontos $x_i$ no conjunto de treinamento usando uma das métricas de distância mencionadas.
2. **Identificação dos K-Vizinhos:** Ordene as distâncias calculadas e selecione os K pontos de dados do treinamento correspondentes às K menores distâncias.
3. **Predição:**
    - **Classificação:** A predição $\hat{y}_{new}$ é a classe mais frequente (moda) entre as classes dos K vizinhos.
	$$\hat{y}_{new} = \arg\max_{c} \sum_{i=1}^{K} I(y_i = c)$$
	onde $I$ é uma função indicadora que é 1 se a classe do vizinho $y_i$ for $c$, e 0 caso contrário.
	- **Regressão:** A predição $\hat{y}_{new}$ é a média dos valores dos K vizinhos.
$$\hat{y}_{new} = \frac{1}{K} \sum_{i=1}^{K} y_i$$

## Quais as Aplicações?

A versatilidade do KNN permite sua aplicação em uma vasta gama de áreas:

- **Sistemas de Recomendação:** Sugerir produtos, filmes ou músicas com base nas preferências de usuários com gostos semelhantes.
- **Reconhecimento de Padrões e Visão Computacional:** Classificação de imagens, como o reconhecimento de dígitos manuscritos ou a identificação de objetos.
- **Finanças:** Previsão de preços de ações, análise de risco de crédito e detecção de fraudes em transações.
- **Saúde:** Diagnóstico de doenças com base em características de pacientes e classificação de células como benignas ou malignas.
- **Genética:** Análise de expressão gênica e classificação de amostras com base em marcadores genéticos.
- **Detecção de Anomalias:** Identificar dados que se desviam significativamente do restante do conjunto, como em detecção de intrusão em redes.

## Aspectos Relevantes

Para a correta utilização do KNN, alguns pontos devem ser cuidadosamente considerados:

- **Escolha do Valor de K:** Um valor de K muito pequeno torna o modelo sensível a ruídos e outliers, podendo levar a um sobreajuste (overfitting). Um K muito grande pode suavizar demais os limites de decisão, resultando em um subajuste (underfitting). A escolha do K ideal é frequentemente realizada através de técnicas como a validação cruzada.
- **A Maldição da Dimensionalidade:** O desempenho do KNN tende a degradar em espaços de alta dimensão. Com muitas características, a distância entre os pontos pode se tornar menos significativa, e a noção de "vizinhança" perde seu sentido.
- **Escalonamento de Características (Feature Scaling):** Como o KNN se baseia em distâncias, características com escalas muito diferentes podem distorcer os cálculos. Por exemplo, uma característica que varia de 0 a 1000 terá um impacto muito maior na distância do que uma que varia de 0 a 1. Portanto, é crucial normalizar ou padronizar os dados antes de aplicar o algoritmo.

## Vantagens e Desvantagens

| Vantagens                                                                                                           | Desvantagens                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Simplicidade e Intuitividade:** Fácil de entender e implementar.                                                  | **Custo Computacional Elevado:** A necessidade de calcular a distância para todos os pontos do treinamento a cada nova predição o torna lento para grandes conjuntos de dados. |
| **Não Paramétrico:** Não faz suposições sobre a distribuição dos dados.                                             | **Sensível a Características Irrelevantes:** Características que não contribuem para a distinção das classes podem prejudicar o desempenho.                                    |
| **Fácil de Atualizar:** Novos dados podem ser adicionados sem retreinar o modelo.                                   | **Requer Grande Quantidade de Memória:** Como armazena todo o conjunto de dados, o consumo de memória pode ser um problema.                                                    |
| **Bom para Problemas Multiclasse:** Lida naturalmente com problemas com várias classes.                             | **Sensível à "Maldição da Dimensionalidade":** O desempenho diminui significativamente com o aumento do número de características.                                             |
| **Não requer fase de treinamento explícita:** Sendo um _"lazy learner"_, a computação principal ocorre na predição. | **Necessidade de Escalonamento de Dados:** O desempenho é altamente dependente da escala das características.                                                                  |

## Como Encontrar o K Ótimo no KNN

Encontrar o valor "K" ótimo é o passo mais crucial para garantir um bom desempenho do modelo KNN (K-Vizinhos Mais Próximos). Não existe um número mágico para o valor de K; ele depende inteiramente da estrutura e da complexidade do seu conjunto de dados.

A escolha de K é um equilíbrio conhecido como **trade-off entre viés e variância (bias-variance trade-off)**:

- **K pequeno (ex: K=1):** O modelo tem **baixa tendência (bias)**, mas **alta variância**. Ele se ajusta muito de perto aos dados de treinamento, tornando-se sensível a ruídos e outliers. Isso geralmente leva ao **superajuste (overfitting)**. A fronteira de decisão será muito irregular.
- **K grande (ex: K=30):** O modelo tem **alta tendência (bias)**, mas **baixa variância**. Ele é mais generalizado e menos afetado por pontos individuais, mas pode ignorar padrões locais importantes nos dados. Isso pode levar ao **subajuste (underfitting)**. A fronteira de decisão será muito suave.

O objetivo é encontrar o "ponto ideal" que minimize os erros em dados nunca vistos. Abaixo estão as técnicas mais comuns e eficazes para encontrar o K ótimo.

### Método 1: O Método do Cotovelo (Elbow Method) - O mais visual

Esta é a abordagem mais popular e intuitiva. A ideia é rodar o algoritmo KNN para um intervalo de valores de K e visualizar qual deles performa melhor.

**Passo a passo:**

1. **Divida seus dados:** Separe seu conjunto de dados em um conjunto de **treinamento** e um conjunto de **teste** (ou validação).
2. **Escolha um intervalo para K:** Selecione uma faixa de valores de K para testar. Por exemplo, de 1 até 25.
3. **Itere sobre K:** Para cada valor de K no intervalo:
    - Treine o modelo KNN usando os dados de treinamento.
    - Faça predições para os dados de teste.
    - Calcule a **taxa de erro** da predição.
4. **Plote o gráfico:** Crie um gráfico com os valores de K no eixo X e a taxa de erro correspondente no eixo Y.
5. **Encontre o "cotovelo":** Observe o gráfico. O ponto onde a taxa de erro para de cair drasticamente e começa a se estabilizar (formando um "cotovelo") é geralmente o valor de K ótimo. Esse ponto representa o melhor equilíbrio, pois aumentar K além dele não traz uma melhoria significativa no desempenho e pode começar a super-simplificar o modelo.

### Método 2: Validação Cruzada (Cross-Validation) - O mais robusto

O Método do Cotovelo é bom, mas depende de uma única divisão treino-teste, o que pode levar a resultados variáveis. A Validação Cruzada é uma técnica estatisticamente mais sólida para avaliar o desempenho do modelo.

**Passo a passo (usando K-Fold Cross-Validation):**

1. **Divida os dados em "folds" (partes):** Divida seu conjunto de dados de treinamento em, por exemplo, 10 partes iguais ("10-fold CV").
2. **Itere sobre os valores de K:** Para cada valor de K que você quer testar:
    - Execute 10 rodadas de treino/validação. Em cada rodada, use 9 partes para treinar o modelo e 1 parte para validar (testar).
    - Calcule a métrica de erro (ex: taxa de erro) para cada rodada.
    - Calcule a **média dos erros** das 10 rodadas. Isso lhe dará uma estimativa de desempenho muito mais estável para aquele valor de K.
3. **Selecione o melhor K:** O valor de K que resultar na menor taxa de erro média é considerado o ótimo.

### Método 3: Busca em Grade com Validação Cruzada (Grid Search CV) - O padrão ouro

Esta é a forma mais prática e automatizada de implementar a Validação Cruzada para encontrar o melhor hiperparâmetro. A maioria das bibliotecas de Machine Learning (como o Scikit-learn em Python) possui funções prontas para isso.

**Como funciona:**

1. **Defina a grade de busca:** Você cria uma "grade" de hiperparâmetros que deseja testar. Para o KNN, seria uma lista de valores de K (ex: `[1, 3, 5, 7, ..., 29]`).
2. **Instancie o Grid Search:** Você usa uma função como `GridSearchCV`, passando o modelo KNN, a grade de K's e o número de folds para a validação cruzada (ex: `cv=10`).
3. **Execute:** A função irá testar automaticamente cada valor de K, realizar a validação cruzada para cada um e registrar o desempenho.
4. **Obtenha o resultado:** Ao final, o objeto do Grid Search informará qual foi o **melhor valor de K** e qual foi o melhor desempenho alcançado.

### Dicas Práticas Adicionais

1. **Use um K ímpar para classificação binária:** Se K for par, você pode ter um empate (ex: 3 vizinhos de uma classe e 3 de outra), e o algoritmo terá que resolver isso de forma arbitrária. Usar um K ímpar (3, 5, 7...) evita esse problema.
2. **Regra de bolso (use com cautela):** Uma heurística comum é começar testando valores de K próximos da raiz quadrada do número de amostras no seu conjunto de treinamento (`K ≈ sqrt(N)`). **Isso não é uma regra**, mas pode ser um bom ponto de partida para definir seu intervalo de busca.
3. **Não se esqueça de escalar os dados!** Antes mesmo de começar a procurar o K ótimo, certifique-se de que seus dados numéricos estão na mesma escala (usando `StandardScaler` ou `MinMaxScaler`). O KNN é baseado em distância, e features com escalas maiores dominarão o cálculo, distorcendo o resultado, não importa qual K você escolha.

**Conclusão:** A maneira mais confiável e recomendada para encontrar o K ótimo é usar a **Busca em Grade com Validação Cruzada (Grid Search CV)**. Ela automatiza o processo robusto de testar vários valores de K e fornece o que tem o melhor desempenho médio no seu conjunto de dados.

## Variantes do Algoritmo K-Nearest Neighbors (KNN)

### Variantes para Eficiência e Escalabilidade

O KNN tradicional precisa calcular a distância para **todos** os pontos do conjunto de treino a cada nova previsão, o que o torna muito lento para grandes volumes de dados. Essas variantes otimizam a busca pelos vizinhos mais próximos.

* **K-D Tree (Árvore K-dimensional):** Esta é uma das otimizações mais famosas. A K-D Tree organiza os dados de treino em uma estrutura de árvore binária, particionando o espaço em hiper-retângulos. Isso permite podar galhos inteiros da árvore durante a busca, evitando o cálculo de muitas distâncias e acelerando drasticamente a consulta, especialmente em baixas dimensões (geralmente até umas 20 dimensões).
* **Ball Tree (Árvore de Bolas):** Similar à K-D Tree, mas mais eficaz em dimensões mais altas. Em vez de dividir o espaço com planos, a Ball Tree o particiona em "bolas" (hiperesferas) aninhadas. Ela funciona melhor quando os dados não estão distribuídos de forma retangular, sendo mais robusta à "maldição da dimensionalidade".
* **Locality-Sensitive Hashing (LSH):** Uma abordagem diferente e probabilística. Em vez de garantir a busca pelos vizinhos exatos, o LSH usa funções de hash para agrupar pontos similares em "baldes" (buckets). A busca é então restrita aos pontos no mesmo balde do novo dado. É uma técnica de *vizinhos mais próximos aproximados* (Approximate Nearest Neighbors - ANN), que troca uma pequena perda de precisão por um ganho massivo de velocidade.

### Variantes para Precisão e Robustez

Essas variantes modificam como o voto dos vizinhos é computado ou como os dados são tratados, tornando o modelo mais robusto a ruídos e a desbalanceamentos.

* **Weighted KNN (KNN Ponderado):** Talvez a variante mais importante e comum. Em vez de cada um dos *k* vizinhos ter um voto de igual peso, o **Weighted KNN** atribui um peso maior aos vizinhos mais próximos. Uma ponderação comum é o inverso da distância ($1/d$) ou o inverso do quadrado da distância ($1/d^2$). Isso faz com que a influência de um vizinho diminua à medida que ele se afasta, tornando a previsão menos sensível à escolha exata do valor de *k*.
* **Fuzzy KNN:** Em vez de atribuir uma única classe ao novo ponto, este método calcula um "grau de pertencimento" para cada classe. A decisão é baseada na distância e na classe dos vizinhos, resultando em uma classificação "suave" que pode ser útil em cenários onde as fronteiras entre as classes são incertas ou sobrepostas.
* **Edited Nearest Neighbor (ENN):** É mais uma **técnica de pré-processamento** para limpar o dataset. O ENN remove do conjunto de treino quaisquer pontos cuja classe seja diferente da classe majoritária de seus próprios vizinhos. O objetivo é **eliminar ruídos e suavizar as fronteiras de decisão** antes de treinar o modelo KNN final.
* **Condensed Nearest Neighbor (CNN):** Uma **técnica de pré-processamento** para **reduzir o tamanho do dataset**. O objetivo é encontrar um subconjunto mínimo de pontos do treino que ainda seja capaz de classificar corretamente todos os pontos originais. Isso torna o processo de previsão muito mais rápido, pois o KNN terá menos pontos para comparar.

### Resumo Comparativo

| Variante                 | Objetivo Principal   | Como Funciona                                                                | Vantagem Principal                                                |
| :----------------------- | :------------------- | :--------------------------------------------------------------------------- | :---------------------------------------------------------------- |
| **K-D Tree / Ball Tree** | Acelerar a busca     | Organiza os dados em estruturas de árvore para evitar buscas exaustivas.     | Muito mais rápido que o KNN padrão em datasets grandes.           |
| **Weighted KNN**         | Aumentar a precisão  | Dá mais peso (importância) aos vizinhos que estão mais perto.                | Reduz o impacto de vizinhos distantes e melhora a robustez.       |
| **ENN / CNN**            | Limpar/Reduzir dados | Técnicas de pré-processamento que removem ruídos ou pontos redundantes.      | Melhora a precisão (ENN) e/ou a velocidade (CNN) do modelo final. |
| **Fuzzy KNN**            | Lidar com incerteza  | Atribui um grau de pertencimento a cada classe em vez de uma decisão "dura". | Útil para problemas com classes sobrepostas.                      |
