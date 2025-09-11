As **distâncias (ou dissimilaridades)** são métricas usadas para quantificar o quão "diferentes" dois pontos de dados são. Em termos mais simples, elas medem a proximidade ou o afastamento entre duas observações em um conjunto de dados.

Essas métricas são fundamentais em muitos algoritmos, especialmente aqueles que se baseiam na similaridade dos dados para fazer previsões ou agrupamentos.

# Tipos de Distâncias (Dissimilaridade)

## 1. Distância Euclidiana

- **O que é:** É a distância "em linha reta" entre dois pontos no espaço. É a mais popular e intuitiva.    
- **Quando usar:** É ideal para dados numéricos contínuos e é a métrica padrão em muitos algoritmos, como o K-Means (agrupamento) e o K-Nearest Neighbors (KNN - classificação).
- **Fórmula:** $d(x,y)= \sqrt{\Sigma _{i=1} ^n ​(x_i ​− y_i​)^2​}$

## 2. Distância de Manhattan (ou "Taxicab")

- **O que é:** É a soma das diferenças absolutas entre as coordenadas de dois pontos. Imagine-se em uma cidade com ruas em grade: é a distância que um táxi percorreria para ir de um ponto a outro.
- **Quando usar:** É útil quando a distância por eixo é mais importante que a diagonal. É menos sensível a outliers do que a Distância Euclidiana e pode ser mais adequada para dados com alta dimensionalidade.
- **Fórmula:** $d(x,y)= \Sigma _{i=1} ^n | x_i - y_i |$

## 3. Distância de Minkowski

- **O que é:** É uma generalização das distâncias Euclidiana e de Manhattan. Ela possui um parâmetro `p` que, ao ser alterado, muda a métrica.
    - Se p=1, é a Distância de Manhattan.
    - Se p=2, é a Distância Euclidiana.
- **Quando usar:** Quando você precisa de uma métrica flexível que pode ser ajustada para o seu problema.
- **Fórmula:** $d(x,y)=( \Sigma _{i=1} ^n |x_i ​- y_i| ^p)^{\frac{1}{p}}​$

## 4. Distância de Chebyshev

- **O que é:** É a maior diferença entre as coordenadas dos dois pontos ao longo de qualquer eixo.
- **Quando usar:** É usada em jogos (como o movimento de um rei no xadrez) ou em situações onde a "distância" é determinada pelo movimento máximo em um único eixo.
- **Fórmula:** $d(x,y) = \max _i ​(|x_i ​- y_i|)$

## 5. Distância de Mahalanobis

- **O que é:** Uma métrica que mede a distância entre um ponto e uma distribuição, levando em consideração a correlação entre as variáveis.
- **Quando usar:** É extremamente útil para dados onde as variáveis são correlacionadas, pois ela ignora essa correlação. Também é robusta a diferentes escalas nas variáveis.
- **Fórmula:** $d(x,y)=\sqrt{(x-y)^T S^{-1}(x-y)​}$ , onde $S$ é a matriz de covariância.

## 6. Distância de Hamming

- **O que é:** Mede o número de posições nas quais duas sequências de igual comprimento são diferentes.
- **Quando usar:** É ideal para dados categóricos ou binários (por exemplo, sequências de DNA ou strings de texto).
- **Exemplo:** A distância entre `1011101` e `1001001` é 2, pois eles diferem na terceira e na quinta posições.

## 7. Similaridade de Cosseno (não é uma distância, mas uma medida de similaridade)

- **O que é:** Mede o ângulo entre dois vetores. Quanto menor o ângulo (e mais próximo de 1 o valor), mais similares são os vetores.
- **Quando usar:** É amplamente utilizada em processamento de linguagem natural (NLP) para comparar documentos, onde cada documento é um vetor de palavras. É ótima para dados de alta dimensionalidade onde a magnitude do vetor não é tão importante quanto a direção.
- **Fórmula:** $s(x,y)=\dfrac{xy}{|x||y|}$

A escolha da distância certa depende do tipo dos seus dados, do algoritmo que você está usando e do problema que você está tentando resolver.

## 8. Geodésica

A **distância geodésica** é a medida do caminho mais curto entre dois pontos sobre uma superfície curva. No contexto geográfico, ela representa a menor distância possível entre duas localidades na superfície da Terra, levando em conta sua curvatura (seja modelada como uma esfera ou, mais precisamente, como um elipsoide).

Imagine esticar um barbante entre dois pontos em um globo terrestre; o comprimento desse barbante representa a distância geodésica. É o análogo de uma linha reta em um plano, mas aplicado a uma superfície curva. Por isso, essa rota é frequentemente chamada de "rota do grande círculo".

- **Diferença da Distância Euclidiana:** 
A principal diferença em relação à distância Euclidiana (ou planar) é que a geodésica considera a curvatura do espaço. A distância Euclidiana, que é a linha reta "através" da Terra, só seria prática se pudéssemos cavar um túnel reto entre os dois pontos. Para viagens na superfície, como voos de avião ou rotas marítimas, a distância geodésica é a medida mais precisa e relevante.

- **Por que é uma medida de Dissimilaridade?**
O termo **dissimilaridade** é usado em análise de dados para quantificar o quão "diferentes" dois objetos são. A distância geodésica funciona como uma medida de dissimilaridade no seguinte sentido: *Quanto maior a distância geodésica entre dois pontos, mais dissimilares eles são em termos de sua localização geográfica.*

Em outras palavras, uma grande distância geodésica implica uma baixa similaridade espacial. Essa métrica é fundamental em:
	- **Navegação Aérea e Marítima:** Para calcular as rotas mais curtas e eficientes.
	- **Sistemas de Informação Geográfica (SIG):** Para análises espaciais precisas, como calcular a área de influência de um serviço.
	- **Sismologia:** Para determinar a distância entre o epicentro de um terremoto e as estações de monitoramento.

Em resumo, a distância geodésica é a forma mais precisa de medir a separação entre dois pontos na superfície terrestre, servindo como uma medida fundamental de dissimilaridade espacial.