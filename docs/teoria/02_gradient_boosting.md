# 2. Gradient boosting e LightGBM

## Intuição

Uma **árvore de decisão** divide os dados em grupos com perguntas do tipo "a incidência atual é maior
que 30?" e "estamos antes da semana 10?", e dá a cada grupo uma previsão. Uma árvore sozinha é fraca:
ou é simples demais, ou decora os dados.

O **gradient boosting** combina centenas de árvores pequenas **em sequência**. Cada nova árvore é
treinada para corrigir o erro que as anteriores ainda cometem. O resultado é um modelo que capta
relações não lineares e interações entre atributos, como "crescimento rápido **e** início do verão".

## Formulação

O modelo é uma soma de $`M`$ árvores:

```math
F_M(x) = F_0 + \sum_{m=1}^{M} \nu \, f_m(x)
```

em que $`f_m`$ é a $`m`$-ésima árvore e $`\nu`$ é a taxa de aprendizado (0,03 neste projeto), que encolhe a
contribuição de cada árvore para evitar sobreajuste.

O objetivo é minimizar uma **função de perda** $`L(y, F(x))`$ somada sobre os dados. A cada passo, a
nova árvore aproxima a direção de maior descida da perda, o **gradiente negativo**:

```math
g_{j} = -\left.\frac{\partial L(y_j, F)}{\partial F}\right|_{F = F_{m-1}(x_j)}
```

Para a perda quadrática, $`L = \tfrac12 (y - F)^2`$, o gradiente negativo é o resíduo $`y - F`$: cada
árvore literalmente "ajusta os resíduos" das anteriores. Para outras perdas, como a de Poisson
(arquivo 3), o gradiente muda, mas a lógica é a mesma. O LightGBM também usa a segunda derivada
(*hessiana*) $`h_j = \partial^2 L / \partial F^2`$, num passo do tipo Newton, para escolher os valores
das folhas:

```math
w^*_{\text{folha}} = -\frac{\sum_{j \in \text{folha}} g_j}{\sum_{j \in \text{folha}} h_j + \lambda}
```

em que $`\lambda`$ é uma penalidade de regularização (`reg_lambda = 1`).

## O que o LightGBM tem de particular

- **Histogramas:** os valores de cada atributo são agrupados em faixas antes do treino, o que torna
  a busca pelas melhores divisões muito mais rápida.
- **Crescimento pela folha (*leaf-wise*):** em vez de crescer a árvore nível por nível, expande a
  folha que mais reduz a perda. É eficiente, mas pode sobreajustar; por isso o projeto limita
  `num_leaves = 63` e exige pelo menos 50 observações por folha (`min_child_samples`).
- **Subamostragem:** cada árvore usa 80% das linhas e 80% dos atributos, o que reduz a variância.

O XGBoost implementa a mesma ideia, com diferenças de implementação. Em dados tabulares bem
tratados, os dois costumam ter desempenho parecido; o LightGBM foi escolhido pela velocidade, já que
o pipeline retreina 4 modelos centrais e 24 quantílicos toda semana.

## Por que foi usado aqui

- **Relações não lineares:** a regressão linear, com os mesmos atributos, perdeu até do baseline
  ingênuo. O efeito de um crescimento rápido depende da época do ano e do tamanho da epidemia.
- **Um modelo global:** um único modelo para os 645 municípios. Cidades pequenas, com poucos casos,
  aprendem com o padrão das grandes.
- **Escala variada:** de 0 a dezenas de milhares de casos por semana. Com a perda de Poisson e o
  offset (arquivo 3), o modelo aprende **variações relativas**, comparáveis entre cidades.

## Hipóteses e riscos neste projeto

- **Árvores não extrapolam.** Uma árvore só prevê valores que viu no treino. Prevendo o número de
  casos diretamente, o modelo subestimou a epidemia de 2024, maior que tudo no treino. A solução foi
  prever a **variação em relação à semana atual**, que tem uma faixa estável mesmo em epidemias
  recordes.
- **Sobreajuste:** controlado pela taxa de aprendizado pequena, pelo limite de folhas e pela
  subamostragem. A busca de hiperparâmetros mudou o erro em menos de 2%, sinal de que o modelo não é
  sensível a essas escolhas.

## No código

- `src/radar_dengue/model.py`: `PARAMS` (hiperparâmetros) e `fit_central`.
- `models/tuning.json`: resultado da busca de hiperparâmetros.

## Referências

- Friedman, J. H. "Greedy function approximation: a gradient boosting machine", *Annals of
  Statistics*, 2001.
- Ke, G. et al. "LightGBM: a highly efficient gradient boosting decision tree", *NeurIPS*, 2017.
- Chen, T. e Guestrin, C. "XGBoost: a scalable tree boosting system", *KDD*, 2016.
- Hastie, T., Tibshirani, R. e Friedman, J. *The Elements of Statistical Learning*, 2ª ed., Springer,
  2009, cap. 10.
