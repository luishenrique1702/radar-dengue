# 4. Regressão binomial negativa

## Intuição

A binomial negativa é a versão da regressão de Poisson que aceita **variância maior que a média**.
Ela aparece duas vezes no projeto:

1. **Na análise inferencial** (notebook 02), para medir a associação entre temperatura e dengue com
   intervalos de confiança corretos.
2. **Na comparação de modelos** (notebook 03), como referência estatística clássica de previsão.

## Formulação

### O modelo

$$
y_{i,t} \mid \mu_{i,t} \sim \text{BinNeg}(\mu_{i,t}, \alpha), \qquad
E[y_{i,t}] = \mu_{i,t}, \qquad \text{Var}(y_{i,t}) = \mu_{i,t} + \alpha \mu_{i,t}^2
$$

Quando $\alpha \to 0$, volta a ser Poisson. Uma forma de entender: a binomial negativa é uma Poisson
cuja média tem uma variação extra, com distribuição gama (uma mistura gama-Poisson).

### Na análise inferencial (efeito da temperatura)

$$
\log \mu_{i,t} = \log N_i + \beta \, T_{i,t-L} + \gamma \, U_{i,t-L}
+ s(\text{semana}) + \delta_{\text{ano}} + \phi_i
$$

- $\log N_i$: offset da população; os coeficientes passam a ser sobre a **incidência**.
- $T_{i,t-L}$: temperatura média das 4 semanas que terminam $L$ semanas antes.
- $s(\text{semana})$: seno e cosseno da semana do ano (sazonalidade).
- $\delta_{\text{ano}}$: efeito fixo de ano (absorve diferenças entre anos, como sorotipo e imunidade).
- $\phi_i$: efeito fixo de município (diferenças permanentes entre cidades).

O coeficiente $e^{\beta}$ é a **razão de taxas de incidência (IRR)**: $e^{\beta} = 1{,}06$ significa
6% mais casos por habitante para cada grau a mais, mantidos os outros termos.

### Erros-padrão robustos agrupados

As semanas de um mesmo município são correlacionadas, então os erros-padrão usuais seriam pequenos
demais. O projeto usa o estimador **sanduíche agrupado** por município:

$$
\widehat{\text{Var}}(\hat\beta) = A^{-1} \left( \sum_{g} s_g s_g^\top \right) A^{-1}
$$

em que $A$ é a matriz de informação e $s_g$ é a soma dos escores do grupo $g$. Como robustez, o
notebook também agrupa em duas dimensões (município e semana), para tratar choques que atingem todas
as cidades na mesma semana.

### Na comparação de previsão

$$
\log \mu_{i,t+h} = \log(1 + y_{i,t}) + \beta^\top z_{i,t}
$$

com $z_{i,t}$ formado só por **variações** (diferenças entre semanas, crescimento, Rt, sazonalidade),
e não por níveis.

## Por que foi usado aqui

- **Sobredispersão forte:** a variância dos casos é cerca de 44 vezes a média (mediana entre
  municípios). Com Poisson, os intervalos de confiança do efeito da temperatura seriam estreitos
  demais.
- **Referência para o machine learning:** sem uma referência estatística bem feita, não dá para dizer
  se o LightGBM agrega algo. A binomial negativa fica perto dele com 3 e 4 semanas de antecedência.

## Hipóteses e riscos neste projeto

- **Confundimento sazonal:** temperatura e dengue sobem juntas no verão. Sem a sazonalidade no modelo,
  cada grau parecia associado a cerca de 30% mais casos; com ela, 4% a 7%. Mesmo assim, é uma
  **associação observacional**: sorotipo, imunidade e ações de controle não foram medidos.
- **Defasagem zero:** calor e casos na mesma semana podem ter causas comuns; essa estimativa deve ser
  lida com cautela.
- **A especificação da sazonalidade** (seno e cosseno) não captura perfeitamente o formato da
  estação, o que pode explicar os IRR levemente abaixo de 1 nas defasagens longas.
- **Na previsão, atributos em nível fazem o modelo explodir.** Com $\log y$ das semanas anteriores
  como atributos, os coeficientes multiplicam valores muito grandes nas cidades grandes durante a
  epidemia, e a previsão para a capital passou de 500 mil casos por semana. Atributos em variação,
  que não dependem do tamanho da cidade, resolveram.

## No código

- `notebooks/02_analise_inferencial.ipynb`: análise do clima.
- `src/radar_dengue/compare.py`: `NB_FEATURES`, `nb_features` e o ajuste para a comparação.

## Referências

- Hilbe, J. M. *Negative Binomial Regression*, 2ª ed., Cambridge University Press, 2011.
- Cameron, A. C. e Trivedi, P. K. *Regression Analysis of Count Data*, 2ª ed., 2013.
- Cameron, A. C., Gelbach, J. B. e Miller, D. L. "Robust inference with multiway clustering",
  *Journal of Business & Economic Statistics*, 2011.
- Lowe, R. et al. "Combined effects of hydrometeorological hazards and urbanisation on dengue risk in
  Brazil: a spatiotemporal modelling study", *The Lancet Planetary Health*, 2021.
