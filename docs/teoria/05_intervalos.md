# 5. Regressão quantílica e calibração conformal

## Intuição

Uma previsão de "45 casos" não diz quanto ela pode errar. Um **intervalo de previsão** de 80% diz:
"o número real deve cair entre 30 e 65 em cerca de 80% das vezes". O projeto constrói esses
intervalos em dois passos:

1. **Regressão quantílica:** modelos que preveem diretamente os limites inferior e superior.
2. **Calibração conformal:** um ajuste, feito com dados de um período separado, para que "80%" cubra
   de fato cerca de 80%.

**Intervalo de previsão não é intervalo de confiança.** O intervalo de confiança é sobre um
**parâmetro** (por exemplo, o efeito da temperatura). O de previsão é sobre **uma observação futura**,
e por isso é mais largo e costuma ser mostrado em níveis como 50% e 80%, além de 95%.

## Formulação

### Regressão quantílica

O quantil $\tau$ de $y$ dado $x$ é o valor $q_\tau(x)$ com $P(y \le q_\tau(x) \mid x) = \tau$. Ele
minimiza a **perda pinball**:

$$
\rho_\tau(u) = u \, (\tau - \mathbb{1}\{u < 0\}), \qquad u = y - q
$$

Para $\tau = 0{,}9$, errar para baixo custa 9 vezes mais que errar para cima, e o mínimo fica no
percentil 90. O LightGBM minimiza essa perda com `objective="quantile"`. O projeto treina os quantis
2,5%, 10%, 25%, 75%, 90% e 97,5%, o que forma os intervalos de 95%, 80% e 50%. Como no modelo
central, o alvo é a variação de $\log(1+y)$ em relação à semana atual.

### Calibração conformal (CQR)

Regressões quantílicas sozinhas costumam errar a cobertura, especialmente quando o período de uso é
diferente do treino. A **Conformalized Quantile Regression** corrige isso:

1. Ajusta os modelos quantílicos num conjunto de treino (aqui, até 2022).
2. Num conjunto de **calibração** separado (2023), calcula para cada observação o escore
   $$E_j = \max\{\hat q_{\text{inf}}(x_j) - y_j,\; y_j - \hat q_{\text{sup}}(x_j)\}$$
   que é positivo quando $y_j$ cai fora do intervalo e mede o quanto.
3. Toma o quantil empírico $Q$ de nível $(1-\alpha)(1 + 1/n)$ desses escores.
4. O intervalo calibrado é $[\hat q_{\text{inf}}(x) - Q,\; \hat q_{\text{sup}}(x) + Q]$.

Se os escores da calibração e do uso forem **permutáveis** (por exemplo, independentes e com a mesma
distribuição), a cobertura é garantida em amostras finitas:

$$
P\big(y \in [\hat q_{\text{inf}}(x) - Q,\; \hat q_{\text{sup}}(x) + Q]\big) \ge 1 - \alpha
$$

sem nenhuma hipótese sobre a distribuição dos dados.

## Por que foi usado aqui

- Sem calibração, o intervalo de 95% cobriu só cerca de 90% dos valores reais no teste (2024–2026),
  porque 2024 foi mais extremo que o histórico.
- Com calibração, cobriu 94,6% (com 4 semanas de antecedência). Os de 50% e 80% ficaram em cerca de
  53% e 78%.

## Hipóteses e riscos neste projeto

- **Permutabilidade não vale exatamente.** Séries temporais têm dependência, e 2024 foi diferente de
  2023, o ano de calibração. A garantia teórica não se aplica literalmente; a cobertura observada,
  porém, ficou perto do nominal.
- **Cobertura média, não por grupo.** A garantia vale na média. Por porte, a cobertura é melhor nas
  cidades pequenas e pior nas grandes, cujas epidemias recentes foram mais extremas. Uma melhoria
  seria calibrar separadamente por porte (conformal condicional).
- **Calibração na escala log:** o ajuste $Q$ é somado em $\log(1+y)$, o que equivale a um ajuste
  multiplicativo na escala de casos. Isso faz sentido para contagens, cujo erro cresce com o nível.

## No código

- `src/radar_dengue/intervals.py`: `fit_quantiles`, `conformal_adjustment`, `predict_bounds` e
  `evaluate`.
- `models/intervalos.json`: ajustes e coberturas por horizonte e nível.

## Referências

- Koenker, R. e Bassett, G. "Regression quantiles", *Econometrica*, 1978.
- Koenker, R. *Quantile Regression*, Cambridge University Press, 2005.
- Romano, Y., Patterson, E. e Candès, E. "Conformalized quantile regression", *NeurIPS*, 2019.
- Vovk, V., Gammerman, A. e Shafer, G. *Algorithmic Learning in a Random World*, Springer, 2005.
- Angelopoulos, A. N. e Bates, S. "A gentle introduction to conformal prediction and distribution-free
  uncertainty quantification", *arXiv:2107.07511*, 2021.
