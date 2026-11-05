# 8. Métricas de avaliação

## Intuição

Cada métrica responde a uma pergunta diferente. O projeto usa várias porque nenhuma, sozinha, conta a
história inteira.

## Formulação

### Erro médio absoluto (MAE)

$$
\text{MAE} = \frac{1}{n} \sum_{i,t} \big| y_{i,t+h} - \hat y_{i,t+h} \big|
$$

Em casos por município-semana. Fácil de interpretar, mas dominado pelas cidades grandes: errar 500
casos na capital pesa mais que errar 5 em cada uma de 50 cidades pequenas.

### Erro absoluto ponderado (WAPE)

$$
\text{WAPE} = \frac{\sum_{i,t} |y_{i,t+h} - \hat y_{i,t+h}|}{\sum_{i,t} y_{i,t+h}}
$$

O erro total como fração dos casos totais. Diferente do erro percentual médio (MAPE), não se torna
infinito quando $y = 0$, o que é essencial aqui, com 60% de semanas sem casos nas cidades pequenas.

### Erro em escala log (RMSE log)

$$
\text{RMSE}_{\log} = \sqrt{ \frac{1}{n} \sum_{i,t} \big( \log(1 + y_{i,t+h}) - \log(1 + \hat y_{i,t+h}) \big)^2 }
$$

Mede o erro **relativo** e dá peso parecido a cidades de portes diferentes.

### Precisão e recall dos alertas

O alerta é tratado como uma classificação: o município estará em risco alto (ou em nível epidêmico
do InfoDengue) daqui a $h$ semanas?

$$
\text{precisão} = \frac{VP}{VP + FP}, \qquad \text{recall} = \frac{VP}{VP + FN}
$$

em que $VP$ são os alertas que se confirmaram, $FP$ os alarmes falsos e $FN$ os casos não
detectados. A precisão responde "quando o modelo alerta, quanto acerta?"; o recall, "das situações
de risco, quantas ele detectou?".

### Cobertura dos intervalos

$$
\text{cobertura} = \frac{1}{n} \sum_{i,t} \mathbb{1}\{ y_{i,t+h} \in [\hat q_{\text{inf}}, \hat q_{\text{sup}}] \}
$$

Deve ficar perto do nível nominal (50%, 80% ou 95%).

### WIS (*weighted interval score*)

A cobertura sozinha premia intervalos enormes. O **interval score** de um intervalo central de nível
$1-\alpha$, com limites $l$ e $u$, pune a largura e cada valor que fica fora:

$$
IS_\alpha(l, u; y) = (u - l) + \frac{2}{\alpha}(l - y)\,\mathbb{1}\{y < l\} + \frac{2}{\alpha}(y - u)\,\mathbb{1}\{y > u\}
$$

O WIS combina a mediana $m$ e os $K$ intervalos (aqui, 50%, 80% e 95%):

$$
WIS = \frac{1}{K + 1/2}\left( \tfrac{1}{2}|y - m| + \sum_{k=1}^{K} \frac{\alpha_k}{2}\, IS_{\alpha_k} \right)
$$

É uma regra de pontuação própria, aproxima o CRPS e está na mesma unidade dos casos: com só a mediana,
vira metade do erro absoluto. Valores menores são melhores. É a métrica dos centros de previsão do CDC
e do desafio Infodengue-Mosqlimate.

## Por que essas métricas aqui

- **MAE** é a métrica principal porque está na unidade que um gestor entende (casos).
- **WAPE** dá a ordem de grandeza relativa (o modelo erra cerca de 17% a 37% dos casos totais).
- **RMSE log** e o teste em escala log garantem que o resultado não depende só das cidades grandes.
- **WIS** avalia a previsão inteira (centro e intervalos) e é usado no backtest de 2019 a 2026,
  contra um baseline que repete a semana atual com a incerteza da variação histórica.
- **Precisão e recall** traduzem a previsão numa decisão (alertar ou não), com o custo dos alarmes
  falsos explícito.

## Hipóteses e riscos neste projeto

- **O baseline importa.** "Repetir a semana atual" é difícil de bater no curto prazo, por isso é a
  referência mais honesta. Comparar só com a média histórica faria qualquer modelo parecer ótimo.
- **O limiar de alerta é uma escolha.** Os níveis de risco (10, 50 e 100 casos por 100 mil habitantes)
  são do projeto, não um padrão oficial. A troca entre precisão e recall depende do custo de cada erro
  para a vigilância.

## No código

- `src/radar_dengue/model.py`: função `metrics`.
- `src/radar_dengue/compare.py`: comparação de métodos e com o alerta do InfoDengue.
- `src/radar_dengue/backtest.py`: `interval_score` e `wis`, no backtest com origem móvel.

## Referências

- Hyndman, R. J. e Koehler, A. B. "Another look at measures of forecast accuracy", *International
  Journal of Forecasting*, 2006.
- Gneiting, T. e Raftery, A. E. "Strictly proper scoring rules, prediction, and estimation",
  *Journal of the American Statistical Association*, 2007.
- Bracher, J., Ray, E. L., Gneiting, T. e Reich, N. G. "Evaluating epidemic forecasts in an interval
  format", *PLOS Computational Biology*, 2021.
