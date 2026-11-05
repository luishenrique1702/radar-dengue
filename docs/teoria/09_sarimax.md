# 9. SARIMAX e modelos locais contra modelos globais

## Intuição

O SARIMAX é o modelo clássico de séries temporais: prevê o próximo valor a partir dos valores
anteriores (a parte **autorregressiva**), dos erros de previsão anteriores (a parte de **médias
móveis**), de um padrão que se repete a cada ano (a parte **sazonal**) e de variáveis externas (o
**X**, como o clima). É o ponto de partida natural para prever uma série com dependência temporal.

O LightGBM do projeto usa os mesmos ingredientes (casos defasados, sazonalidade, clima), mas de outra
forma. A pergunta deste arquivo é: **por que um modelo global de machine learning, e não um SARIMAX por
município?** E a resposta é empírica: o SARIMAX foi incluído na comparação.

## Formulação

### ARIMA

Para a série $`z_t = \log(1 + y_t)`$ de um município, um ARIMA$`(p, d, q)`$ é

```math
\phi(B)\,(1 - B)^d \, z_t = c + \theta(B)\, \varepsilon_t, \qquad \varepsilon_t \sim N(0, \sigma^2)
```

em que $`B`$ é o operador de defasagem ($`B z_t = z_{t-1}`$), $`\phi(B) = 1 - \phi_1 B - \dots - \phi_p B^p`$
é a parte autorregressiva e $`\theta(B) = 1 + \theta_1 B + \dots + \theta_q B^q`$ a de médias móveis.

O baseline do projeto, "repetir a semana atual", é o caso mais simples: o **passeio aleatório**,
ARIMA(0, 1, 0), $`z_t = z_{t-1} + \varepsilon_t`$.

### Sazonalidade: SARIMA ou termos de Fourier

O SARIMA clássico acrescenta polinômios sazonais em $`B^{s}`$, com $`s = 52`$ para dados semanais. Com um
período tão longo, a estimação fica lenta e instável. A alternativa usual é a **regressão harmônica
dinâmica**: a sazonalidade entra como regressores de Fourier e o ARIMA modela o restante:

```math
z_t = \sum_{k=1}^{K} \Big[ a_k \sin\!\Big(\frac{2\pi k t}{52{,}18}\Big) + b_k \cos\!\Big(\frac{2\pi k t}{52{,}18}\Big) \Big]
+ \beta^\top x_t + \eta_t, \qquad \eta_t \sim \text{ARIMA}(p, d, q)
```

com $`K = 3`$ harmônicos. Na versão com clima, $`x_t`$ contém a temperatura e a umidade médias das 4
semanas que terminam 4 semanas antes, valores que **já são conhecidos** no momento de qualquer previsão
de até 4 semanas (sem isso, seria preciso prever o clima).

### Ajuste e previsão

- Para cada município com mais de 100 mil habitantes, a ordem $`(p, d, q)`$ é escolhida pelo menor
  **AIC** entre algumas ordens comuns, com dados até 2023.
- No teste, os parâmetros ficam fixos, e o **filtro de Kalman** incorpora cada semana nova antes de
  prever de 1 a 4 semanas à frente, a mesma regra dos outros métodos.
- A previsão em casos é $`\exp(\hat z_{t+h}) - 1`$. Como $`\hat z`$ é a média do log, isso é a **mediana**
  na escala de casos, que é a previsão ótima para o erro absoluto (MAE).

## Modelos locais e globais

| | SARIMAX por município (local) | LightGBM com todos os municípios (global) |
|---|---|---|
| Parâmetros | Um conjunto por cidade | Um modelo para todas |
| Relação com o passado | Linear (no log) | Não linear, com interações |
| Informação de outras cidades | Nenhuma | A tendência do estado e o padrão das outras cidades |
| Contagens pequenas e zeros | Supõe dados contínuos e normais | Perda de Poisson, própria para contagens |
| Incerteza | Intervalos com base na normalidade | Regressão quantílica calibrada |

Em séries **individuais** bem comportadas, modelos locais como ARIMA e ETS são muito difíceis de
bater. Em **muitas séries relacionadas**, modelos globais tendem a ganhar, porque cada série tem pouca
informação sozinha e o modelo aprende padrões comuns. Na competição M5, com dezenas de milhares de
séries de vendas, os primeiros colocados usaram LightGBM global.

## Por que foi usado aqui

- **Para responder com dados à pergunta "por que não SARIMAX?"**, o modelo que um estatístico
  consideraria primeiro para uma série temporal.
- A comparação usa **exatamente as mesmas cidades, semanas e horizontes** para todos os métodos. O
  resultado está no notebook 03, seção 1.1.

## Hipóteses e riscos neste projeto

- **Só cidades grandes.** Nas pequenas, com muitos zeros, o SARIMAX não é apropriado; restringir a
  comparação às grandes favorece o SARIMAX, não o contrário.
- **Ordens limitadas.** A busca considera algumas ordens comuns, não todas; uma busca completa
  (`auto_arima`) poderia melhorar um pouco o ajuste.
- **Parâmetros fixos no teste**, como nos outros métodos. Reestimar a cada semana poderia ajudar
  durante a epidemia de 2024, mas a comparação deixaria de ser igual.
- **Normalidade no log:** a epidemia gera saltos que uma distribuição normal subestima.

## No código

- `src/radar_dengue/sarimax.py`: ajuste por cidade e previsões com origem móvel.
- `src/radar_dengue/compare.py`: função `sarimax_comparison`.

## Referências

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C. e Ljung, G. M. *Time Series Analysis: Forecasting and
  Control*, 5ª ed., Wiley, 2015.
- Hyndman, R. J. e Athanasopoulos, G. *Forecasting: Principles and Practice*, 3ª ed., OTexts, 2021,
  cap. 9 e 10 (ARIMA e regressão harmônica dinâmica).
- Makridakis, S., Spiliotis, E. e Assimakopoulos, V. "M5 accuracy competition: results, findings, and
  conclusions", *International Journal of Forecasting*, 2022.
- Montero-Manso, P. e Hyndman, R. J. "Principles and algorithms for forecasting groups of time series:
  locality and globality", *International Journal of Forecasting*, 2021.
