# 1. Validação temporal e atraso de notificação

## Intuição

Para saber se um modelo de previsão funciona, é preciso testá-lo em dados que ele **não viu**. Em
séries temporais, "não viu" significa mais do que separar linhas ao acaso: significa que, no momento
de cada previsão, o modelo só pode usar informação que **já existia naquele momento**.

## Formulação

### Divisão por data

Com dados de 2015 a 2026, a divisão é:

$$
\text{treino} = \{(i,t) : t + h < \text{01/01/2024}\}, \qquad
\text{teste} = \{(i,t) : t \geq \text{01/01/2024}\}
$$

A condição do treino é sobre a **data do alvo** ($t+h$), e não só sobre $t$. Sem isso, uma previsão
feita em dezembro de 2023 para janeiro de 2024 entraria no treino, e o alvo, que é um dado do período
de teste, vazaria para o modelo.

### Por que não validação cruzada aleatória?

Se as semanas fossem sorteadas entre treino e teste, o modelo seria treinado com, por exemplo, a semana
12 de 2024 e testado na semana 11 de 2024. Como semanas vizinhas são muito parecidas, o erro de teste
ficaria artificialmente baixo. Esse problema é chamado de **vazamento temporal**.

### Validação cruzada temporal (para escolher hiperparâmetros)

Dentro do treino, a escolha de hiperparâmetros usa **origens móveis**: treina até 2020 e valida em
2021, treina até 2021 e valida em 2022, treina até 2022 e valida em 2023. O período de teste
(2024 em diante) nunca é usado em nenhuma escolha.

### Atributos sem vazamento

Todo atributo de $x_{i,t}$ é calculado só com dados até a semana $t$: defasagens ($y_{i,t}, y_{i,t-1},
\dots$), médias móveis que terminam em $t$, a tendência do estado nas últimas 4 semanas. Há testes
automáticos (`tests/test_features.py`) que conferem isso.

## O problema do atraso de notificação

Um caso de dengue de uma semana continua chegando ao sistema nas semanas seguintes. Seja $y_{i,t}^{(s)}$
o número de casos da semana $t$ conhecido na semana $s \geq t$. Na semana mais recente desta base,

$$
\frac{y_{i,t}^{(t)}}{y_{i,t}^{(\infty)}} \approx 0{,}28 ,
$$

ou seja, só cerca de 28% dos casos já estavam notificados. O InfoDengue publica uma estimativa
corrigida, o **nowcast** $\tilde{y}_{i,t}$, que estima $y_{i,t}^{(\infty)}$ a partir do padrão
histórico de atrasos.

O projeto trata isso de três formas:

1. **Na avaliação**, as 4 semanas mais recentes ficam fora, porque ainda não são a "verdade".
2. **Na previsão**, as 4 semanas mais recentes usam $\max(y_{i,t}^{(t)}, \tilde{y}_{i,t})$.
3. **Na análise de sensibilidade** (notebook 03, seção 3.4), simula-se o erro do nowcast.

## Hipóteses e riscos neste projeto

- **O teste é otimista.** Os atributos históricos usam os valores consolidados $y^{(\infty)}$, que não
  eram conhecidos em tempo real. O InfoDengue não guarda as versões antigas (*vintages*) de cada
  semana, então não é possível refazer o teste exatamente como seria em tempo real. A simulação mostra
  que o erro de curto prazo cresce com o erro do nowcast, mas a vantagem sobre o baseline se mantém.
- **O Rt e o nível de alerta** históricos também são versões finais, com o mesmo problema.
- **Um único período de teste** (2024–2026), com uma única grande epidemia. O desempenho pode ser
  diferente em outras epidemias.
- **Observações dependentes:** municípios vizinhos e semanas seguidas são correlacionados, então as
  cerca de 90 mil previsões do teste não são 90 mil evidências independentes. O teste de
  Diebold-Mariano (arquivo 7) leva isso em conta.

## No código

- `src/radar_dengue/model.py`: funções `consolidated` e `split`.
- `src/radar_dengue/predict.py`: função `correct_recent_weeks`.
- `src/radar_dengue/features.py`: construção dos atributos.

## Referências

- Hyndman, R. J. e Athanasopoulos, G. *Forecasting: Principles and Practice*, 3ª ed., OTexts, 2021,
  cap. 5 (avaliação com origens móveis).
- Bergmeir, C. e Benítez, J. M. "On the use of cross-validation for time series predictor evaluation",
  *Information Sciences*, 2012.
- Codeço, C. et al. "Infodengue: a nowcasting system for the surveillance of arboviruses in Brazil",
  *Revue d'Épidémiologie et de Santé Publique*, 2018.
