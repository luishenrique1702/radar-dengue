# 7. Teste de Diebold-Mariano

## Intuição

O modelo erra, em média, menos que o baseline no período de teste. Mas a diferença poderia ser acaso,
fruto de algumas semanas favoráveis? O **teste de Diebold-Mariano** responde a essa pergunta para
previsões, levando em conta que os erros de semanas vizinhas são correlacionados.

## Formulação

Para cada semana $`t`$ do teste, define-se o **diferencial de perda** entre o modelo (A) e o baseline (B):

```math
d_t = \frac{1}{n_t} \sum_{i} \Big( \big|y_{i,t+h} - \hat y^{A}_{i,t+h}\big|
- \big|y_{i,t+h} - \hat y^{B}_{i,t+h}\big| \Big)
```

a média, entre os municípios, da diferença de erro absoluto. A hipótese nula é que os dois métodos
são igualmente precisos: $`H_0: E[d_t] = 0`$. A estatística é

```math
DM = \frac{\bar d}{\sqrt{\widehat{\text{Var}}(\bar d)}}
```

que, sob $`H_0`$, tem distribuição aproximadamente normal padrão. O ponto central é a variância:
previsões de $`h`$ semanas à frente feitas em semanas seguidas compartilham informação, então $`d_t`$ é
autocorrelacionado. O projeto usa o estimador de **Newey-West (HAC)** com até 8 defasagens:

```math
\widehat{\text{Var}}(\bar d) = \frac{1}{T} \left( \hat\gamma_0
+ 2 \sum_{k=1}^{K} \left(1 - \frac{k}{K+1}\right) \hat\gamma_k \right)
```

em que $`\hat\gamma_k`$ é a autocovariância de $`d_t`$ na defasagem $`k`$.

O teste é feito em duas escalas:

- **Erro absoluto em casos:** dominado pelas cidades grandes e pelos picos.
- **Erro absoluto em $`\log(1+y)`$:** dá peso parecido a todos os municípios.

## Por que foi usado aqui

Para não afirmar que o modelo é melhor sem evidência. O resultado foi:

- **Com 2 a 4 semanas de antecedência:** melhora significativa na escala log (p entre 0,007 e 0,015).
  Em casos absolutos, significativa com 3 semanas e no limite com 2 e 4.
- **Com 1 semana:** ganho pequeno e não significativo (p ≈ 0,07).

## Hipóteses e riscos neste projeto

- **Média por semana:** agregar os municípios em cada semana trata a correlação **entre** municípios
  na mesma semana, mas reduz a amostra a cerca de 140 semanas.
- **Poucos eventos extremos:** o erro absoluto é dominado por algumas semanas de pico de 2024, o que
  aumenta a variância e reduz o poder do teste nessa escala.
- **Vários testes:** 4 horizontes e 2 escalas somam 8 testes. Os resultados são consistentes entre
  eles, mas um p-valor isolado perto de 0,05 deve ser lido com cautela.
- **Amostras pequenas:** com $`T`$ moderado, a correção de Harvey, Leybourne e Newbold melhora a
  aproximação normal; com cerca de 140 semanas, o efeito é pequeno.

## No código

- `notebooks/02_analise_inferencial.ipynb`, pergunta 2.

## Referências

- Diebold, F. X. e Mariano, R. S. "Comparing predictive accuracy", *Journal of Business & Economic
  Statistics*, 1995.
- Newey, W. K. e West, K. D. "A simple, positive semi-definite, heteroskedasticity and
  autocorrelation consistent covariance matrix", *Econometrica*, 1987.
- Harvey, D., Leybourne, S. e Newbold, P. "Testing the equality of prediction mean squared errors",
  *International Journal of Forecasting*, 1997.
- Diebold, F. X. "Comparing predictive accuracy, twenty years later", *Journal of Business & Economic
  Statistics*, 2015.
