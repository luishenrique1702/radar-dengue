# 3. Perda de Poisson com offset

## Intuição

Casos de dengue são **contagens**: números inteiros, não negativos, com variância que cresce com a
média. A "perda" é a régua que o modelo usa para medir o próprio erro durante o treino. Trocar a
régua dos erros ao quadrado pela **perda de Poisson** faz o LightGBM tratar os casos como contagens,
do mesmo jeito que um modelo de Poisson tradicional.

O **offset** é um termo fixo somado à previsão. Aqui, ele é o número de casos da semana atual: o
modelo não precisa aprender o nível da epidemia, só **quanto ela vai mudar**.

## Formulação

### O modelo de Poisson tradicional (um GLM)

$$
y_{i,t+h} \sim \text{Poisson}(\mu_{i,t+h}), \qquad
\log \mu_{i,t+h} = \underbrace{\log(1 + y_{i,t})}_{\text{offset}} + \beta^\top x_{i,t}
$$

### O LightGBM com perda de Poisson

A mesma hipótese sobre os dados, mas a parte linear $\beta^\top x$ é trocada por uma soma de árvores
$F(x)$:

$$
\log \mu_{i,t+h} = \log(1 + y_{i,t}) + F(x_{i,t})
$$

O modelo minimiza a **deviance de Poisson** (o negativo da log-verossimilhança, sem as constantes):

$$
L(y, \mu) = \mu - y \log \mu
$$

Na escala do preditor $\eta = \log \mu$, o gradiente e a hessiana são

$$
\frac{\partial L}{\partial \eta} = \mu - y, \qquad \frac{\partial^2 L}{\partial \eta^2} = \mu ,
$$

ou seja, cada árvore corrige a diferença entre o previsto e o observado, como na perda quadrática,
mas com pesos proporcionais a $\mu$. No LightGBM, o offset entra como `init_score`.

### Por que a Poisson serve mesmo com sobredispersão

Os casos de dengue têm **sobredispersão**: a variância é dezenas de vezes maior que a média, e a
Poisson supõe $\text{Var}(y) = \mu$. Ainda assim, a perda de Poisson estima a **média** de forma
consistente. As equações de estimação

$$
\sum_j (y_j - \mu_j)\, \frac{\partial \eta_j}{\partial \theta} = 0
$$

só exigem que $E[y \mid x] = \mu(x)$ esteja certa; não dependem da variância. É o resultado clássico
da **quase-verossimilhança** e da **pseudo-máxima verossimilhança de Poisson**. A sobredispersão afeta
a **incerteza** (erros-padrão e intervalos), que o projeto trata separadamente, com regressão
quantílica calibrada (arquivo 5).

## Por que foi usado aqui

- **Contagens com muitos zeros:** 60% das semanas das cidades pequenas têm zero casos. Prever o
  logaritmo e voltar para a escala original introduz viés (a média do log não é o log da média); a
  Poisson modela a média diretamente.
- **Resultado:** errou de 0,4% a 4,8% menos que o mesmo LightGBM prevendo $\log(1+y)$ com perda
  quadrática. A Tweedie, que acomoda variância maior, deu resultado praticamente igual.

## Hipóteses e riscos neste projeto

- **O offset usa $\log(1+y)$, não $\log y$**, para evitar $\log 0$ nas semanas sem casos. É uma
  aproximação; o modelo aprende a compensá-la, porque a mesma transformação é usada no treino e na
  previsão.
- **A média condicional precisa estar bem especificada**, o que depende dos atributos. Se faltar uma
  informação importante (por exemplo, o sorotipo circulante), a previsão da média fica enviesada,
  qualquer que seja a perda.

## No código

- `src/radar_dengue/model.py`: `fit_central` (treino com `objective="poisson"` e `init_score`) e
  `predict_central`.

## Referências

- McCullagh, P. e Nelder, J. A. *Generalized Linear Models*, 2ª ed., Chapman & Hall, 1989.
- Wedderburn, R. W. M. "Quasi-likelihood functions, generalized linear models, and the Gauss-Newton
  method", *Biometrika*, 1974.
- Gourieroux, C., Monfort, A. e Trognon, A. "Pseudo maximum likelihood methods: applications to
  Poisson models", *Econometrica*, 1984.
- Cameron, A. C. e Trivedi, P. K. *Regression Analysis of Count Data*, 2ª ed., Cambridge University
  Press, 2013.
