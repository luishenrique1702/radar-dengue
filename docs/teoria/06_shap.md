# 6. Valores SHAP

## Intuição

Um modelo com centenas de árvores não pode ser lido diretamente. Os **valores SHAP** dividem cada
previsão entre os atributos: quanto cada um empurrou a previsão para cima ou para baixo em relação à
previsão média. É como repartir uma conta entre os participantes de forma justa.

## Formulação

A ideia vem dos **valores de Shapley**, da teoria dos jogos cooperativos. Para um modelo $f$, um ponto
$x$ e o conjunto de atributos $\mathcal{F}$, a contribuição do atributo $j$ é a média da sua
contribuição marginal sobre todas as ordens possíveis de "entrada" dos atributos:

$$
\phi_j(x) = \sum_{S \subseteq \mathcal{F} \setminus \{j\}}
\frac{|S|!\,(|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!}
\Big[ v(S \cup \{j\}) - v(S) \Big]
$$

em que $v(S)$ é a previsão esperada do modelo quando só os atributos de $S$ são conhecidos. As
contribuições somam a previsão:

$$
f(x) = \phi_0 + \sum_{j} \phi_j(x)
$$

com $\phi_0$ igual à previsão média. Os valores de Shapley são a única forma de repartição que
satisfaz um conjunto de propriedades de justiça (eficiência, simetria, atributo nulo e
aditividade).

Em geral, calcular $\phi_j$ exige um número exponencial de combinações. Para árvores, o algoritmo
**TreeSHAP** calcula os valores exatos em tempo polinomial, aproveitando a estrutura das árvores. O
LightGBM o implementa diretamente (`predict(..., pred_contrib=True)`).

## Por que foi usado aqui

- **Importância global:** a média de $|\phi_j|$ sobre muitas previsões mostra o que o modelo mais usa:
  o nível recente de casos, a época do ano, a tendência do estado, o porte do município. O clima
  aparece por último entre os dez mais importantes, coerente com a análise inferencial.
- **Explicação individual:** por que o modelo previu cerca de 3.600 casos para Campinas em março de
  2024, quando o real foi cerca de 6.500. A tendência de alta do estado empurrou para cima; a
  incidência atual já alta puxou para baixo. Esse tipo de explicação permite a um gestor entender, e
  questionar, um alerta.

## Hipóteses e riscos neste projeto

- **SHAP explica o modelo, não a doença.** Se "tendência do estado" tem SHAP alto, o modelo usa muito
  essa informação; isso não significa que ela **causa** os casos.
- **Atributos correlacionados:** casos da semana atual, da anterior e as médias móveis carregam
  informação parecida. O SHAP divide o crédito entre eles de um jeito que depende do modelo, então a
  importância individual de cada um deve ser lida com cuidado; agrupados, eles são claramente o mais
  importante.
- **Escala:** no modelo com perda de Poisson, as contribuições estão na escala do log da média de
  casos, e não incluem o offset (a semana atual), que é somado à parte.

## No código

- `notebooks/03_modelagem_e_avaliacao.ipynb`, seção 4.

## Referências

- Shapley, L. S. "A value for n-person games", em *Contributions to the Theory of Games II*,
  Princeton University Press, 1953.
- Lundberg, S. M. e Lee, S.-I. "A unified approach to interpreting model predictions", *NeurIPS*,
  2017.
- Lundberg, S. M. et al. "From local explanations to global understanding with explainable AI for
  trees", *Nature Machine Intelligence*, 2020.
- Molnar, C. *Interpretable Machine Learning*, 2ª ed., 2022, cap. 9.
