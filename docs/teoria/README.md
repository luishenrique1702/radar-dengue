# Fundamentos teóricos

Esta pasta explica os métodos usados no Radar Dengue. Cada arquivo segue a mesma estrutura:

1. **Intuição:** o que o método faz, em poucas frases.
2. **Formulação:** as equações principais.
3. **Por que foi usado aqui:** que problema dos dados ele resolve.
4. **Hipóteses e riscos neste projeto:** onde ele pode falhar.
5. **No código:** onde o método aparece.
6. **Referências.**

O foco não é repetir os livros, e sim mostrar a teoria **aplicada às escolhas deste projeto**.

| # | Método | Papel no projeto |
|---|---|---|
| 1 | [Validação temporal e atraso de notificação](01_validacao_temporal.md) | Como o modelo é avaliado sem "ver o futuro" |
| 2 | [Gradient boosting e LightGBM](02_gradient_boosting.md) | O algoritmo do modelo de previsão |
| 3 | [Perda de Poisson com offset](03_perda_poisson.md) | Como o modelo trata os casos como contagens |
| 4 | [Regressão binomial negativa](04_binomial_negativa.md) | Inferência sobre o clima e referência estatística na comparação |
| 5 | [Regressão quantílica e calibração conformal](05_intervalos.md) | Intervalos de previsão de 50%, 80% e 95% |
| 6 | [Valores SHAP](06_shap.md) | Explicação das previsões |
| 7 | [Teste de Diebold-Mariano](07_diebold_mariano.md) | Se a melhora sobre o baseline é estatisticamente significativa |
| 8 | [Métricas de avaliação](08_metricas.md) | MAE, WAPE, precisão e recall dos alertas |
| 9 | [SARIMAX: modelos locais contra globais](09_sarimax.md) | Por que não um SARIMAX por município: comparação com dados |

## Notação

| Símbolo | Significado |
|---|---|
| $`i`$ | município |
| $`t`$ | semana (a semana da previsão) |
| $`h`$ | horizonte: quantas semanas à frente (1 a 4) |
| $`y_{i,t}`$ | casos notificados no município $`i`$ na semana $`t`$ |
| $`x_{i,t}`$ | vetor de atributos disponíveis na semana $`t`$ |
| $`N_i`$ | população do município |
| $`\hat{y}_{i,t+h}`$ | previsão feita em $`t`$ para a semana $`t+h`$ |
