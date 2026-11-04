# Radar Dengue SP

Previsão semanal de casos de dengue para os 645 municípios de São Paulo, de 1 a 4 semanas à frente, em produção e atualizada automaticamente.

**[Dashboard](https://radar-dengue-gsqumy4teckmjnzgugdd97.streamlit.app/)** · **[Análises](notebooks)** · **[Consultas SQL](sql)** · **[API](src/radar_dengue/api.py)**

![Mapa de risco](reports/figures/dashboard_mapa.png)

## Em resumo

- **Problema:** secretarias de saúde precisam saber com semanas de antecedência onde a dengue vai aumentar, para reforçar o combate ao mosquito e preparar o atendimento.
- **Solução:** um modelo LightGBM com função de perda de Poisson, treinado com 12 anos de dados do InfoDengue (Fiocruz/FGV), que prevê os casos de cada município com intervalos de incerteza calibrados.
- **Resultado:** erra de **13% a 30% menos** que repetir a semana atual e, combinado com o alerta oficial do InfoDengue, identifica com 4 semanas de antecedência **89% dos municípios que entram em nível epidêmico**, contra 74% do alerta sozinho.

## Resultados

Avaliação em 2024–2026, período que o modelo nunca viu no treino e que inclui a maior epidemia de dengue da história do estado.

| | |
|---|---|
| **Precisão** | Erro de 11,6 casos por município-semana com 1 semana de antecedência e 25,8 com 4 semanas, contra 13,4 e 36,6 do baseline. A diferença é estatisticamente significativa com 2 a 4 semanas (teste de Diebold-Mariano, p ≤ 0,015 na escala log); com 1 semana, o ganho é pequeno e não significativo (p ≈ 0,07). |
| **Comparação** | Venceu cinco alternativas, incluindo a binomial negativa, referência estatística para contagens, que fica perto só com 3 e 4 semanas. |
| **Alerta oficial** | O alerta do InfoDengue é melhor para dizer quem continua em epidemia; o modelo antecipa cerca de metade das novas epidemias. Os dois juntos detectam mais, ao custo de mais alarmes falsos. |
| **Incerteza** | Intervalos de 50%, 80% e 95% com calibração conformal: o de 95% cobre 94,6% dos valores reais. |
| **Clima** | Controlando a época do ano, cada +1 °C se associa a só 4% a 7% mais casos, no curto prazo (associação observacional). A correlação forte vista nos dados brutos era sazonal. |
| **Epidemia de 2024** | Foi recorde porque atingiu o leste do estado, que tinha pouca dengue: incidência quase 9 vezes maior que em 2023. |
| **Ponto fraco** | Subestima o início das epidemias; reconhece bem o fim. |

## O que o projeto demonstra

- **Modelagem de séries temporais:** validação temporal sem vazamento, comparação com baselines e modelos estatísticos, ajuste de hiperparâmetros, análise de erros.
- **Estatística aplicada:** regressão binomial negativa com efeitos fixos, teste de Diebold-Mariano, regressão quantílica, calibração conformal, valores SHAP.
- **Engenharia de dados:** coleta via API, validação, banco analítico em DuckDB com consultas SQL (funções de janela, CTEs).
- **Produção:** atualização semanal pelo GitHub Actions, registro de versões do modelo, monitoramento do desempenho com alerta, API em FastAPI, dashboard em Streamlit, Docker, 37 testes automatizados.

## Como funciona

```
API do InfoDengue ──► coleta e validação (645 municípios, 2015–hoje)
                              │
               atributos: casos recentes, tendência, Rt, clima, sazonalidade
                              │
        LightGBM Poisson por horizonte + regressão quantílica calibrada
                              │
     ┌────────────────────────┼────────────────────────┐
  dashboard                  API                 banco DuckDB (SQL)
                              │
   GitHub Actions: retreina, prevê e monitora toda segunda-feira
```

## Análises

| Notebook | Pergunta |
|---|---|
| [01 · Descritiva](notebooks/01_analise_descritiva.ipynb) | Como a dengue se comporta no tempo, no espaço e com o clima? |
| [02 · Inferencial](notebooks/02_analise_inferencial.ipynb) | O clima importa além da sazonalidade? O modelo é significativamente melhor? |
| [03 · Modelagem](notebooks/03_modelagem_e_avaliacao.ipynb) | Qual método prevê melhor, onde ele erra e quão confiáveis são os intervalos? Como se compara ao alerta oficial? |
| [04 · SQL](notebooks/04_consultas_sql.ipynb) | As mesmas perguntas respondidas em SQL, num banco analítico. |

## Decisões técnicas

- **Validação temporal:** treino até 2023, teste a partir de 2024. Embaralhar os dados deixaria o modelo ver o futuro.
- **Prever a variação, não o nível:** árvores não extrapolam; prevendo o nível, o modelo subestimou a epidemia de 2024. Com a semana atual como *offset*, passou a acompanhar epidemias de qualquer tamanho.
- **Perda de Poisson:** própria para contagens; reduziu o erro em 2% a 6% em relação a prever o logaritmo.
- **Atraso de notificação:** na semana mais recente, só cerca de 28% dos casos já estão notificados. Essas semanas ficam fora da avaliação e, na previsão, usam o nowcast do InfoDengue.
- **Um modelo para todos os municípios:** cidades pequenas, com poucos casos, aprendem com o padrão das grandes.

## Limitações

- O teste usa os **casos já consolidados**; em tempo real, a semana atual depende de uma estimativa (nowcast) com erro. Numa simulação, isso aumenta o erro de curto prazo, mas o modelo continua melhor que o baseline.
- Avaliado em **um estado e uma grande epidemia**; o desempenho em outros contextos não foi testado.
- **Casos notificados**, não confirmados, como na vigilância.
- Intervalos ainda **estreitos nas cidades grandes**, cujas epidemias recentes foram mais extremas que o histórico.
- Sem informação de **sorotipo e vacinação**. Os níveis de risco são ilustrativos.

## Rodando localmente

Requer Python 3.11+.

```bash
git clone https://github.com/luishenrique1702/Radar-Dengue.git
cd Radar-Dengue
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install    # dependências
make pipeline   # coleta, treina, prevê, monitora e cria o banco SQL (~15 min)
make app        # dashboard em http://localhost:8501
make api        # API em http://localhost:8000/docs
make testes
```

## Estrutura

```
├── app/                 dashboard (Streamlit)
├── notebooks/           análises descritiva, inferencial, de modelagem e SQL
├── sql/                 consultas analíticas
├── src/radar_dengue/    coleta, validação, atributos, modelos, intervalos, previsão,
│                        monitoramento, comparação, banco SQL e API
├── tests/               37 testes
└── .github/workflows/   testes a cada mudança e atualização semanal
```

## Dados

[InfoDengue](https://info.dengue.mat.br) (Fiocruz/FGV): casos semanais, nowcast, Rt, nível de alerta e clima por município, 2015–2026, cerca de 398 mil registros. Malha municipal do IBGE via [geodata-br](https://github.com/tbrugz/geodata-br). Trabalhos relacionados: o [desafio de previsão Infodengue-Mosqlimate](https://sciety.org/articles/activity/10.1101/2025.05.12.25327419) e o [modelo por bairros do Rio de Janeiro](https://agencia.fapesp.br/modelo-de-aprendizado-de-maquina-pode-ser-utilizado-para-prever-surtos-de-dengue/39217/) (USP/UFABC/FGV).

## Autor

**Luis Henrique Toledo** · Doutorando em Probabilidade e Estatística · IME-USP · [LinkedIn](https://www.linkedin.com/in/luis-henrique-toledo-057b49161/) · [Lattes](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?id=K1107936U3)
