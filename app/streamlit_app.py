"""Dashboard do Radar Dengue.

Rodar localmente:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar_dengue.config import DATA_PROCESSED, GEOJSON, MODELS, ROOT  # noqa: E402

# Paleta (rampa sequencial azul e cores categóricas fixas)
RAMPA = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
NIVEIS = {"Baixo": "#86b6ef", "Moderado": "#3987e5", "Alto": "#1c5cab", "Muito alto": "#0d366b"}
COR_REAL, COR_PREV, COR_BASE = "#2a78d6", "#eb6834", "#898781"

st.set_page_config(page_title="Radar Dengue SP", page_icon="🦟", layout="wide")


@st.cache_data
def load():
    prev = pd.read_parquet(DATA_PROCESSED / "previsao_atual.parquet")
    hist = pd.read_parquet(DATA_PROCESSED / "historico.parquet")
    teste = pd.read_parquet(DATA_PROCESSED / "previsoes_teste.parquet")
    with open(GEOJSON, encoding="utf-8") as f:
        geo = json.load(f)
    with open(MODELS / "metricas.json", encoding="utf-8") as f:
        metricas = json.load(f)
    arq_mon = ROOT / "data" / "monitoring" / "desempenho.parquet"
    desempenho = pd.read_parquet(arq_mon) if arq_mon.exists() else pd.DataFrame()
    arq_reg = MODELS / "registro" / "versoes.csv"
    versoes = pd.read_csv(arq_reg) if arq_reg.exists() else pd.DataFrame()
    arq_cmp = MODELS / "comparacao.json"
    comparacao = json.loads(arq_cmp.read_text(encoding="utf-8")) if arq_cmp.exists() else {}
    arq_al = MODELS / "alerta_infodengue.json"
    alerta = json.loads(arq_al.read_text(encoding="utf-8")) if arq_al.exists() else {}
    return prev, hist, teste, geo, metricas, desempenho, versoes, comparacao, alerta


prev, hist, teste, geo, metricas, desempenho, versoes, comparacao, alerta = load()
semana_base = pd.Timestamp(prev["semana_base"].iloc[0])

st.title("Radar Dengue · Estado de São Paulo")
st.caption(
    f"Previsão semanal de casos de dengue para os 645 municípios paulistas. "
    f"Última semana com dados: {semana_base:%d/%m/%Y}. "
    "Fonte: InfoDengue (casos notificados e clima)."
)

aba_mapa, aba_cidade, aba_modelo, aba_monit = st.tabs(
    ["Mapa de risco", "Por município", "Desempenho do modelo", "Monitoramento"]
)

# ---------------------------------------------------------------- Mapa
with aba_mapa:
    h = st.select_slider("Previsão para daqui a", options=[1, 2, 3, 4], value=2,
                         format_func=lambda x: f"{x} semana{'s' if x > 1 else ''}")
    p = prev[prev["horizonte"] == h]
    data_alvo = pd.Timestamp(p["data_alvo"].iloc[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Casos previstos no estado", f"{p['casos_previstos'].sum():,.0f}".replace(",", "."))
    c2.metric("Municípios em risco alto ou muito alto",
              int(p["nivel_risco"].isin(["Alto", "Muito alto"]).sum()))
    c3.metric("Semana prevista", f"{data_alvo:%d/%m/%Y}")

    fig = px.choropleth(
        p, geojson=geo, locations="geocode", featureidkey="properties.id",
        color="incidencia_100k", color_continuous_scale=RAMPA,
        range_color=(0, max(50, p["incidencia_100k"].quantile(0.98))),
        hover_name="municipio",
        hover_data={"geocode": False, "casos_previstos": ":.0f",
                    "incidencia_100k": ":.1f", "nivel_risco": True},
        labels={"incidencia_100k": "Casos por 100 mil hab.",
                "casos_previstos": "Casos previstos", "nivel_risco": "Nível de risco"},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_traces(marker_line_width=0.2, marker_line_color="#fcfcfb")
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_colorbar=dict(title="Casos por<br>100 mil hab."))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Municípios com maior incidência prevista")
    top = (p[p["casos_previstos"] >= 5].sort_values("incidencia_100k", ascending=False).head(15)
           [["municipio", "casos_previstos", "incidencia_100k", "nivel_risco"]]
           .rename(columns={"municipio": "Município", "casos_previstos": "Casos previstos",
                            "incidencia_100k": "Por 100 mil hab.", "nivel_risco": "Nível"}))
    st.dataframe(top, hide_index=True, width="stretch",
                 column_config={"Casos previstos": st.column_config.NumberColumn(format="%d"),
                                "Por 100 mil hab.": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Níveis de risco por incidência semanal: Baixo < 10 · Moderado 10–50 · "
               "Alto 50–100 · Muito alto > 100 casos por 100 mil habitantes. "
               "Ranking considera apenas municípios com pelo menos 5 casos previstos, "
               "para evitar distorções em cidades muito pequenas.")

# ---------------------------------------------------------------- Município
with aba_cidade:
    nomes = sorted(prev["municipio"].unique())
    cidade = st.selectbox("Município", nomes, index=nomes.index("São Paulo"))
    hc = hist[hist["municipio"] == cidade].sort_values("data_iniSE")
    hc = hc[hc["data_iniSE"] >= semana_base - pd.Timedelta(weeks=52)].copy()
    # Semanas recentes: usa a estimativa corrigida pelo atraso de notificação
    recente = hc["data_iniSE"] > semana_base - pd.Timedelta(weeks=4)
    hc["casos"] = hc["casos"].astype(float)
    hc.loc[recente, "casos"] = hc.loc[recente, ["casos", "casos_est"]].max(axis=1).round()
    pc = prev[prev["municipio"] == cidade].sort_values("horizonte")

    fig = go.Figure()
    if "casos_min" in pc.columns:
        fig.add_scatter(x=pd.concat([pc["data_alvo"], pc["data_alvo"][::-1]]),
                        y=pd.concat([pc["casos_max"], pc["casos_min"][::-1]]),
                        fill="toself", fillcolor="rgba(235,104,52,0.2)", line=dict(width=0),
                        mode="lines",
                        name="Intervalo de 80%", hoverinfo="skip")
    fig.add_scatter(x=hc["data_iniSE"], y=hc["casos"], name="Casos notificados",
                    line=dict(color=COR_REAL, width=2))
    ponte = pd.concat([hc.tail(1).rename(columns={"data_iniSE": "data_alvo",
                                                  "casos_est": "casos_previstos"}),
                       pc])
    fig.add_scatter(x=ponte["data_alvo"], y=ponte["casos_previstos"], name="Previsão",
                    line=dict(color=COR_PREV, width=2, dash="dot"), mode="lines+markers",
                    marker=dict(size=8))
    fig.update_layout(height=420, hovermode="x unified", yaxis_title="Casos por semana",
                      legend=dict(orientation="h", y=1.08), margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("As últimas semanas ainda recebem notificações atrasadas; por isso a previsão "
               "parte da estimativa corrigida (nowcast) do InfoDengue. A faixa sombreada é o "
               "intervalo de 80%: o valor real deve cair dentro dela em cerca de 8 de cada 10 "
               "semanas (no teste, caiu em 76%).")

    cols = ["data_alvo", "casos_previstos", "casos_min", "casos_max", "incidencia_100k",
            "nivel_risco"]
    st.dataframe(pc[[c for c in cols if c in pc.columns]]
                 .rename(columns={"data_alvo": "Semana", "casos_previstos": "Casos previstos",
                                  "casos_min": "Mínimo provável", "casos_max": "Máximo provável",
                                  "incidencia_100k": "Por 100 mil hab.", "nivel_risco": "Nível"}),
                 hide_index=True, width="stretch",
                 column_config={"Semana": st.column_config.DateColumn(format="DD/MM/YYYY"),
                                "Casos previstos": st.column_config.NumberColumn(format="%d"),
                                "Mínimo provável": st.column_config.NumberColumn(format="%d"),
                                "Máximo provável": st.column_config.NumberColumn(format="%d"),
                                "Por 100 mil hab.": st.column_config.NumberColumn(format="%.1f")})

# ---------------------------------------------------------------- Modelo
with aba_modelo:
    st.success(
        "**Conclusão:** o LightGBM com objetivo Poisson prevê os casos de dengue melhor que todos "
        "os métodos testados, em todos os horizontes. Com 4 semanas de antecedência, erra 30% "
        "menos que repetir a semana atual. Ele não substitui o alerta oficial do InfoDengue, "
        "mas o complementa: usando os dois juntos, a detecção de epidemias com 4 semanas de "
        "antecedência sobe de 74% para 89%."
    )
    st.markdown(
        "O modelo foi treinado com dados de **2015 a 2023** e avaliado em **2024 a 2026**, "
        "período que ele nunca viu e que inclui a maior epidemia de dengue já registrada no "
        "estado."
    )
    st.subheader("Comparação entre métodos")
    if comparacao:
        cores = {"Repetir o mesmo período do ano anterior": "#c3c2b7",
                 "Regressão linear": "#1baf7a",
                 "Repetir a semana atual (baseline)": COR_BASE,
                 "Binomial negativa": "#6250d6",
                 "LightGBM em escala log": "#2a78d6",
                 "LightGBM Poisson (modelo escolhido)": COR_PREV}
        linhas = [{"Horizonte": int(k[1:]), "Método": metodo, "Erro médio (casos)": v}
                  for k, d in comparacao.items() for metodo, v in d.items() if metodo in cores]
        cmp = pd.DataFrame(linhas)
        fig = px.bar(cmp, x="Horizonte", y="Erro médio (casos)", color="Método",
                     barmode="group", color_discrete_map=cores,
                     category_orders={"Método": list(cores)})
        fig.update_layout(height=440, xaxis_title="Semanas à frente",
                          legend=dict(orientation="h", y=1.2, title=None),
                          margin=dict(l=0, r=0, t=60, b=0), bargap=0.2, bargroupgap=0.04)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width="stretch")
        tabela = cmp.pivot(index="Método", columns="Horizonte", values="Erro médio (casos)")
        tabela = tabela.loc[[m for m in cores if m in tabela.index]].rename(columns=lambda c: f"{c} sem.")
        st.dataframe(tabela.round(1), width="stretch")
        st.caption(
            "Erro médio absoluto, em casos por município por semana; menor é melhor. Repetir o ano "
            "anterior falha porque os anos são muito diferentes (2024 teve 5 vezes mais casos que "
            "2023). A regressão linear perde até do baseline: as relações não são lineares. A "
            "binomial negativa, modelo estatístico clássico para contagens, fica perto do "
            "LightGBM com 3 e 4 semanas de antecedência, mas perde com 1 semana. O objetivo "
            "Poisson, próprio para contagens, melhorou o LightGBM em 2% a 6%."
        )

    if alerta:
        st.subheader("Comparação com o alerta oficial do InfoDengue")
        st.markdown(
            "O InfoDengue (Fiocruz/FGV) classifica cada município em quatro níveis. O nível 4 "
            "indica incidência acima do limiar epidêmico do próprio município. A pergunta: "
            "o município estará no nível 4 daqui a algumas semanas?"
        )
        al = pd.DataFrame([{
            "Semanas à frente": int(k[1:]),
            "Alerta oficial de hoje: precisão": v["oficial"]["precisao"] * 100,
            "Alerta oficial de hoje: detecção": v["oficial"]["recall"] * 100,
            "Oficial + modelo: precisão": v["combinado"]["precisao"] * 100,
            "Oficial + modelo: detecção": v["combinado"]["recall"] * 100,
            "Novas epidemias detectadas pelo modelo": v["novas_detectadas_pelo_modelo"] * 100,
        } for k, v in alerta.items()])
        fig = go.Figure()
        fig.add_scatter(x=al["Semanas à frente"], y=al["Alerta oficial de hoje: detecção"],
                        name="Só o alerta oficial", mode="lines+markers",
                        line=dict(color=COR_BASE, width=2), marker=dict(size=9))
        fig.add_scatter(x=al["Semanas à frente"], y=al["Oficial + modelo: detecção"],
                        name="Alerta oficial + modelo", mode="lines+markers",
                        line=dict(color=COR_PREV, width=2), marker=dict(size=9))
        fig.update_layout(height=340, yaxis_title="Epidemias detectadas (%)",
                          xaxis=dict(title="Semanas à frente", tickvals=[1, 2, 3, 4]),
                          yaxis=dict(range=[60, 100]), hovermode="x unified",
                          legend=dict(orientation="h", y=1.15), margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(al.set_index("Semanas à frente").round(0), width="stretch")
        st.caption(
            "O alerta oficial de hoje é muito bom para dizer quem continuará em epidemia, mas, por "
            "definição, não antecipa municípios que ainda não entraram nela. O modelo detecta "
            "cerca de metade dessas novas epidemias. Usando os dois juntos, a detecção sobe; o "
            "custo é mais alarmes falsos (a precisão cai), uma escolha de gestão."
        )

    st.subheader("Previsão x realidade no período de teste")
    nomes_t = sorted(teste["municipio"].unique())
    cidade_t = st.selectbox("Município", nomes_t, key="cidade_teste",
                            index=nomes_t.index("São Paulo"))
    h_t = st.radio("Horizonte", [1, 2, 3, 4], index=1, horizontal=True)
    t = teste[(teste["municipio"] == cidade_t) & (teste["horizonte"] == h_t)]
    t = t.sort_values("data_alvo")
    fig = go.Figure()
    fig.add_scatter(x=t["data_alvo"], y=t["casos_reais"], name="Real",
                    line=dict(color=COR_REAL, width=2))
    fig.add_scatter(x=t["data_alvo"], y=t["casos_previstos"], name="Previsto pelo modelo",
                    line=dict(color=COR_PREV, width=2))
    fig.update_layout(height=400, hovermode="x unified", yaxis_title="Casos por semana",
                      legend=dict(orientation="h", y=1.08), margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Monitoramento
with aba_monit:
    st.markdown(
        "Cada previsão publicada fica registrada. Quando a semana prevista se consolida (cerca de "
        "4 semanas depois, por causa do atraso de notificação), comparamos o previsto com o real. "
        "Se o modelo ficar pior que o baseline por 4 semanas seguidas, a atualização semanal "
        "dispara um alerta."
    )
    if desempenho.empty:
        st.info("Ainda não há semanas consolidadas para avaliar.")
    else:
        hm = st.radio("Horizonte da previsão", [1, 2, 3, 4], index=3, horizontal=True,
                      format_func=lambda x: f"{x} semana{'s' if x > 1 else ''}", key="hm")
        dm = desempenho[desempenho["horizonte"] == hm].sort_values("semana_base")
        fig = go.Figure()
        fig.add_scatter(x=dm["semana_base"], y=dm["mae_modelo"], name="Modelo",
                        line=dict(color=COR_PREV, width=2), mode="lines+markers")
        fig.add_scatter(x=dm["semana_base"], y=dm["mae_baseline"], name="Baseline",
                        line=dict(color=COR_BASE, width=2), mode="lines+markers")
        fig.update_layout(height=360, hovermode="x unified", yaxis_title="Erro médio (casos)",
                          xaxis_title="Semana em que a previsão foi feita",
                          legend=dict(orientation="h", y=1.1), margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, width="stretch")
        ultimas = dm.tail(8)
        c1, c2 = st.columns(2)
        c1.metric("Erro médio do modelo, últimas 8 semanas", f"{ultimas['mae_modelo'].mean():.2f}")
        c2.metric("Erro médio do baseline, últimas 8 semanas",
                  f"{ultimas['mae_baseline'].mean():.2f}")
        st.caption("Na baixa temporada (julho a outubro) os casos mudam pouco de uma semana para "
                   "outra, e repetir a semana atual fica difícil de bater. O ganho do modelo "
                   "aparece nas fases de subida e queda.")
        if (dm["origem"] == "retroativo").any():
            st.caption("Semanas marcadas como retroativas vêm do período de teste, previstas por "
                       "modelos que não tinham visto aqueles dados. As semanas seguintes vêm das "
                       "previsões publicadas pela atualização semanal.")
    if not versoes.empty:
        st.subheader("Versões do modelo")
        st.dataframe(versoes[["versao", "dados_ate", "mae_h1", "mae_h4", "mae_baseline_h4"]]
                     .rename(columns={"versao": "Versão", "dados_ate": "Dados até",
                                      "mae_h1": "Erro 1 sem.", "mae_h4": "Erro 4 sem.",
                                      "mae_baseline_h4": "Baseline 4 sem."})
                     .sort_values("Versão", ascending=False),
                     hide_index=True, width="stretch")

st.divider()
st.caption("Projeto de portfólio · código aberto · os níveis de risco são ilustrativos e não "
           "substituem os boletins oficiais da vigilância epidemiológica.")
