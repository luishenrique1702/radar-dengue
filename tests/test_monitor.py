import pandas as pd

from radar_dengue import monitor


def test_monitoramento_compara_previsto_e_real(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "MONITOR", tmp_path)
    monkeypatch.setattr(monitor, "LOG", tmp_path / "log.parquet")
    monkeypatch.setattr(monitor, "DESEMPENHO", tmp_path / "desempenho.parquet")

    datas = pd.date_range("2025-01-05", periods=12, freq="7D")
    raw = pd.DataFrame({"geocode": "1", "data_iniSE": datas, "casos": range(10, 22)})
    prev = pd.DataFrame({
        "semana_base": [datas[0]], "geocode": ["1"], "horizonte": [2],
        "data_alvo": [datas[2]], "casos_previstos": [13.0],
        "casos_min": [11.0], "casos_max": [15.0],
    })
    monitor.log_forecast(prev)
    monitor.log_forecast(prev)  # executar duas vezes não duplica
    assert len(pd.read_parquet(monitor.LOG)) == 1

    des = monitor.evaluate(raw)
    linha = des.iloc[0]
    assert linha["mae_modelo"] == 1   # real 12, previsto 13
    assert linha["mae_baseline"] == 2  # baseline 10
    assert linha["cobertura"] == 1


def test_alerta_quando_modelo_piora():
    des = pd.DataFrame({"semana_base": pd.date_range("2025-01-05", periods=4, freq="7D"),
                        "horizonte": 4, "razao": [1.5, 1.4, 1.3, 1.6]})
    assert monitor.check(des)
    assert not monitor.check(des.assign(razao=[1.5, 0.9, 1.3, 1.6]))
