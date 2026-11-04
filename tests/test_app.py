from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data/processed/previsao_atual.parquet").exists(),
                    reason="dados processados ausentes")
def test_dashboard_abre_sem_erros():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert len(at.metric) >= 3
    assert len(at.tabs) == 4
