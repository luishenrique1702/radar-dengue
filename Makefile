.PHONY: sarimax install dados treino ajuste backtest experimentos comparar previsao monitor sql pipeline app api testes lint

install:
	pip install -e ".[dev,analise,api]"

dados:
	python -m radar_dengue.collect

treino:
	python -m radar_dengue.model

ajuste:          # busca de hiperparâmetros (~25 min)
	python -m radar_dengue.tuning

backtest:        # backtest de 2019 a 2026 com WIS (~1 h)
	python -m radar_dengue.backtest

experimentos:    # vizinhos e tendência do Rt (~15 min)
	python -m radar_dengue.experimentos

sarimax:
	python -m radar_dengue.sarimax

comparar:
	python -m radar_dengue.compare

previsao:
	python -m radar_dengue.predict

monitor:
	python -m radar_dengue.monitor

sql:
	python -m radar_dengue.warehouse

pipeline: dados treino comparar previsao monitor sql

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn radar_dengue.api:app --reload

testes:
	pytest -q

lint:
	ruff check src app tests
