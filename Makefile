.PHONY: install dados treino comparar previsao monitor pipeline app testes lint

install:
	pip install -e ".[dev,analise]"

dados:
	python -m radar_dengue.collect

treino:
	python -m radar_dengue.model

comparar:
	python -m radar_dengue.compare

previsao:
	python -m radar_dengue.predict

monitor:
	python -m radar_dengue.monitor

pipeline: dados treino comparar previsao monitor

app:
	streamlit run app/streamlit_app.py

testes:
	pytest -q

lint:
	ruff check src app tests
