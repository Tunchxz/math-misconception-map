.PHONY: help env data clean-data notebooks app lint

PY := .venv/bin/python

help:
	@echo "env         crea el entorno virtual e instala las dependencias"
	@echo "data        descarga los datos crudos desde la API de Kaggle"
	@echo "notebooks   ejecuta todos los notebooks en orden"
	@echo "app         levanta la aplicacion de Streamlit"
	@echo "clean-data  borra los datos generados en data/transformed y data/processed"
	@echo "lint        formatea y revisa el codigo"

env:
	uv venv --python 3.12 .venv
	VIRTUAL_ENV=.venv uv pip install --system-certs -r requirements.txt
	VIRTUAL_ENV=.venv uv pip install --system-certs -e .
	$(PY) -m ipykernel install --user --name map-misconceptions --display-name "MAP (py3.12)"

data:
	$(PY) -m src.data.make_dataset

notebooks:
	for nb in notebooks/*.ipynb; do \
		$(PY) -m jupyter nbconvert --to notebook --execute --inplace $$nb || exit 1; \
	done

app:
	.venv/bin/streamlit run src/front_end/app.py

clean-data:
	rm -rf data/transformed/* data/processed/*

lint:
	$(PY) -m black src
	$(PY) -m ruff check src --fix
