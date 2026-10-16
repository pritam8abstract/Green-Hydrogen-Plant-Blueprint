all:
	python scripts/run_all.py
test:
	pytest -q

dossier3:
	cd dossiers/03-project-book && xelatex -interaction=nonstopmode main.tex && xelatex -interaction=nonstopmode main.tex
