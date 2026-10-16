# Dossiers

| Folder | Title | What it answers |
|---|---|---|
| `01-without-the-hype/` | *Green Hydrogen Without the Hype* | What hydrogen is, why India needs it clean, and which electrolyser stack to bet on |
| `02-where-the-money-is/` | *Where the Money Is* | Where profit pools sit in the value chain; five entry points for a new venture |
| `03-project-book/` | *The Project Book* | Site, schedule, operations, finance, risk, real options and the research paper for the chosen 100 MW hybrid plant |

Compiled PDFs are in `pdf/`. To rebuild one (XeLaTeX, fonts in `assets/fonts/`):

```bash
cd dossiers/03-project-book
xelatex main.tex && xelatex main.tex
```

Dossier 3 reads its figures straight from `../../figures/`, so run `python scripts/run_all.py` first if you change any assumption.
