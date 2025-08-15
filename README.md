# TraductionBD

Projet pour l'extraction, la traduction et l'analyse de texte à partir de fichiers bruts (HTML, PDF, images) via OCR et LLM.

## Structure

- `data/` : fichiers bruts (HTML, PDF, images)
- `outputs/` : texte OCR, texte traduit, résultats LLM
- `notebooks/` : expérimentations
- `src/` : code source
  - `config.py` : chemins, API keys, paramètres
  - `scraping.py` : scraping de pages ou fichiers
  - `ocr.py` : extraction texte depuis images ou PDF
  - `traduction.py` : traduction de texte
  - `llm_utils.py` : interaction avec LLM
  - `main.py` : pipeline principal
- `requirements.txt` : dépendances
