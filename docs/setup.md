# Setup

Full install for Windows and Linux/macOS. The README has the five-line version.

## 1. Get the code

```bash
git clone https://github.com/Farahani1/rag-dlai-local.git
cd rag-dlai-local
```

## 2. Install Python dependencies

You need **Python 3.12 or newer** (tested with 3.12 on Linux and 3.14 on Windows 11). The W1 notebook uses f-strings that nest the same quote type (PEP 701), which is a syntax error on 3.11.

Windows (PowerShell):
```powershell
python -m venv .env
.env\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell refuses to run the activation script, allow local scripts once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt and run `.env\Scripts\activate.bat` instead. In Git Bash on Windows, activate with `source .env/Scripts/activate` (`Scripts`, not `bin`).

Activate the environment **before** running `pip install`. Otherwise pip installs everything into your main Python instead of the project's `.env`. To check, `which python` (Git Bash) or `Get-Command python` (PowerShell) should point into `.env`.

Linux / macOS:
```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists only the packages the notebooks and modules import directly. The exact versions this project was tested with (Windows 11, Python 3.14) are in `requirements-lock-windows.txt`.

## 3. Install Ollama and pull a model

Download Ollama from https://ollama.com/download and install it, then pull a small model:

```bash
ollama pull gemma3:1b
```

For W5, `qwen2.5:1.5b` is the model the adaptation was verified against:

```bash
ollama pull qwen2.5:1.5b
```

Start Ollama (`ollama serve`, or open the Ollama app) before running a notebook. If it isn't running, the first LLM call stops with a clear "Failed to connect to Ollama" error.

Other models that run on modest hardware: `phi3:mini`, `gemma3:4b` (quantised, e.g. `google/gemma-3-4b-it-qat-q4_0`). The embedding model is `sentence-transformers/all-MiniLM-L6-v2`; `multi-qa-MiniLM-L6-cos-v1` also works, and `ms-marco-MiniLM-L-6-v2` is a reranker useful from Week 3 onward.

## 4. Create the config file

```bash
cp config_example.yaml config.yaml
```

Then set your model names in `config.yaml`:

```yaml
models:
  ollama:
    modelName: "gemma3:1b"
    url: http://localhost:11434/api/generate
  embeddingModel: "sentence-transformers/all-MiniLM-L6-v2"
```

All file paths used by any week (embeddings, CSVs, Chroma folders) are declared under `data:` in `config.yaml` and resolved relative to the project root by `setting.py`. Forward and back slashes both work, on every operating system.

## 5. Populate the local vector databases (W4–W5)

W1 and W2 work directly off the CSV and joblib files under `data/`. W3 builds its Chroma collection in the notebook's own setup cell, from `data/news_data_dedup.csv` and the provided `data/embeddings.joblib`.

W4 and W5 read from Chroma collections that must be built once:

```bash
# W4: product catalogue used by the chatbot
python w4/populate_products.py

# W5: its own product collection and the FAQ collection
python w5/populate_products.py
python w5/populate_faq.py
```

These are one-time, CPU-bound scripts (encoding about 44,000 products takes a few minutes on modest hardware). Nothing runs them automatically. Re-run them if you delete `data/chroma_db_products/`, and re-run `populate_faq.py` after editing `data/faq.yaml` (rebuild `data/faq.joblib` first; the command is in the YAML file's header).

## 6. Run a notebook

```bash
jupyter notebook
```

Open any week's `wN/C1MN_Assignment.ipynb` and run all cells top to bottom. The notebooks expect to be started from their own folder, which is Jupyter's default when you open them from the file browser.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Failed to connect to Ollama. Is it running?` | Start the Ollama app or `ollama serve`. |
| `model "…" not found` from Ollama | The model in `config.yaml` isn't pulled: `ollama pull <name>`. |
| `Collection 'products_w5' not found` | Run the population scripts in step 5. |
| `SyntaxError` in the W1 notebook | Python older than 3.12. |
| `Config file not found` | Copy `config_example.yaml` to `config.yaml` (step 4). |
| PowerShell won't activate the venv | See the execution-policy note in step 2. |
