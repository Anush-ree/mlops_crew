# MLOps Crew — Phishing Email Detection System

A machine learning system to detect phishing emails, built with MLOps best practices using DVC for data versioning, and structured with the SE489 MLOps cookiecutter template.

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Anush-ree/mlops_crew.git
cd mlops_crew
```

### 2. Set up the environment
```bash
make install
```

For dev tools (linting, pre-commit hooks):
```bash
make dev
```

### 3. Set up DVC credentials
Ask the project owner for:
- Access to the shared Google Drive folder (`phishing-data`)
- The DVC credentials (`gdrive_client_id` and `gdrive_client_secret`)

Then configure DVC locally (these are NOT committed to git):
```bash
dvc remote modify --local storage gdrive_client_id "<client_id>"
dvc remote modify --local storage gdrive_client_secret "<client_secret>"
```

### 4. Pull the data
```bash
dvc pull
```
This downloads all datasets from Google Drive into `data/raw/archive/`.

---

## Project Structure

```
mlops_crew/
├── data/
│   ├── raw/
│   │   └── archive/          ← raw datasets (managed by DVC, not in git)
│   └── processed/            ← cleaned/processed data (managed by DVC)
├── mlops_crew/
│   ├── data/                 ← data processing scripts
│   ├── models/               ← model training scripts
│   └── predict_model.py      ← inference scripts
├── tests/                    ← unit tests
├── notebooks/                ← exploratory notebooks
├── .dvc/                     ← DVC configuration
├── Makefile                  ← common commands
└── requirements.txt          ← dependencies
```

---

## Dataset

| File | Description |
|------|-------------|
| `SpamAssasin.csv` | SpamAssassin public email corpus |
| `CEAS_08.csv` | CEAS 2008 spam competition dataset |
| `Enron.csv` | Enron email dataset |
| `Ling.csv` | Ling spam dataset |
| `Nazario.csv` | Nazario phishing dataset |
| `Nigerian_Fraud.csv` | Nigerian fraud email dataset |
| `phishing_email.csv` | General phishing email dataset |

> **Note:** Raw data is stored on Google Drive via DVC and is **not** committed to this repo. Run `dvc pull` to download it.

---

## Available Commands

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make dev` | Install dev dependencies + pre-commit hooks |
| `make test` | Run test suite |
| `make lint` | Lint code with ruff |
| `make format` | Auto-format code with ruff |
| `make clean` | Remove build/cache artifacts |
| `dvc pull` | Download data from Google Drive |
| `dvc push` | Upload data to Google Drive |

---

## Team Workflow

### First time setup
```bash
git clone https://github.com/Anush-ree/mlops_crew.git
cd mlops_crew
make install
```

Configure DVC credentials locally (ask project owner for the values — these are stored locally only and never committed to git):
```bash
dvc remote modify --local storage gdrive_client_id "<client_id>"
dvc remote modify --local storage gdrive_client_secret "<client_secret>"
```

Then pull the data:
```bash
dvc pull
```

### Day-to-day workflow
```bash
git pull                  # get latest code
dvc pull                  # get latest data
# ... make changes ...
git add .
git commit -m "your message"
git push
dvc push                  # if you changed/added data
```

---

## Tech Stack

- **Python 3.13**
- **DVC** — data version control
- **Google Drive** — remote data storage
- **Ruff** — linting and formatting
- **Pytest** — testing
- **Pre-commit** — code quality hooks
- **MkDocs** — documentation

---

## Contact

For DVC credentials or Google Drive access, contact the project owner.