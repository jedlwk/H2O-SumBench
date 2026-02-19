# Data Folder

Contains datasets, templates, and processing scripts for H2O SumBench.

## Structure

```
data/
├── scripts/          # Data processing pipeline
│   ├── download_cnn_dm.py       # Download CNN/DM from HuggingFace
│   ├── generate_summaries.py    # Generate AI summaries via H2OGPTE
│   └── create_templates.py      # Create template files
├── raw/              # Raw downloaded data 
├── processed/        # Processed data with AI summaries 
└── examples/         # Template files (CSV, JSON, XLSX)
```

## Quick Start

**Generate sample data:**
```bash
python data/scripts/download_cnn_dm.py        # Download 10 CNN/DM articles
python data/scripts/generate_summaries.py     # Generate AI summaries (needs API key)
python data/scripts/create_templates.py       # Create template files
```

**Use templates:**
1. Launch: `streamlit run app.py`
2. Upload a template from `data/examples/`
3. Select columns and evaluate

## Scripts

### download_cnn_dm.py
- Downloads CNN/DailyMail articles from HuggingFace using streaming
- Output: `raw/cnn_dm_sample.json` (10 articles)
- Requirements: `pip install datasets`

### generate_summaries.py
- Generates AI summaries using H2OGPTE API
- Input: `raw/cnn_dm_sample.json`
- Output: `processed/cnn_dm_sample_with_gen_sum.json`
- Requirements: `pip install h2ogpte python-dotenv`
- Needs `.env` file with `H2OGPTE_API_KEY` and `H2OGPTE_ADDRESS`

### create_templates.py
- Extracts 3 samples and creates templates in multiple formats
- Input: `processed/cnn_dm_sample_with_gen_sum.json`
- Output: `examples/template.{csv,json,xlsx}`
- Requirements: `pip install pandas openpyxl`

## Data Files

**Version controlled (included in repository):**
- `examples/template.*` - Sample templates in CSV, JSON, XLSX formats (3 samples)
- `raw/cnn_dm_sample.json` - Example raw CNN/DM articles (10 samples, ~50KB)
- `processed/cnn_dm_sample_with_gen_sum.json` - Example articles with AI summaries (10 samples, ~80KB)

**Why keep sample data in git?**
- Enables immediate testing without API keys
- Small file size (~100KB total)
- Better onboarding experience
- Scripts still available for regeneration/updates

**Gitignored (larger datasets, if generated):**
- `raw/*_large.json` - Large dataset downloads
- `processed/*_large.json` - Large processed datasets

## Format Examples

**Required columns:** `source` (article text), `summary` (summary text)

**Optional columns:** `reference_summary` (ground truth for comparison)

**CSV:**
```csv
source,summary,reference_summary
"Article text...","Summary...",""
```

**JSON:**
```json
[
  {
    "source": "Article text...",
    "summary": "Summary...",
    "reference_summary": "Optional..."
  }
]
```

## Environment Setup

Create `.env` in project root:
```env
H2OGPTE_API_KEY=your-key
H2OGPTE_ADDRESS=https://your-server
```

## Data Sources

- **CNN/DailyMail Dataset:** [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) (Hermann et al., 2015)
- **AI Summaries:** Generated using H2OGPTE API (GPT-4o by default)

---

For complete format specifications, see [docs/FILE_FORMATS.md](../docs/FILE_FORMATS.md)
