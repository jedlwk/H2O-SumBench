# Dataset Upload Guide

H2O.ai SumBench allows you to upload your own datasets for evaluation. This guide explains the supported formats and how to structure your data.

---

## Overview

**Upload Process:**
1. Upload a file with **multiple rows** (CSV, JSON, Excel, or TSV)
2. File must have **at least 2 columns** (source text and summary are required)
3. Select which column contains **Source Text**
4. Select which column contains **Summary**
5. Optional: Include **Reference Summary** column for ground truth comparison
6. Choose rows from dropdown to evaluate

**Once uploaded, the data selector will show your rows instead of sample data.**

---

## Required vs Optional Columns

### ✅ Required Columns

- **Source Text** - Full article or document text to be summarized
  - Common names: `source`, `report`, `document`, `article`, `text`
- **Summary** - Generated or human-written summary to evaluate
  - Common names: `summary`, `abstract`, `generated_summary`, `model_output`

### 🔄 Optional Columns

- **Reference Summary** - Ground truth summary for comparison metrics
  - Common names: `reference_summary`, `human_summary`, `ground_truth`, `highlights`
  - Can be empty or omitted for some rows
  - Useful when you have both model summaries and human references

---

## Supported Formats

### 1. CSV (Comma-Separated Values)

**Best for:** Exporting from spreadsheets (Excel, Google Sheets)

**Basic Structure (Required Fields Only):**
```csv
source,summary
"Full text of document 1...","Summary of document 1..."
"Full text of document 2...","Summary of document 2..."
"Full text of document 3...","Summary of document 3..."
```

**Extended Structure (With Optional Reference Summary):**
```csv
source,summary,reference_summary
"Full text of document 1...","Summary of document 1...","Ground truth summary 1..."
"Full text of document 2...","Summary of document 2...",""
"Full text of document 3...","Summary of document 3...","Ground truth summary 3..."
```

**Notes:**
- First row = column headers (can be any names)
- Quotes around text containing commas
- Standard UTF-8 encoding
- `reference_summary` can be empty for some rows (demonstrated in Row 2 above)

**Example:**
```csv
article,abstract,reference
"The quarterly earnings report shows that the company exceeded expectations with a 15% increase in revenue. The technology sector drove most of the growth, while retail remained flat.","Company Q3 earnings beat expectations with 15% revenue growth, driven by technology sector.","Q3 earnings exceeded expectations with 15% revenue increase, primarily from technology."
"Recent studies in climate science indicate that global temperatures have risen by 1.2°C since pre-industrial times. The Arctic region has experienced the most dramatic changes.","Global temperatures up 1.2°C since pre-industrial era, with Arctic showing greatest impact.",""
```

---

### 2. Excel (.xlsx or .xls)

**Best for:** Existing spreadsheet data

**Structure:**
- Standard Excel workbook
- First sheet will be used
- First row = column headers
- At least 2 columns required

**Notes:**
- Works with both `.xlsx` (modern) and `.xls` (legacy) formats
- Empty rows are automatically removed
- Empty columns are automatically removed

**Example:**
| document | summary |
|----------|---------|
| Full text of document 1... | Summary of document 1... |
| Full text of document 2... | Summary of document 2... |

---

### 3. JSON (Array of Objects)

**Best for:** API responses, programmatic data

**Basic Structure (Required Fields Only):**
```json
[
  {
    "source": "Full text of document 1...",
    "summary": "Summary of document 1..."
  },
  {
    "source": "Full text of document 2...",
    "summary": "Summary of document 2..."
  }
]
```

**Extended Structure (With Optional Reference Summary):**
```json
[
  {
    "source": "Full text of document 1...",
    "summary": "Summary of document 1...",
    "reference_summary": "Ground truth summary 1..."
  },
  {
    "source": "Full text of document 2...",
    "summary": "Summary of document 2..."
  },
  {
    "source": "Full text of document 3...",
    "summary": "Summary of document 3...",
    "reference_summary": "Ground truth summary 3..."
  }
]
```

**Notes:**
- Must be an array `[...]` of objects `{...}`
- Column names can be anything (you'll select them in the UI)
- UTF-8 encoding
- `reference_summary` field is optional and can be omitted entirely (as shown in object 2 above)

**Example:**
```json
[
  {
    "article_text": "The quarterly earnings report shows that the company exceeded expectations...",
    "summary_text": "Company Q3 earnings beat expectations with 15% revenue growth.",
    "human_summary": "Q3 earnings exceeded expectations with 15% revenue increase."
  },
  {
    "article_text": "Recent studies in climate science indicate that global temperatures...",
    "summary_text": "Global temperatures up 1.2°C since pre-industrial era."
  }
]
```

---

### 4. TSV (Tab-Separated Values)

**Best for:** Tab-delimited data exports

**Structure:**
```
report	summary
Full text of document 1...	Summary of document 1...
Full text of document 2...	Summary of document 2...
```

**Notes:**
- Columns separated by tabs (not spaces)
- First row = column headers
- Same as CSV but with tab delimiter

---

## Validation Rules

### ✅ File must pass these checks:

1. **At least 2 columns**
   - Need source column + summary column minimum
   - Can have additional columns (they'll be ignored)

2. **At least 1 data row**
   - Empty files are rejected
   - Empty rows are automatically removed

3. **Supported file format**
   - CSV (`.csv`)
   - Excel (`.xlsx`, `.xls`)
   - JSON (`.json`)
   - TSV (`.tsv`)

---

## Column Selection

After uploading, you'll see dropdowns for mapping your columns:

**Source Text Column:** (Required)
- Select the column with full documents
- Usually named: `report`, `source`, `document`, `article`, `text`, etc.

**Summary Column:** (Required)
- Select the column with summaries to evaluate
- Usually named: `summary`, `abstract`, `tldr`, `brief`, `generated_summary`, etc.
- Cannot be the same as source column

**Reference Summary Column:** (Optional)
- Select if you have ground truth summaries for comparison
- Usually named: `reference_summary`, `human_summary`, `ground_truth`, `highlights`, etc.
- Can be omitted if not needed for your evaluation

**Example:**
```
Your file has columns: ["article_id", "full_text", "generated_summary", "human_summary", "author"]

Select:
- Source Text Column: "full_text"
- Summary Column: "generated_summary"
- Reference Summary Column: "human_summary" (Optional)
```

The other columns (`article_id`, `author`) will be ignored.

---

## Row Selection

Once columns are mapped:

1. Dropdown shows: "-- Select a row --", "Row 1", "Row 2", "Row 3", etc.
2. Select a row to load its data into the text areas
3. Source and Summary text areas will populate automatically
4. Click "Evaluate Summary" to run metrics
5. Switch between rows to evaluate different pairs

---

## Error Messages

### ❌ "File must have at least 2 columns"
**Cause:** Your file has 0 or 1 columns
**Fix:** Add at least 2 columns with data

### ❌ "File is empty (no data rows)"
**Cause:** No data rows after header
**Fix:** Add at least 1 row of data

### ❌ "JSON must be an array of objects"
**Cause:** JSON is not formatted as `[{...}, {...}]`
**Fix:** Wrap objects in array brackets: `[{...}]`

### ❌ "Unsupported file format"
**Cause:** File extension not recognized
**Fix:** Use `.csv`, `.xlsx`, `.xls`, `.json`, or `.tsv`

---

## Templates

Download example files from the `data/examples/` folder:

### cnn_dm_template.csv
```csv
source,summary,reference_summary
"TechCorp announces new AI platform...","TechCorp launches AI platform...","Company unveils AI platform..."
"Climate summit reaches historic agreement...","Climate summit achieves breakthrough...",""
"New smartphone model breaks sales records...","Latest smartphone sets records...","Phone sales hit record numbers..."
```

### cnn_dm_template.json
```json
[
  {
    "source": "TechCorp announces new AI platform...",
    "summary": "TechCorp launches AI platform...",
    "reference_summary": "Company unveils AI platform..."
  },
  {
    "source": "Climate summit reaches historic agreement...",
    "summary": "Climate summit achieves breakthrough..."
  },
  {
    "source": "New smartphone model breaks sales records...",
    "summary": "Latest smartphone sets records...",
    "reference_summary": "Phone sales hit record numbers..."
  }
]
```

**Note:** Templates demonstrate that `reference_summary` is optional (Sample 2 omits it).

---

## Generating Sample Data

H2O.ai SumBench includes scripts to generate sample datasets from the CNN/DailyMail corpus:

**Step 1: Download CNN/DM articles**
```bash
python data/scripts/download_cnn_dm.py
```
- Downloads 10 articles from HuggingFace (abisee/cnn_dailymail)
- Uses streaming to save disk space
- Output: `data/raw/cnn_dm_sample.json`
- No API key required

**Step 2: Generate AI summaries** (Optional)
```bash
python data/scripts/generate_summaries.py
```
- Generates AI summaries using H2OGPTE API
- Uses GPT-4o by default
- Output: `data/processed/cnn_dm_sample_with_gen_sum.json`
- Requires `.env` file with `H2OGPTE_API_KEY` and `H2OGPTE_ADDRESS`

**Step 3: Create template files**
```bash
python data/scripts/create_templates.py
```
- Extracts 3 samples from processed data
- Generates CSV, JSON, and XLSX templates
- Output: `data/examples/cnn_dm_template.*`

**See [data/README.md](../data/README.md) for complete pipeline documentation.**

---

## Data Sources

### CNN/DailyMail Dataset
- **Source:** [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) on HuggingFace
- **Citation:** Hermann et al., 2015
- **Structure:** News articles with human-written highlights
- **Use case:** Ideal for testing summarization evaluation metrics

### Custom Datasets
- Upload any dataset following the formats above
- Required: source text + summary
- Optional: reference summary for ground truth comparison

---

## Tips

1. **Column names don't matter** - You'll select them in the UI
2. **Extra columns are OK** - Only source/summary columns are used
3. **Reference summary is optional** - Include only if you have ground truth
4. **Preview before evaluating** - Use the preview expander to check data
5. **CSV is simplest** - Best compatibility across all tools
6. **Excel works great** - Direct export from spreadsheets
7. **JSON for APIs** - If you're pulling data programmatically
8. **Use templates** - Download from `data/examples/` as starting point

---

## Workflow Example

**Step 1:** Export your data to CSV with columns:
- `document` (full text)
- `model_summary` (generated summary)
- `human_summary` (reference summary)

**Step 2:** Upload CSV to H2O.ai SumBench
- File appears in uploader
- Dataset info shows: "10 rows × 3 columns"

**Step 3:** Select columns:
- Source Text Column: `document`
- Summary Column: `model_summary`

**Step 4:** Select data row:
- Dropdown shows: "-- Select a row --"
- Choose "Row 1" to load data

**Step 5:** Click "Evaluate Summary"
- All metrics run on Row 1

**Step 6:** Review results, then select "Row 2" to evaluate next pair

---

## Clearing Uploaded Data

Click **"🗑️ Clear Uploaded Dataset"** to remove the uploaded file and return to sample data.

---

**Questions?** See [SETUP.md](SETUP.md) or create an issue on GitHub.
