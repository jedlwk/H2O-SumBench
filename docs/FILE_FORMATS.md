# Dataset Upload Guide

SumOmniEval allows you to upload your own datasets for evaluation. This guide explains the supported formats and how to structure your data.

---

## Overview

**Upload Process:**
1. Upload a file with **multiple rows** (CSV, JSON, Excel, or TSV)
2. File must have **at least 2 columns**
3. Select which column contains **Source Text**
4. Select which column contains **Summary**
5. Choose rows from dropdown to evaluate

**Once uploaded, the data selector will show your rows instead of sample data.**

---

## Supported Formats

### 1. CSV (Comma-Separated Values)

**Best for:** Exporting from spreadsheets (Excel, Google Sheets)

**Structure:**
```csv
report,summary
"Full text of document 1...","Summary of document 1..."
"Full text of document 2...","Summary of document 2..."
"Full text of document 3...","Summary of document 3..."
```

**Notes:**
- First row = column headers (can be any names)
- Quotes around text containing commas
- Standard UTF-8 encoding

**Example:**
```csv
article,abstract
"The quarterly earnings report shows that the company exceeded expectations with a 15% increase in revenue. The technology sector drove most of the growth, while retail remained flat.","Company Q3 earnings beat expectations with 15% revenue growth, driven by technology sector."
"Recent studies in climate science indicate that global temperatures have risen by 1.2°C since pre-industrial times. The Arctic region has experienced the most dramatic changes.","Global temperatures up 1.2°C since pre-industrial era, with Arctic showing greatest impact."
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

**Structure:**
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

**Alternative column names:**
```json
[
  {
    "report": "Full text...",
    "abstract": "Summary..."
  }
]
```

**Notes:**
- Must be an array `[...]` of objects `{...}`
- Column names can be anything (you'll select them in the UI)
- UTF-8 encoding

**Example:**
```json
[
  {
    "article_text": "The quarterly earnings report shows that the company exceeded expectations...",
    "summary_text": "Company Q3 earnings beat expectations with 15% revenue growth."
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

After uploading, you'll see two dropdowns:

**Source Text Column:**
- Select the column with full documents
- Usually named: `report`, `source`, `document`, `article`, `text`, etc.

**Summary Column:**
- Select the column with summaries
- Usually named: `summary`, `abstract`, `tldr`, `brief`, etc.
- Cannot be the same as source column

**Example:**
```
Your file has columns: ["article_id", "full_text", "generated_summary", "author"]

Select:
- Source Text Column: "full_text"
- Summary Column: "generated_summary"
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

## Example Datasets

Download example files from the `examples/` folder:

### example_dataset.csv
```csv
source,summary
"Text 1...","Summary 1..."
"Text 2...","Summary 2..."
"Text 3...","Summary 3..."
```

### example_dataset.json
```json
[
  {"source": "Text 1...", "summary": "Summary 1..."},
  {"source": "Text 2...", "summary": "Summary 2..."},
  {"source": "Text 3...", "summary": "Summary 3..."}
]
```

---

## Tips

1. **Column names don't matter** - You'll select them in the UI
2. **Extra columns are OK** - Only source/summary columns are used
3. **Preview before evaluating** - Use the preview expander to check data
4. **CSV is simplest** - Best compatibility across all tools
5. **Excel works great** - Direct export from spreadsheets
6. **JSON for APIs** - If you're pulling data programmatically

---

## Workflow Example

**Step 1:** Export your data to CSV with columns:
- `document` (full text)
- `model_summary` (generated summary)
- `human_summary` (reference summary)

**Step 2:** Upload CSV to SumOmniEval
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
