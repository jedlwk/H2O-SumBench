# Example Datasets

This folder contains example dataset files you can upload to SumOmniEval to test the evaluation system.

## Files

### 1. example_dataset.csv
- **Format:** CSV (Comma-Separated Values)
- **Rows:** 3 sample documents
- **Columns:** `source`, `summary`
- **Best for:** Testing CSV upload and understanding the basic format

### 2. example_dataset.json
- **Format:** JSON (Array of Objects)
- **Rows:** 3 sample documents
- **Columns:** `source`, `summary`
- **Best for:** Testing JSON upload and programmatic data

### 3. example_dataset.xlsx
- **Format:** Excel Workbook
- **Rows:** 3 sample documents
- **Columns:** `source`, `summary`
- **Best for:** Testing Excel upload from spreadsheet exports

## Sample Content

All three files contain the same 3 document pairs:

**Row 1:** TechCorp quarterly earnings report
**Row 2:** Climate science research findings
**Row 3:** New smartphone product announcement

## How to Use

1. **Launch SumOmniEval:**
   ```bash
   streamlit run app.py
   ```

2. **Upload a dataset:**
   - Click "📤 Upload Your Dataset" in the sidebar
   - Select one of the example files from this folder

3. **Map columns:**
   - Select "source" for Source Text Column
   - Select "summary" for Summary Column

4. **Select a row:**
   - Dropdown shows: "-- Select a row --"
   - Choose "Row 1", "Row 2", or "Row 3"
   - Data loads into text areas automatically

5. **Evaluate:**
   - Click "Evaluate Summary"
   - View results across all 15 metrics

## Creating Your Own Dataset

See [docs/FILE_FORMATS.md](../docs/FILE_FORMATS.md) for:
- Detailed format specifications
- Validation rules
- Workflow examples
- Troubleshooting tips

## Format Quick Reference

### CSV Format
```csv
source,summary
"Full text here...","Summary here..."
```

### JSON Format
```json
[
  {"source": "Full text here...", "summary": "Summary here..."}
]
```

### Excel Format
- First row: Column headers
- Data rows: Your documents and summaries
- Standard `.xlsx` or `.xls` format

---

**Questions?** See [../docs/FILE_FORMATS.md](../docs/FILE_FORMATS.md) for complete documentation.
