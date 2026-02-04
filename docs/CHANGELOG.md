# SumOmniEval - Changelog

All notable changes to this project are documented here.

---

## [2.2.0] - 2026-02-03

### Data Folder Reorganization

Complete restructuring of the data folder following GitHub best practices with improved organization and documentation.

### Added

#### Data Processing Pipeline
- **download_cnn_dm.py** - Downloads CNN/DailyMail articles from HuggingFace
  - Uses streaming to save disk space
  - Downloads 10 sample articles from abisee/cnn_dailymail dataset
  - Outputs to `data/raw/cnn_dm_sample.json`
  - No API key required
- **generate_summaries.py** - Generates AI summaries using H2OGPTE API
  - Reads from `data/raw/` folder
  - Outputs to `data/processed/` folder
  - Uses GPT-4o by default (configurable)
  - Requires H2OGPTE_API_KEY and H2OGPTE_ADDRESS
- **create_templates.py** - Creates template files in multiple formats
  - Extracts 3 samples from processed data
  - Generates CSV, JSON, and XLSX templates
  - Demonstrates optional `reference_summary` field
  - Outputs to `data/examples/` folder

#### Folder Structure
- **data/raw/** - Raw downloaded data (gitignored)
- **data/processed/** - Processed data with AI summaries (gitignored)
- **data/scripts/** - Data processing scripts
- **data/examples/** - Template files (version controlled)

#### Documentation
- **data/README.md** - Consolidated, concise data folder documentation
  - Quick start guide for data processing pipeline
  - Script documentation with requirements
  - Format specifications and examples
  - Environment setup instructions

### Changed

#### File Organization
- Moved all processing scripts to `data/scripts/` subdirectory
- Separated raw data from processed data
- Moved template files to `data/examples/`
- Updated all script paths to use new folder structure

#### Version Control
- Updated `.gitignore` to exclude generated data files
  - Ignores `data/raw/*.json`
  - Ignores `data/processed/*.json`
  - Keeps template files in version control
- Removed redundant README files from subdirectories

#### Data Sources
- Added CNN/DailyMail dataset integration (Hermann et al., 2015)
- Template files now demonstrate optional fields with variance

#### Agent Utilities Simplification
- Simplified `load_summaries()` to accept `sample_idx` integer
- Changed `--sample` to `--sample-idx` for clarity
- Uses actual field names (`summary` instead of `generated_summary`)

### Removed

- **sample_summaries.json** - Obsolete test data file
- **data/scripts/README.md** - Consolidated into main README
- **data/raw/README.md** - Consolidated into main README
- **data/processed/README.md** - Consolidated into main README
- **ui/pages/1_Agent_Evaluation.py** - Only standalone eval on UI
- **ui/pages/2_MCP_Dashboard.py** - Only standalone eval on UI

### Fixed

- Script path resolution using `Path(__file__).parent`
- .gitignore consistency (`seed_data.csv` vs `sample_data.csv`)
- Template files now properly show optional `reference_summary` field

---

## [2.1.0] - 2026-02-02

### Agent and MCP Server Integration

Added orchestrator agent functionality and Model Context Protocol (MCP) server for enhanced agent-based evaluation workflows.

### Added

#### Agent Infrastructure
- **shared_utils.py** - Shared utility functions
  - `load_prompt()` - Loads prompts from markdown files
  - `load_summaries()` - Loads summary data from JSON
  - Cross-directory import support
- **agents/prompts/** - Organized prompt templates
  - `system.md` - System prompts for agents
  - `user.md` - User prompt templates

#### MCP Server
- **MCP server implementation** - Exposes evaluation functions via Model Context Protocol
  - Uses functions from `tool_logic.py`
  - Enables external tools to call evaluation metrics
  - Supports agent-based workflows
- **bundle.py** - Automated packaging script
  - Creates zip file for deployment
  - Simplifies distribution

#### Agent Modes
- **agent_mode** - H2OGPTe agent with direct function calling
- **agent_with_mcp** - H2OGPTe agent using MCP server integration

#### Frontend Features
- **Agent Evaluation page** - New Streamlit page for agent-based evaluation
- **MCP Dashboard page** - Monitor and manage MCP server connections
  - Note: Further testing required

#### Data Management
- **data/sample_summaries.json** - Sample data for agent testing
  - Moved from root to data folder

### Changed

#### Orchestrator Refactoring
- Refactored `orchestrator.py` to use shared functions
  - Replaced hardcoded values with utility functions
  - Improved modularity and maintainability
- Updated file path resolution for cross-directory imports

#### Build Configuration
- Updated `.gitignore` to exclude:
  - `.zip` files (agent bundles)
  - `mcp_server/dist_mcp/` (MCP server distribution artifacts)

### Fixed

- File path resolution for prompts and data across directories
- Cross-module imports for shared utilities

---

## [2.0.0] - 2026-01-29

### Standardize Evaluation Suite for Agentic Integration

Major architectural standardization to make evaluation metrics "Agent-ready" for seamless integration with tool-calling agents and future MCP server implementations.

### Added

#### Agent-Ready Functions
- **24 standalone evaluation functions** - Decomposed from monolithic evaluator classes
  - Explicit function schemas for each metric
  - Comprehensive docstrings with Purpose, Args, Returns, and Examples
  - Direct tool-calling registration support
  - Simplified integration with AI agents

#### Function Wrappers
- **High-level wrapper functions** - One for each of the 24 metrics
  - Consistent interface across all metrics
  - Standardized input validation
  - Explicit keyword arguments enforced

#### Testing Infrastructure
- **test_agent_function_calling.py** - Validates agent function calling
  - Tests all 24 metrics as standalone functions
  - Verifies schema compatibility
  - Ensures proper error handling
- **Reference implementation** - H2OGPTe agent example
  - Demonstrates tool-calling with evaluation metrics
  - Complete working example for integration

### Changed

#### Core Architecture
- **Refactored LLMJudgeEvaluator class** - Transitioned to modular functions
  - Removed tight coupling between metrics
  - Simplified testing and maintenance
  - Enhanced reusability across different contexts

#### Error Handling Standardization
- **Consistency fix for Error Scores**
  - Converted `0.0` error scores to `None`
  - Prevents statistical skewing in aggregate metrics
  - Improves data quality for analysis
- **Field normalization**
  - Success cases now return `Error: None` explicitly
  - Predictable JSON parsing for downstream tools
  - Clear distinction between scores and error states

#### Input Validation
- **Enforced keyword arguments** - All 24 metrics updated
  - Prevents positional argument errors
  - Improves code clarity
  - Better IDE autocomplete support
- **Schema validation** - Comprehensive input checking
  - Type validation for all parameters
  - Clear error messages for invalid inputs
  - Fail-fast approach for debugging

#### Documentation
- **Enhanced docstrings** - All 24 metrics updated with:
  - **Purpose**: What the metric measures
  - **Args**: Parameter descriptions with types
  - **Returns**: Return value structure with examples
  - **Examples**: Code snippets showing usage

### Breaking Changes

⚠️ **Function Signatures Changed**
- All evaluation functions now require **keyword arguments only**
- Positional arguments will raise `TypeError`

**Migration Guide:**
```python
# Before (positional)
score = evaluate_metric(source, summary)

# After (keyword)
score = evaluate_metric(source=source, summary=summary)
```

### Performance

- No performance degradation
- Function-based approach may enable future optimizations
- Better memory management with independent functions

---

## [1.1.0] - 2026-01-26

### Added

#### Dataset Upload Feature
- **Upload Your Own Datasets** - Upload files with multiple rows of data
  - Support for CSV (`.csv`) - Standard comma-separated format
  - Support for JSON (`.json`) - Array of objects format
  - Support for Excel (`.xlsx`, `.xls`) - Standard spreadsheet format
  - Support for TSV (`.tsv`) - Tab-separated format
  - Validation: Must have at least 2 columns
  - Validation: Must have at least 1 data row

#### Column Mapping
- **Smart Column Selection** - Choose which columns to use after upload
  - "Source Text Column" dropdown - Select column with full documents
  - "Summary Column" dropdown - Select column with summaries
  - Preview selected row before evaluation
  - Support for any column names (not limited to "report" or "summary")

#### Row Selection
- **Dataset Row Navigation** - Evaluate multiple document pairs
  - Dropdown shows placeholder: "-- Select a row --" (no auto-load)
  - User selects row: "Row 1", "Row 2", "Row 3", etc.
  - Once dataset uploaded, replaces sample data in dropdown
  - Selected row data loads into text areas automatically
  - Easy switching between rows for batch evaluation

#### Documentation
- **docs/FILE_FORMATS.md** - Complete dataset upload guide
  - Format specifications for CSV, JSON, Excel, TSV
  - Column selection workflow
  - Validation rules and error messages
  - Example datasets and workflow tutorial

#### Example Datasets
- **examples/example_dataset.csv** - CSV with 3 sample rows
- **examples/example_dataset.json** - JSON array with 3 sample rows
- **examples/example_dataset.xlsx** - Excel workbook with 3 sample rows

### Changed

#### UI Improvements
- Renamed "Sample Data" to "Upload Your Dataset" in sidebar
- Data selector dropdown shows uploaded rows or sample data
- Added column mapping dropdowns after file upload
- Added "Clear Uploaded Dataset" button
- Shows dataset info: "X rows × Y columns"
- Text areas cleared on new file upload (no auto-population)
- Row selection uses placeholder "-- Select a row --" (no auto-select Row 1)

#### Dependencies
- Added `openpyxl==3.1.2` for Excel file support

#### Documentation Updates
- Updated README.md with dataset upload workflow
- Updated project structure to show example datasets
- Updated FILE_FORMATS.md with placeholder dropdown workflow
- Removed old single-document example files

### Fixed

#### Clear Dataset Functionality
- **File uploader now clears properly** - Filename disappears after clicking clear
  - Used dynamic uploader key (`uploader_key_{counter}`) to force new widget
  - Increments counter on clear to create fresh uploader
- **Dataset stays cleared after clicking sample data** - No longer reverts to uploaded file
  - Tracks which uploader widget file came from
  - Prevents re-processing same file after clear
- **Re-uploading after clear works correctly** - Can upload same or different file
  - Detects new uploader widget vs new filename
  - Processes file correctly even if same filename as cleared file

#### Column Selection & Row Loading
- **Text areas cleared on new upload** - Source and Summary start empty
- **Row 1 no longer auto-selected** - User must explicitly choose row
  - Dropdown starts with "-- Select a row --" placeholder
  - Data only loads when user selects a row
- **Columns must be selected before row selection appears** - Clearer workflow
  - Column selection shown first
  - Data selector only appears after both columns mapped

---

## [1.0.0] - 2026-01-25

### Major Release: 15 Metrics Complete

Complete implementation of workshop plan metrics with comprehensive testing.

### Added

#### New Metrics (Era 3A)
- **FactCC (BERT)** - Local BERT-based consistency checker (~400MB)
  - Fine-tuned model for factual consistency
  - Binary classification: Consistent vs Inconsistent
  - Confidence scoring (0-1 scale)
  - Fast inference (~15 seconds)
  - UI checkbox: "Era 3A: Logic Checkers → + FactCC (BERT)"

#### New Metrics (Era 3B)
- **DAG (DeepEval)** - API-based decision tree evaluation
  - Structured 3-step evaluation process
  - Step 1: Factual Accuracy (0-2 points)
  - Step 2: Completeness (0-2 points)
  - Step 3: Clarity (0-2 points)
  - Total scoring: 0-6 points (normalized to 0-1)
  - UI checkbox: "Era 3B: AI Simulators → + DAG (DeepEval)"

#### Documentation
- **README.md** - Comprehensive project overview
- **docs/METRICS.md** - Detailed metric explanations
- **docs/SETUP.md** - Installation and configuration guide
- **docs/CHANGELOG.md** - This file
- **tests/README.md** - Testing documentation

#### Tests
- **test_all_new_metrics.py** - Comprehensive test suite
  - Tests FactCC with good/bad summaries
  - Tests DAG decision tree evaluation
  - Tests Era 3A integration (all 3 metrics)
  - Tests Era 3B integration (all 5 metrics)
- Moved all test files to `tests/` directory

### Changed

#### Model Configuration
- **Default model**: Changed to `meta-llama/Llama-3.3-70B-Instruct`
- **Model dropdown**: Limited to 3 tested models only
  - meta-llama/Llama-3.3-70B-Instruct
  - meta-llama/Meta-Llama-3.1-70B-Instruct
  - deepseek-ai/DeepSeek-R1
- Removed untested models from dropdown

#### Era Naming
- **Corrected**: "Era 4" → "Era 3B: AI Simulators"
- Aligned with workshop plan structure
- Updated all documentation and UI labels
- Renamed: `ERA4_LLM_JUDGE.md` → `ERA3B_LLM_JUDGE.md`

#### UI Improvements
- Added 2-column layout for Era 3A metrics
- Added step-by-step display for DAG evaluation
- Updated metric count: "Up to 15 metrics"
- Improved help text for new checkboxes

#### Project Structure
- Organized tests into `tests/` directory
- Organized documentation into `docs/` directory
- Cleaned up redundant markdown files
- Created modular documentation structure

### Fixed

- Model name verification (added "Meta-" prefix for Llama-3.1)
- Era 3B foolproofing - each dimension works individually
- Improved error handling for API metrics
- Fixed timeout issues with long evaluations

### Removed

- **Redundant files**:
  - FINAL_IMPLEMENTATION_SUMMARY.md (consolidated into README)
  - H2OGPTE_API_SUMMARY.md (consolidated into SETUP)
  - PROJECT_STRUCTURE.md (integrated into README)
  - QUICK_START.md (integrated into README)
  - Multiple duplicate test files

### Metrics Not Implemented

Documented reasons for skipped workshop plan metrics:

- **AlignScore** (~1.4GB) - Over 1GB budget constraint
- **QuestEval** - Cython compilation errors with spacy dependency
- **MENLI** - Redundant with existing NLI implementation
- **Prometheus** - Complex local model setup, would require significant infrastructure

### Workshop Plan Coverage

**Total**: 9/12 workshop metrics (75%)

**Implemented**:
- ✅ Era 1: All 5 word overlap metrics (ROUGE, BLEU, METEOR, Levenshtein, Perplexity)
- ✅ Era 2: Embedding metrics (BERTScore + MoverScore)
- ✅ Era 3A: NLI (DeBERTa-v3) + FactCC (BERT)
- ✅ Era 3A: FactChecker (LLM-powered API)
- ✅ Era 3B: G-Eval (4 dimensions: Faithfulness, Coherence, Relevance, Fluency)
- ✅ Era 3B: DAG (Decision tree evaluation)

**Skipped** (with valid technical reasons):
- ❌ AlignScore (budget)
- ❌ QuestEval (dependencies)
- ❌ MENLI (redundant)
- ❌ Prometheus (complexity)

---

## [0.9.0] - 2026-01-24

### Era 3B Implementation

#### Added
- **G-Eval** implementation with 4 dimensions:
  - Faithfulness: Factual accuracy evaluation
  - Coherence: Logical flow assessment
  - Relevance: Information coverage check
  - Fluency: Writing quality analysis
- **Era 3B UI** with individual metric toggles
- **LLM Judge** class for API-based evaluation
- **Configurable model selection** in UI

#### Changed
- Updated Era 3 to support both local (NLI) and API (LLM) metrics
- Enhanced UI with Era 3B section
- Added timeout configuration for API calls

---

## [0.8.0] - 2026-01-23

### Era 3A Implementation

#### Added
- **NLI (DeBERTa-v3)** - Natural Language Inference
- **FactChecker (API)** - LLM-based fact-checking
- Era 3A UI section with checkboxes
- H2OGPTE API integration

#### Changed
- Refactored Era 3 evaluator structure
- Added API key configuration via .env
- Updated requirements.txt with h2ogpte

---

## [0.7.0] - 2026-01-20

### Era 2 Implementation

#### Added
- **BERTScore** - Contextual embedding similarity
- **MoverScore** - Earth Mover's Distance alignment
- Model caching for faster subsequent runs

#### Changed
- Improved error handling for model downloads
- Added progress indicators for model loading

---

## [0.6.0] - 2026-01-18

### Era 1 Complete

#### Added
- **ROUGE** (1, 2, L variants)
- **BLEU**
- **METEOR**
- **Levenshtein Distance**
- **Perplexity**
- Basic Streamlit UI

#### Changed
- Organized evaluators by era
- Created modular metric structure

---

## [0.5.0] - 2026-01-15

### Initial Implementation

#### Added
- Project structure
- Basic ROUGE implementation
- requirements.txt
- Initial README

---

## Metric Evolution Summary

| Version | Era 1 | Era 2 | Era 3A | Era 3B | Total |
|---------|-------|-------|--------|--------|-------|
| 0.5.0   | 1     | 0     | 0      | 0      | 1     |
| 0.6.0   | 5     | 0     | 0      | 0      | 5     |
| 0.7.0   | 5     | 2     | 0      | 0      | 7     |
| 0.8.0   | 5     | 2     | 2      | 0      | 9     |
| 0.9.0   | 5     | 2     | 2      | 4      | 13    |
| **1.0.0** | **5** | **2** | **3** | **5** | **15** |

---

## Performance Evolution

| Version | Local Metrics | API Metrics | Evaluation Time |
|---------|---------------|-------------|-----------------|
| 0.6.0   | 5             | 0           | ~5s             |
| 0.7.0   | 7             | 0           | ~15s            |
| 0.8.0   | 8             | 1           | ~45s            |
| 0.9.0   | 8             | 5           | ~7min           |
| **1.0.0** | **9**     | **6**       | **~8min**       |

---

## Breaking Changes

### None in 1.0.0
All changes are backward compatible. Existing workflows continue to work.

### Future Considerations
- May add batch evaluation API
- May add result export functionality
- May add custom metric plugins

---

## Known Issues

### Current Limitations
1. **Era 3B speed**: API metrics take ~1-2 minutes each
   - Workaround: Use local metrics only for fast evaluation
2. **Model download size**: Initial download is ~3GB
   - Workaround: Pre-download models during setup
3. **Memory usage**: Full evaluation uses ~4GB RAM
   - Workaround: Run fewer metrics concurrently

### Planned Improvements
- [ ] Add result caching to avoid re-evaluation
- [ ] Add batch processing for multiple summaries
- [ ] Add progress bars for long evaluations
- [ ] Add result export (CSV, JSON)
- [ ] Add visualization charts

---

## Upgrade Guide

### From 0.9.0 to 1.0.0

**No breaking changes!** Simply:

1. Pull latest code
2. Update dependencies:
   ```bash
   pip install -r requirements.txt --upgrade
   ```
3. Run tests to verify:
   ```bash
   python3 tests/test_all_new_metrics.py
   ```

**New features available**:
- ✅ FactCC checkbox in Era 3A
- ✅ DAG checkbox in Era 3B
- ✅ 2 additional metrics (13 → 15)

---

## Contributors

- Initial implementation and all metrics
- Workshop plan alignment
- Documentation
- Testing suite

---

## License

See LICENSE file

---

## Version Summary

| Version | Focus Area | Key Achievement |
|---------|-----------|-----------------|
| 2.2.0 | Data Infrastructure | Organized data pipeline with CNN/DM integration |
| 2.1.0 | Agent Integration | MCP server and orchestrator agent |
| 2.0.0 | Architecture | Agent-ready standardization of 24 metrics |
| 1.1.0 | Dataset Upload | Multi-format file upload with column mapping |
| 1.0.0 | Metrics Complete | 15 evaluation metrics across 4 eras |

---

**Current Version**: 2.2.0
**Release Date**: 2026-02-03
**Status**: Production Ready ✅