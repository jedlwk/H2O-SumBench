# H2O.ai SumBench - Setup Guide

Complete installation and configuration guide.

---

## Quick Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**First install takes**: ~5-10 minutes
**Disk space needed**: ~3GB

### 2. Configure API Access

For API metrics (G-Eval, DAG, Prometheus), create `.env` file:

```bash
# Create .env in project root
cat > .env << 'EOF'
H2OGPTE_API_KEY=your_api_key_here
H2OGPTE_ADDRESS=https://your-instance.h2ogpte.com
EOF
```

### 3. Launch Application

```bash
streamlit run ui/app.py
```

App opens at: `http://localhost:8501`

---

## Detailed Installation

### Prerequisites

**Python Version**: 3.8 or higher
- Check: `python3 --version`
- If needed: Install from [python.org](https://www.python.org/downloads/)

**Pip Version**: 20.0 or higher
- Check: `pip3 --version`
- Upgrade: `pip3 install --upgrade pip`

**Operating System**:
- ✅ macOS (tested)
- ✅ Linux (tested)
- ✅ Windows (should work, but not extensively tested)

### Install Dependencies

```bash
# Clone or download the repository
cd H2O.ai SumBench

# Install all dependencies
pip install -r requirements.txt
```

**What gets installed**:
- Streamlit (web UI framework)
- Transformers (BERT, DeBERTa models)
- PyTorch (deep learning framework)
- H2OGPTE (API client)
- NLTK, SpaCy (NLP tools)
- Rouge, BERTScore, MoverScore (metrics)

### First Run Model Downloads

On first use, models auto-download:

| Dimension | Models | Size | Download Time |
|-----------|--------|------|---------------|
| Surface Overlap | NLTK data | ~100MB | <1 min |
| Semantic Alignment | BERTScore, MoverScore | ~1.7GB | 2-3 min |
| Faithfulness | DeBERTa-v3, BERT-MNLI, AlignScore | ~2.3GB | 2-3 min |
| Linguistic Quality | GPT-2 | ~600MB | 1-2 min |

**Total**: ~6GB, 6-10 minutes (one-time only)

Models cache to: `~/.cache/huggingface/`

---

## API Configuration

### H2OGPTE Setup (for API metrics)

**What you need**:
- H2OGPTE API key
- H2OGPTE instance URL

#### Step 1: Get API Credentials

Contact your H2OGPTE administrator for:
1. API key (looks like: `sk-...`)
2. Instance address (looks like: `https://your-instance.h2ogpte.com`)

#### Step 2: Create .env File

```bash
# In project root directory
nano .env
```

Add these lines:
```
H2OGPTE_API_KEY=sk-your-actual-key-here
H2OGPTE_ADDRESS=https://your-actual-instance.h2ogpte.com
```

Save and exit (Ctrl+O, Ctrl+X in nano)

#### Step 3: Verify Configuration

```bash
python3 tests/test_h2ogpte_api.py
```

**Expected output**:
```
✅ API connection successful
✅ Model query successful
```

### Without API Access

You can still use **14 local metrics**:
- No API configuration needed
- No internet required (after initial model download)
- 100% free

Simply skip the API configuration step.

---

## Verification

### Test All Metrics

```bash
# Test all metrics (requires .env for API metrics)
python3 -m pytest tests/test_all_metrics.py -v
```

### Expected Output

```
✅ ROUGE: Working
✅ BLEU: Working
✅ METEOR: Working
✅ BERTScore: Working
✅ MoverScore: Working
✅ NLI: Working
✅ FactCC: Working
✅ FactChecker: Working (if API configured)
✅ G-Eval: Working (if API configured)
✅ DAG: Working (if API configured)
```

---

## Common Issues

### Issue: "ModuleNotFoundError"

**Problem**: Missing Python package

**Solution**:
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: "Model download failed"

**Problem**: No internet or firewall blocking

**Solutions**:
- Check internet connection
- Configure proxy if needed:
  ```bash
  export http_proxy=http://proxy:port
  export https_proxy=http://proxy:port
  pip install -r requirements.txt
  ```

### Issue: "Out of memory"

**Problem**: System RAM insufficient for models

**Solutions**:
1. Close other applications
2. Use fewer metrics at once (disable Era 2 or 3A)
3. Increase swap space:
   ```bash
   # Linux
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Issue: "API connection failed"

**Problem**: Invalid API credentials or network

**Check**:
1. `.env` file exists in project root
2. API key is correct (no extra spaces)
3. Address is correct (include `https://`)
4. Network allows HTTPS connections

**Debug**:
```bash
python3 tests/test_h2ogpte_api.py
```

### Issue: "Invalid model" error

**Problem**: Model not available in your H2OGPTE instance

**Solution**: Use only tested models:
- `meta-llama/Llama-3.3-70B-Instruct` (default)
- `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `deepseek-ai/DeepSeek-R1`

**Check available models**:
```bash
python3 tests/test_corrected_models.py
```

### Issue: `pyemd` fails to install on Windows

**Problem**: `pyemd` requires a C++ compiler

**Solution**: Install Visual Studio Build Tools:
1. Download from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Select "Desktop development with C++" workload
3. Restart terminal and retry `pip install -r requirements.txt`

---

### Issue: Streamlit port already in use

**Problem**: Port 8501 already occupied

**Solution**: Use different port
```bash
streamlit run ui/app.py --server.port 8502
```

---

## Advanced Configuration

### Custom Model Cache Location

Change where models are downloaded:

```bash
# Set environment variable
export HF_HOME=/path/to/custom/cache
streamlit run ui/app.py
```

### Custom API Timeout

Edit `src/evaluators/era3_llm_judge.py`:

```python
# Line ~217
timeout=60  # Change to 120 for slower connections
```

### Disable Specific Metrics

Edit `ui/app.py` to hide metrics in UI:

```python
# Example: Hide Era 2
available = {
    'era1': True,
    'era2': False,  # Disable Era 2
    'era3': True,
}
```

---

## System Requirements

### Minimum Requirements

- **CPU**: 2 cores
- **RAM**: 8GB
- **Disk**: 5GB free
- **Internet**: Required for initial setup

### Recommended Requirements

- **CPU**: 4+ cores (faster parallel processing)
- **RAM**: 16GB (smooth operation with all metrics)
- **Disk**: 10GB free
- **Internet**: Stable connection for API metrics

### Performance Notes

**Local metrics** (14 metrics):
- CPU-only, no GPU needed
- ~30 seconds on modern laptop
- No ongoing internet required

**API metrics** (10 metrics):
- Network dependent
- ~2 minutes (depends on API latency)
- Requires stable internet

---

## Updating

### Update Dependencies

```bash
# Update all packages
pip install -r requirements.txt --upgrade

# Update specific package
pip install streamlit --upgrade
```

### Update Models

Models auto-update on launch if newer versions available.

To force re-download:
```bash
rm -rf ~/.cache/huggingface/transformers
```

---

## Uninstallation

### Remove Application

```bash
# Keep models for future use
rm -rf H2O.ai SumBench

# Remove models too (frees ~3GB)
rm -rf ~/.cache/huggingface
```

### Remove Python Packages

```bash
pip uninstall -r requirements.txt -y
```

---

## Docker Setup (Optional)

For containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py"]
```

Build and run:
```bash
docker build -t sumbench .
docker run -p 8501:8501 -v $(pwd)/.env:/app/.env sumbench
```

---

## Getting Help

### Check Documentation

1. **README.md** - Overview and quick start
2. **METRICS.md** - Detailed metric explanations
3. **tests/README.md** - Testing guide
4. **CHANGELOG.md** - Recent changes

### Run Diagnostics

```bash
# Check Python version
python3 --version

# Check installed packages
pip list | grep -E "(streamlit|transformers|torch|h2ogpte)"

# Test all metrics
python3 -m pytest tests/test_all_metrics.py -v

# Test API
python3 tests/test_h2ogpte_api.py
```

### Common Diagnostic Commands

```bash
# Check disk space
df -h

# Check RAM usage
free -h  # Linux
top      # macOS/Linux

# Check internet
ping google.com

# Check Python path
which python3

# Check pip path
which pip3
```

---

## Production Deployment

### Security Checklist

- [ ] Move API keys to secure secret manager
- [ ] Use environment variables, not .env file
- [ ] Enable HTTPS
- [ ] Set up authentication
- [ ] Configure firewall rules
- [ ] Enable logging
- [ ] Set up monitoring

### Performance Tuning

```python
# In app.py, add caching
@st.cache_resource
def load_models():
    # Load models once
    pass

@st.cache_data
def compute_metrics(source, summary):
    # Cache results
    pass
```

### Scaling Considerations

- Use Redis for caching
- Deploy multiple instances behind load balancer
- Queue API requests to avoid rate limits
- Monitor API usage and costs

---

**Last Updated**: 2026-02-07
**Support**: See README.md for contact info
