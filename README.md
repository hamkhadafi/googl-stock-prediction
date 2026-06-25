# GOOGL Stock Movement Prediction

A multi-horizon stock return prediction system for GOOGL (Alphabet Inc.) comparing three model architectures — LightGBM, Temporal Fusion Transformer, and iTransformer — across 1-day, 5-day, and 20-day forecast horizons.

> Academic research exercise. Not financial advice.

---

## Results (Test Set: Jan 2024 – Jun 2026)

### Sharpe Ratio (simulated long/flat strategy)

| Model | 1d | 5d | 20d |
|---|---|---|---|
| LightGBM | 0.286 | 0.843 | **1.613** |
| TFT | 0.391 | 0.720 | 0.985 |
| iTransformer | **1.091** | **1.832** | 1.089 |
| Buy & Hold | 1.484 | 1.532 | 1.526 |

### Directional Accuracy

| Model | 1d | 5d | 20d |
|---|---|---|---|
| LightGBM | 0.470 | 0.494 | **0.645** |
| TFT | 0.464 | 0.485 | 0.477 |
| iTransformer | **0.514** | **0.559** | 0.626 |
| Base Rate | 0.553 | 0.589 | 0.649 |

**Key finding:** LightGBM achieves lowest MAE across all horizons. iTransformer achieves highest Sharpe ratio and directional accuracy at 1d/5d despite higher MAE — evidence that return magnitude prediction and direction prediction are distinct tasks with different optimal architectures.

---

## Data Sources

| Source | Data | Coverage |
|---|---|---|
| Yahoo Finance (`yfinance`) | Daily OHLCV (adjusted) | 2010–present |
| SEC EDGAR XBRL API | Quarterly financial statements | 2012–present |
| FRED API (free key required) | Macroeconomic indicators | 2010–present |

> **Note on news data:** Finnhub and GDELT were evaluated. Finnhub free tier covers ~2 years; GDELT Doc API covers a 3-month rolling window. Neither provides full 2014–2026 coverage. News modality was excluded and documented as a known limitation.

---

## Feature Engineering

**Feature matrix:** 3,134 rows × 157 columns (2014–2026)

| Group | Count | Examples |
|---|---|---|
| Technical | 70 | Log returns, SMA/EMA, MACD, RSI, Stochastic, Bollinger Bands, ATR, ADX, OBV, VWAP, Donchian/Keltner channels, Ichimoku |
| Fundamental | 47 | Margins, ROE/ROA/ROIC, Piotroski F-Score, Altman Z-Score, QoQ growth, accruals ratio |
| Availability masks | 3 | Binary flags for sparse EDGAR columns (dda, short_term_investments, long_term_debt) |
| Macro | 26 | Fed funds rate, VIX, yield curve spread, real rate, CPI, GDP, oil, USD index |
| Valuation | 6 | P/E, P/B, P/S, EV/EBITDA, FCF yield, log market cap |

**Walk-forward split:** Train 2014–2021 / Val 2022–2023 / Test 2024–2026

---

## Models

### LightGBM — `04_modeling_lgbm.ipynb`
Three independent GBDT models (one per horizon). 111 features including non-stationary levels (tree splits handle these natively). Conservative hyperparameters given ~2,000 training samples. Early stopping on validation MAE.

### Temporal Fusion Transformer — `05_modeling_tft.ipynb`
Lim et al., NeurIPS 2021. Implemented via `pytorch-forecasting`. Variable Selection Networks provide soft per-timestep feature selection. 89 stationary/normalized features. ~1M parameters. **Requires GPU.**

### iTransformer — `06_modeling_itransformer.ipynb`
Liu et al., ICLR 2024. Implemented from scratch in PyTorch. Inverts the standard Transformer: attention is applied across variates (features), not across time — each feature's full time series becomes a token. ~106K parameters. **Requires GPU.**

---

## Repository Structure

```
googl-stock-prediction/
├── 01_data_collection.ipynb
├── 02_feature_engineering.ipynb
├── 03_eda.ipynb
├── 04_modeling_lgbm.ipynb          # CPU sufficient
├── 05_modeling_tft.ipynb           # GPU required
├── 06_modeling_itransformer.ipynb  # GPU required
├── 07_evaluation.ipynb
│
├── data/
│   ├── raw/                        # price_raw.csv, macro_raw.csv, fundamental_raw.csv
│   ├── processed/                  # features.csv, predictions, metrics, feature lists
│   └── picture/                    # all exported plots (PNG)
│
├── models/
│   ├── lgbm_1d.txt / lgbm_5d.txt / lgbm_20d.txt
│   ├── tft_1d/ tft_5d/ tft_20d/    # best.ckpt per horizon
│   └── itransformer_1d.pt / itransformer_5d.pt / itransformer_20d.pt
├── whitepaper_googl_prediction.pdf
└── README.md
```

> `data/` and `models/` are excluded from version control via `.gitignore` due to file size. See **Reproduction** below to regenerate all artifacts.

---

## Reproduction

### 1. Clone and install

```bash
git clone https://github.com/your-username/googl-stock-prediction
cd googl-stock-prediction
pip install yfinance fredapi pandas numpy matplotlib seaborn scipy statsmodels scikit-learn lightgbm pandas-ta tqdm requests
```

### 2. Get a free FRED API key

Register at https://fred.stlouisfed.org/docs/api/api_key.html, then set it before running Part 1:

```python
import os
os.environ["FRED_API_KEY"] = "your_key_here"
```

### 3. Run notebooks in order

| Notebook | Runtime | Input | Output |
|---|---|---|---|
| `01_data_collection` | Colab / local | — | `data/raw/` |
| `02_feature_engineering` | Colab / local | `data/raw/` | `data/processed/features.csv` |
| `03_eda` | Colab / local | `data/processed/` | `selected_features.json` |
| `04_modeling_lgbm` | Colab / local | `data/processed/` | `models/lgbm_*.txt` |
| `05_modeling_tft` | **Kaggle GPU** | `data/processed/` | `models/tft_*/` |
| `06_modeling_itransformer` | **Kaggle GPU** | `data/processed/` | `models/itransformer_*.pt` |
| `07_evaluation` | Colab / local | predictions + metrics | `comparison_table.csv` |

For GPU notebooks (Parts 5–6), upload `data/processed/` as a Kaggle dataset and update:
```python
PROCESSED_DIR = "/kaggle/input/<your-dataset-name>/data/processed"
MODEL_DIR     = "/kaggle/working/models"
```

### 4. Dependency note for TFT (Part 5)

```bash
pip install pytorch-forecasting "lightning>=2.0.0"
```

Use `import lightning.pytorch as pl` — **not** `import pytorch_lightning`. The old import path breaks in Lightning v2.x and causes a `TypeError` where `TemporalFusionTransformer` is not recognized as a `LightningModule`.

---

## Known Limitations

- **Single ticker.** Results are specific to GOOGL dynamics and period.
- **No news/sentiment.** Free historical news sources don't cover 2014–2026.
- **Small training set (~2,000 samples).** TFT is under-resourced; iTransformer 5d/20d shows overfitting.
- **Strong bull test period.** GOOGL 2024–2026 is a hard benchmark for any long/flat strategy.
- **No hyperparameter search.** Results use reasonable hand-tuned defaults.
- **Simulated strategy only.** No position sizing, no shorting, no risk management.

---

## Tech Stack

`Python` · `pandas` · `numpy` · `yfinance` · `fredapi` · `LightGBM` · `PyTorch` · `Lightning` · `pytorch-forecasting` · `scikit-learn` · `pandas-ta` · `matplotlib` · `seaborn` · `SEC EDGAR XBRL API`

---

## References

1. Y. Liu et al., "iTransformer," ICLR 2024.
2. B. Lim et al., "Temporal Fusion Transformers," IJF 2021.
3. L. Grinsztajn et al., "Why Tree-based Models Still Outperform Deep Learning on Tabular Data," NeurIPS 2022.

---

## Author

**Ilham Khadafi**
ilhamkhadafi.dkh@gmail.com
[github.com/hamkhadafi](https://github.com/hamkhadafi)
