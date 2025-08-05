# 🔬 Nano-Cap Trading Universe

## Overview
This document lists the 50 nano-cap stocks (market cap < $350M) used for strategy evaluation and benchmarking in the NanoCap Trader system.

## Stock Universe (50 Symbols)

### 📊 Healthcare & Biotech (16 stocks)
- **ADTX** - Aditxt Inc. - Immune monitoring platform
- **APTO** - Aptose Biosciences - Oncology-focused biopharmaceutical
- **AVIR** - Atea Pharmaceuticals - Antiviral therapeutics
- **BBAI** - BigBear.ai Holdings - AI-powered analytics
- **BCEL** - Atreca Inc. - Antibody therapeutics
- **BDSX** - Biodesix Inc. - Molecular diagnostics
- **CELC** - Celcuity Inc. - Cellular analysis company
- **CELU** - Celularity Inc. - Cellular therapeutics
- **CGTX** - Cognition Therapeutics - Alzheimer's treatment
- **CRMD** - CorMedix Inc. - Pharmaceutical products
- **DMAC** - DiaMedica Therapeutics - Neurological disorders
- **DRMA** - Dermata Therapeutics - Dermatology treatments
- **ELDN** - Eledon Pharmaceuticals - Anti-rejection therapeutics
- **EVAX** - Evaxion Biotech - AI-immunology platform
- **GTHX** - G1 Therapeutics - Cancer treatments
- **HOWL** - Werewolf Therapeutics - Immunotherapy

### 💻 Technology & Software (16 stocks)
- **INSG** - Inseego Corp. - 5G and IoT device solutions
- **KTRA** - Kintara Therapeutics - Cancer and COVID-19 treatments
- **LTRN** - Lantern Pharma - AI drug discovery
- **MMAT** - Meta Materials Inc. - Metamaterial technologies
- **NMTC** - NeuroOne Medical Technologies - Electrode technology
- **ONCT** - Oncternal Therapeutics - Oncology company
- **OPTT** - Ocean Power Technologies - Wave energy systems
- **PGNY** - Progyny Inc. - Fertility benefits management
- **PRPL** - Purple Innovation - Sleep technology
- **PTGX** - Protagonist Therapeutics - Peptide therapeutics
- **QNST** - QuinStreet Inc. - Digital marketing services
- **RBOT** - Vicarious Surgical - Robotic surgery
- **RUBY** - Rubicon Technologies - Waste management technology
- **SEER** - Seer Inc. - Proteomics technology platform
- **SGTX** - Sigilon Therapeutics - Cell therapy platform
- **SOUN** - SoundHound AI - Voice AI platform

### 💰 Financial & Services (16 stocks)
- **LOVE** - Lovesac Company - Furniture technology
- **MGIC** - Magic Software Enterprises - Business integration
- **LOAN** - Manhattan Bridge Capital - Bridge lending
- **TREE** - LendingTree Inc. - Online lending marketplace
- **CLOV** - Clover Health - Medicare Advantage plans
- **SOFI** - SoFi Technologies - Digital financial services
- **UPST** - Upstart Holdings - AI lending platform
- **BLZE** - Backblaze Inc. - Cloud storage platform
- **BMEA** - Biomea Fusion - Oncology therapeutics
- **BTBT** - Bit Digital Inc. - Digital asset mining
- **CNET** - CynergisTek Inc. - Cybersecurity services
- **EAST** - Eastside Distilling - Craft spirits
- **FNKO** - Funko Inc. - Pop culture collectibles
- **GEVO** - Gevo Inc. - Renewable fuels and chemicals
- **HCDI** - Harbor Custom Development - Real estate development
- **INMB** - INmune Bio Inc. - Immunotherapy company

### 🛒 Consumer & Retail (2 stocks)
- **KOSS** - Koss Corporation - Audio equipment manufacturer
- **MARK** - Remark Holdings - AI and data analytics

## Strategy Alignment

These stocks are specifically chosen because they:

1. **Match Target Market Cap**: All under $350M market cap (nano-cap classification)
2. **High Volatility**: Provide opportunities for momentum and mean-reversion strategies
3. **Low Liquidity**: Suitable for testing position sizing and volume constraints
4. **Sector Diversification**: Spread across healthcare, technology, financial services, and consumer sectors
5. **Research Relevance**: Align with academic research on small-cap market inefficiencies

## Usage in System

- **Single Strategy Benchmark**: Strategy-weighted portfolio vs Equal-weighted portfolio (2% each stock)
  - This answers: "Does the strategy add value over simply holding all stocks equally?"
  - Alternative: Strategy vs market indices (Russell 2000, S&P 500, etc.)
- **A/B Testing**: Multiple strategies compete on the same stock universe
- **Risk Management**: Position sizing and concentration limits tested on realistic targets
- **Performance Attribution**: Sector-based analysis across the 4 main categories

## Benchmarking Methodology

### Equal-Weighted Baseline (Default)
- **Benchmark Portfolio**: 2% allocation to each of the 50 nano-cap stocks
- **Strategy Portfolio**: Same 50 stocks with weights determined by strategy signals
- **Key Metrics**: 
  - **Alpha**: Strategy return - risk-free rate - beta × (benchmark return - risk-free rate)
  - **Information Ratio**: Alpha / tracking error (measures skill vs risk)
  - **Excess Return**: Strategy total return - benchmark total return
  - **Win Rate**: Percentage of days strategy outperformed benchmark

### Why This Approach?
1. **Fair Comparison**: Both portfolios trade the same stocks, only weights differ
2. **Skill Measurement**: Shows if strategy adds value through stock selection/timing
3. **Risk-Adjusted**: Accounts for differences in portfolio volatility and correlation
4. **Institutional Standard**: This is how professional fund managers are evaluated

## Notes

- This universe is used for **demo and testing purposes**
- For production trading, the universe would be:
  - Dynamically updated based on current market cap
  - Filtered for liquidity requirements
  - Screened for fundamental criteria
  - Adjusted based on strategy-specific requirements

---

*Last Updated: January 2025*
*Market cap data subject to change with market conditions*