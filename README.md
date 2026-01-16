# 📊 U.S. Policy-Driven Investment Scanner

A Streamlit dashboard that identifies investment opportunities based on U.S. fiscal and monetary policy shifts, with global sentiment analysis and technical entry/exit zones.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

- **Policy Analysis**: Simulated extraction of White House fiscal policy and Federal Reserve monetary policy announcements
- **Sector Mapping**: Automatic identification of impacted sectors (Green Energy, Semiconductors, Financials, etc.)
- **ETF Scanning**: Uses yfinance to scan sector ETFs for undervalued opportunities based on:
  - RSI (Relative Strength Index) - identifies oversold conditions
  - Price vs 200-day Moving Average - identifies discounts
  - P/E Ratio analysis
- **Global Sentiment**: Risk assessment for competing economies (EU, China, Japan)
- **Technical Analysis**: 
  - Interactive Plotly charts with candlestick patterns
  - Support and resistance levels
  - Best Buy and Target Sell price calculations
  - Entry/Exit zone visualization

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/policy-investment-scanner.git
   cd policy-investment-scanner
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud

### Step-by-Step Deployment

1. **Push to GitHub**
   - Create a new repository on GitHub
   - Push this code to your repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git push -u origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"

3. **Configure Deployment**
   - **Repository**: Select your GitHub repository
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - Click "Deploy!"

4. **Wait for Deployment**
   - Streamlit Cloud will automatically install dependencies from `requirements.txt`
   - Your app will be live at `https://your-app-name.streamlit.app`

### Environment Variables (Optional)

If you add real API integrations, you can set secrets in Streamlit Cloud:
1. Go to your app's settings
2. Click "Secrets"
3. Add your API keys in TOML format:
   ```toml
   [api_keys]
   news_api = "your-api-key"
   ```

## 📁 Project Structure

```
policy-investment-scanner/
├── streamlit_app.py      # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore file (optional)
```

## 🔧 Configuration

### Sector ETF Mapping

The default sector-to-ETF mapping can be modified in the `SECTOR_ETF_MAPPING` dictionary:

```python
SECTOR_ETF_MAPPING = {
    "Green Energy": ["ICLN", "TAN", "QCLN", "PBW", "FAN"],
    "Semiconductors": ["SMH", "SOXX", "PSI", "XSD", "SOXQ"],
    # Add more sectors...
}
```

### Analysis Parameters

Adjust these in the sidebar:
- **Top picks per sector**: 1-5 ETFs
- **RSI Oversold Threshold**: 20-50 (default: 30)
- **Global Sentiment Filter**: Include/exclude high-risk assets

## 📊 Dashboard Tabs

1. **Policy Impact**: View latest fiscal and monetary policy summaries
2. **Global Sentiment**: Analyze trade war risks and regulatory resistance by region
3. **Opportunities**: Scan selected sectors for undervalued ETFs
4. **Technical Analysis**: Deep-dive into individual assets with interactive charts

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It does not constitute financial advice. The policy data is simulated for demonstration purposes. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.

## 🛠️ Future Enhancements

- [ ] Real-time policy data extraction using web scraping
- [ ] Integration with news APIs for sentiment analysis
- [ ] Machine learning-based sector prediction
- [ ] Portfolio optimization module
- [ ] Alert system for entry/exit signals
- [ ] Historical backtesting capabilities

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ using Streamlit, Plotly, and yfinance**
