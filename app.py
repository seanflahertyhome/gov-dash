import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

SECTOR_ETF_MAPPING = {
    "Green Energy": ["ICLN", "TAN", "QCLN", "PBW", "FAN"],
    "Semiconductors": ["SMH", "SOXX", "PSI", "XSD", "SOXQ"],
    "Financials": ["XLF", "VFH", "IYF", "KBE", "KRE"],
    "Infrastructure": ["PAVE", "IFRA", "IGF", "NFRA", "PKB"],
    "Healthcare": ["XLV", "VHT", "IBB", "XBI", "ARKG"],
    "Defense": ["ITA", "PPA", "XAR", "DFEN", "SHLD"],
    "Technology": ["XLK", "VGT", "QQQ", "IGV", "FTEC"],
    "Real Estate": ["VNQ", "IYR", "XLRE", "RWR", "SCHH"],
}

COMPETING_ECONOMIES = {
    "EU": {"sentiment": 0.65, "trade_risk": "Medium", "regulatory_alignment": 0.7},
    "China": {"sentiment": 0.35, "trade_risk": "High", "regulatory_alignment": 0.3},
    "Japan": {"sentiment": 0.75, "trade_risk": "Low", "regulatory_alignment": 0.8},
    "UK": {"sentiment": 0.70, "trade_risk": "Low", "regulatory_alignment": 0.75},
}

# ============================================================
# DATA SIMULATION FUNCTIONS
# ============================================================

def get_simulated_white_house_policy():
    """Simulate White House fiscal policy press releases."""
    policies = [
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "title": "Infrastructure Investment and Jobs Act Update",
            "summary": "Administration announces $50B allocation for clean energy infrastructure, semiconductor manufacturing facilities, and broadband expansion.",
            "key_areas": ["Infrastructure", "Green Energy", "Semiconductors"],
            "fiscal_impact": "Expansionary",
            "spending_increase": "$50B over 5 years"
        },
        {
            "date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            "title": "Trade Policy Review with Strategic Partners",
            "summary": "New tariff exemptions announced for allied nations. Focus on securing semiconductor supply chains and rare earth materials.",
            "key_areas": ["Semiconductors", "Technology", "Defense"],
            "fiscal_impact": "Neutral",
            "trade_implications": "Positive for allies, restrictive for competitors"
        },
        {
            "date": (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"),
            "title": "Climate Action Economic Initiative",
            "summary": "Tax credits extended for renewable energy investments. New incentives for domestic solar and wind manufacturing.",
            "key_areas": ["Green Energy", "Infrastructure"],
            "fiscal_impact": "Expansionary",
            "spending_increase": "$30B in tax credits"
        }
    ]
    return policies

def get_simulated_fed_policy():
    """Simulate Federal Reserve monetary policy announcements."""
    policies = [
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "title": "FOMC Meeting Statement",
            "summary": "Federal Reserve maintains current interest rate target at 5.25-5.50%. Signals potential rate cuts in coming quarters if inflation continues to moderate.",
            "interest_rate": "5.25-5.50%",
            "rate_direction": "Hold with dovish bias",
            "inflation_outlook": "Moderating toward 2% target",
            "employment_outlook": "Strong labor market with slight cooling",
            "impacted_sectors": ["Financials", "Real Estate", "Technology"]
        },
        {
            "date": (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d"),
            "title": "Monetary Policy Report",
            "summary": "Fed signals patience on rate decisions. Balance sheet reduction continues at measured pace.",
            "interest_rate": "5.25-5.50%",
            "rate_direction": "Hold",
            "quantitative_policy": "Gradual tightening",
            "impacted_sectors": ["Financials", "Real Estate"]
        }
    ]
    return policies

def get_global_sentiment_data():
    """Simulate global sentiment and trade resistance data."""
    return {
        "EU": {
            "sentiment_score": 0.65,
            "trade_war_risk": "Low",
            "regulatory_resistance": "Medium",
            "key_concerns": ["Digital Services Tax", "GDPR Compliance"],
            "favorable_sectors": ["Green Energy", "Healthcare", "Technology"]
        },
        "China": {
            "sentiment_score": 0.35,
            "trade_war_risk": "High",
            "regulatory_resistance": "High",
            "key_concerns": ["Semiconductor Export Controls", "Investment Restrictions"],
            "favorable_sectors": ["Defense", "Healthcare"]
        },
        "Japan": {
            "sentiment_score": 0.75,
            "trade_war_risk": "Low",
            "regulatory_resistance": "Low",
            "key_concerns": ["Currency Policy"],
            "favorable_sectors": ["Semiconductors", "Technology", "Defense"]
        }
    }

# ============================================================
# FINANCIAL ANALYSIS FUNCTIONS
# ============================================================

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="2y"):
    """Fetch stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {e}")
        return None, None

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_moving_averages(prices):
    """Calculate various moving averages."""
    return {
        "MA_20": prices.rolling(window=20).mean(),
        "MA_50": prices.rolling(window=50).mean(),
        "MA_200": prices.rolling(window=200).mean()
    }

def calculate_support_resistance(prices, window=20):
    """Calculate support and resistance levels."""
    rolling_min = prices.rolling(window=window).min()
    rolling_max = prices.rolling(window=window).max()
    
    support = rolling_min.iloc[-1] if len(rolling_min) > 0 else prices.iloc[-1] * 0.95
    resistance = rolling_max.iloc[-1] if len(rolling_max) > 0 else prices.iloc[-1] * 1.05
    
    return support, resistance

def analyze_etf(ticker):
    """Perform comprehensive analysis on an ETF."""
    hist, info = fetch_stock_data(ticker)
    
    if hist is None or hist.empty:
        return None
    
    current_price = hist['Close'].iloc[-1]
    
    # Calculate technical indicators
    rsi = calculate_rsi(hist['Close'])
    current_rsi = rsi.iloc[-1] if not rsi.empty else 50
    
    mas = calculate_moving_averages(hist['Close'])
    ma_200 = mas['MA_200'].iloc[-1] if not mas['MA_200'].empty else current_price
    
    support, resistance = calculate_support_resistance(hist['Close'])
    
    # Calculate discount metrics
    discount_to_ma200 = ((current_price - ma_200) / ma_200) * 100 if ma_200 > 0 else 0
    
    # Get P/E ratio if available
    pe_ratio = info.get('trailingPE', None) if info else None
    
    # Calculate 52-week metrics
    high_52w = hist['Close'].tail(252).max() if len(hist) >= 252 else hist['Close'].max()
    low_52w = hist['Close'].tail(252).min() if len(hist) >= 252 else hist['Close'].min()
    
    # Best Buy Price (based on 200-day MA with a discount factor)
    best_buy_price = ma_200 * 0.95
    
    # Target Sell Price (based on resistance with momentum factor)
    target_sell_price = resistance * 1.05
    
    return {
        "ticker": ticker,
        "name": info.get('shortName', ticker) if info else ticker,
        "current_price": current_price,
        "rsi": current_rsi,
        "ma_200": ma_200,
        "discount_to_ma200": discount_to_ma200,
        "pe_ratio": pe_ratio,
        "support": support,
        "resistance": resistance,
        "best_buy_price": best_buy_price,
        "target_sell_price": target_sell_price,
        "52w_high": high_52w,
        "52w_low": low_52w,
        "history": hist
    }

def scan_sector_opportunities(sector, num_picks=3):
    """Scan sector for best investment opportunities."""
    if sector not in SECTOR_ETF_MAPPING:
        return []
    
    etfs = SECTOR_ETF_MAPPING[sector]
    analyzed = []
    
    for ticker in etfs:
        analysis = analyze_etf(ticker)
        if analysis:
            analyzed.append(analysis)
    
    # Score based on RSI (lower is better for buying) and discount to MA200
    for item in analyzed:
        rsi_score = 100 - item['rsi'] if item['rsi'] else 50
        discount_score = -item['discount_to_ma200'] if item['discount_to_ma200'] else 0
        item['opportunity_score'] = (rsi_score * 0.4) + (discount_score * 0.6)
    
    # Sort by opportunity score and return top picks
    analyzed.sort(key=lambda x: x['opportunity_score'], reverse=True)
    return analyzed[:num_picks]

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_price_chart(analysis):
    """Create interactive price chart with entry/exit zones."""
    hist = analysis['history']
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{analysis['ticker']} - Price Action", "RSI Indicator")
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Moving Averages
    mas = calculate_moving_averages(hist['Close'])
    fig.add_trace(
        go.Scatter(x=hist.index, y=mas['MA_50'], name="MA 50", 
                   line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=hist.index, y=mas['MA_200'], name="MA 200", 
                   line=dict(color='purple', width=1)),
        row=1, col=1
    )
    
    # Entry Zone (Best Buy)
    fig.add_hline(y=analysis['best_buy_price'], line_dash="dash", 
                  line_color="green", annotation_text="Best Buy Zone",
                  row=1, col=1)
    
    # Exit Zone (Target Sell)
    fig.add_hline(y=analysis['target_sell_price'], line_dash="dash", 
                  line_color="red", annotation_text="Target Sell Zone",
                  row=1, col=1)
    
    # Support and Resistance
    fig.add_hline(y=analysis['support'], line_dash="dot", 
                  line_color="blue", annotation_text="Support",
                  row=1, col=1)
    fig.add_hline(y=analysis['resistance'], line_dash="dot", 
                  line_color="orange", annotation_text="Resistance",
                  row=1, col=1)
    
    # RSI
    rsi = calculate_rsi(hist['Close'])
    fig.add_trace(
        go.Scatter(x=hist.index, y=rsi, name="RSI", 
                   line=dict(color='blue', width=1)),
        row=2, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    
    return fig

def create_sector_comparison_chart(opportunities):
    """Create comparison chart for sector opportunities."""
    if not opportunities:
        return None
    
    tickers = [o['ticker'] for o in opportunities]
    scores = [o['opportunity_score'] for o in opportunities]
    rsi_values = [o['rsi'] for o in opportunities]
    discounts = [o['discount_to_ma200'] for o in opportunities]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Opportunity Score", "RSI (Lower = Oversold)", "Discount to 200-MA (%)")
    )
    
    colors = ['#00CC96', '#636EFA', '#EF553B', '#AB63FA', '#FFA15A']
    
    fig.add_trace(
        go.Bar(x=tickers, y=scores, marker_color=colors[:len(tickers)], name="Score"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=tickers, y=rsi_values, marker_color=colors[:len(tickers)], name="RSI"),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=tickers, y=discounts, marker_color=colors[:len(tickers)], name="Discount"),
        row=1, col=3
    )
    
    fig.update_layout(height=300, showlegend=False, template="plotly_dark")
    return fig

# ============================================================
# STREAMLIT UI COMPONENTS
# ============================================================

def render_sidebar():
    """Render the sidebar with controls."""
    st.sidebar.title("🎯 Investment Scanner")
    st.sidebar.markdown("---")
    
    # Sector Selection
    st.sidebar.subheader("📊 Sector Selection")
    selected_sectors = st.sidebar.multiselect(
        "Choose sectors to analyze:",
        list(SECTOR_ETF_MAPPING.keys()),
        default=["Green Energy", "Semiconductors"]
    )
    
    st.sidebar.markdown("---")
    
    # Analysis Parameters
    st.sidebar.subheader("⚙️ Analysis Parameters")
    num_picks = st.sidebar.slider("Top picks per sector:", 1, 5, 3)
    
    rsi_threshold = st.sidebar.slider(
        "RSI Oversold Threshold:", 
        min_value=20, max_value=50, value=30,
        help="ETFs with RSI below this are considered oversold"
    )
    
    st.sidebar.markdown("---")
    
    # Global Sentiment Filter
    st.sidebar.subheader("🌍 Global Sentiment Filter")
    include_high_risk = st.sidebar.checkbox(
        "Include high trade-war risk assets", 
        value=False
    )
    
    st.sidebar.markdown("---")
    
    # Refresh Button
    if st.sidebar.button("🔄 Refresh Analysis", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    return selected_sectors, num_picks, rsi_threshold, include_high_risk

def render_policy_cards():
    """Render policy impact summary cards."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏛️ White House Fiscal Policy")
        policies = get_simulated_white_house_policy()
        
        for policy in policies[:2]:
            with st.expander(f"📋 {policy['title']}", expanded=True):
                st.markdown(f"**Date:** {policy['date']}")
                st.markdown(f"**Summary:** {policy['summary']}")
                st.markdown(f"**Fiscal Impact:** `{policy['fiscal_impact']}`")
                st.markdown("**Impacted Sectors:**")
                for sector in policy['key_areas']:
                    st.markdown(f"  - {sector}")
    
    with col2:
        st.markdown("### 🏦 Federal Reserve Monetary Policy")
        fed_policies = get_simulated_fed_policy()
        
        for policy in fed_policies:
            with st.expander(f"📋 {policy['title']}", expanded=True):
                st.markdown(f"**Date:** {policy['date']}")
                st.markdown(f"**Summary:** {policy['summary']}")
                st.markdown(f"**Interest Rate:** `{policy['interest_rate']}`")
                st.markdown(f"**Rate Direction:** `{policy['rate_direction']}`")
                st.markdown("**Impacted Sectors:**")
                for sector in policy['impacted_sectors']:
                    st.markdown(f"  - {sector}")

def render_global_sentiment():
    """Render global sentiment analysis."""
    st.markdown("### 🌍 Global Sentiment & Trade Risk Analysis")
    
    sentiment_data = get_global_sentiment_data()
    
    cols = st.columns(len(sentiment_data))
    
    for i, (region, data) in enumerate(sentiment_data.items()):
        with cols[i]:
            # Color based on sentiment
            sentiment_color = "green" if data['sentiment_score'] > 0.6 else "orange" if data['sentiment_score'] > 0.4 else "red"
            
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background-color: #1E1E1E; border-left: 4px solid {sentiment_color};">
                <h4 style="margin: 0;">{region}</h4>
                <p style="margin: 5px 0;">Sentiment: <strong>{data['sentiment_score']:.0%}</strong></p>
                <p style="margin: 5px 0;">Trade War Risk: <strong>{data['trade_war_risk']}</strong></p>
                <p style="margin: 5px 0;">Regulatory Risk: <strong>{data['regulatory_resistance']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Favorable Sectors:**")
            for sector in data['favorable_sectors']:
                st.markdown(f"✅ {sector}")

def render_opportunities(selected_sectors, num_picks, rsi_threshold):
    """Render investment opportunities."""
    st.markdown("### 💰 Investment Opportunities")
    
    all_opportunities = []
    
    for sector in selected_sectors:
        st.markdown(f"#### 📈 {sector}")
        
        with st.spinner(f"Scanning {sector} ETFs..."):
            opportunities = scan_sector_opportunities(sector, num_picks)
        
        if not opportunities:
            st.warning(f"No opportunities found in {sector}")
            continue
        
        all_opportunities.extend(opportunities)
        
        # Comparison chart
        comparison_fig = create_sector_comparison_chart(opportunities)
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Detailed cards
        opp_cols = st.columns(len(opportunities))
        
        for i, opp in enumerate(opportunities):
            with opp_cols[i]:
                # Determine signal
                if opp['rsi'] < rsi_threshold:
                    signal = "🟢 BUY"
                    signal_color = "green"
                elif opp['rsi'] > 70:
                    signal = "🔴 SELL"
                    signal_color = "red"
                else:
                    signal = "🟡 HOLD"
                    signal_color = "orange"
                
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 10px; background-color: #262730; margin-bottom: 10px;">
                    <h4 style="margin: 0;">{opp['ticker']}</h4>
                    <p style="font-size: 0.8em; color: #888;">{opp['name']}</p>
                    <h3 style="color: {signal_color}; margin: 10px 0;">{signal}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Current Price", f"${opp['current_price']:.2f}")
                st.metric("RSI", f"{opp['rsi']:.1f}", delta=f"{'Oversold' if opp['rsi'] < 30 else 'Overbought' if opp['rsi'] > 70 else 'Neutral'}")
                st.metric("vs 200-MA", f"{opp['discount_to_ma200']:.1f}%")
                
                st.markdown(f"**Best Buy:** ${opp['best_buy_price']:.2f}")
                st.markdown(f"**Target Sell:** ${opp['target_sell_price']:.2f}")
        
        st.markdown("---")
    
    return all_opportunities

def render_detailed_charts(opportunities):
    """Render detailed charts for selected opportunities."""
    if not opportunities:
        return
    
    st.markdown("### 📊 Detailed Technical Analysis")
    
    selected_ticker = st.selectbox(
        "Select asset for detailed analysis:",
        [f"{o['ticker']} - {o['name']}" for o in opportunities]
    )
    
    if selected_ticker:
        ticker = selected_ticker.split(" - ")[0]
        opp = next((o for o in opportunities if o['ticker'] == ticker), None)
        
        if opp:
            # Key metrics row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Current Price", f"${opp['current_price']:.2f}")
            with col2:
                st.metric("52W High", f"${opp['52w_high']:.2f}")
            with col3:
                st.metric("52W Low", f"${opp['52w_low']:.2f}")
            with col4:
                st.metric("Best Buy", f"${opp['best_buy_price']:.2f}")
            with col5:
                st.metric("Target Sell", f"${opp['target_sell_price']:.2f}")
            
            # Price chart
            fig = create_price_chart(opp)
            st.plotly_chart(fig, use_container_width=True)
            
            # Entry/Exit Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Entry Strategy")
                upside = ((opp['target_sell_price'] - opp['current_price']) / opp['current_price']) * 100
                downside = ((opp['current_price'] - opp['best_buy_price']) / opp['current_price']) * 100
                
                st.markdown(f"""
                - **Potential Upside:** {upside:.1f}%
                - **Distance to Buy Zone:** {downside:.1f}%
                - **Support Level:** ${opp['support']:.2f}
                - **200-day MA:** ${opp['ma_200']:.2f}
                """)
            
            with col2:
                st.markdown("#### 🚪 Exit Strategy")
                st.markdown(f"""
                - **Target Sell Price:** ${opp['target_sell_price']:.2f}
                - **Resistance Level:** ${opp['resistance']:.2f}
                - **Risk/Reward Ratio:** {upside/max(downside, 1):.2f}
                - **RSI Status:** {'Oversold' if opp['rsi'] < 30 else 'Overbought' if opp['rsi'] > 70 else 'Neutral'}
                """)

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    st.set_page_config(
        page_title="Policy-Driven Investment Scanner",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .stMetric {
            background-color: #262730;
            padding: 10px;
            border-radius: 5px;
        }
        .main .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3, h4 {
            color: #FAFAFA;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("📊 U.S. Policy-Driven Investment Scanner")
    st.markdown("""
    *Identify investment opportunities based on fiscal and monetary policy shifts, 
    with global sentiment analysis and technical entry/exit zones.*
    """)
    st.markdown("---")
    
    # Sidebar
    selected_sectors, num_picks, rsi_threshold, include_high_risk = render_sidebar()
    
    # Main Content
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Policy Impact", 
        "🌍 Global Sentiment", 
        "💰 Opportunities",
        "📈 Technical Analysis"
    ])
    
    with tab1:
        render_policy_cards()
    
    with tab2:
        render_global_sentiment()
    
    with tab3:
        opportunities = render_opportunities(selected_sectors, num_picks, rsi_threshold)
    
    with tab4:
        # We need to get opportunities again for this tab
        all_opps = []
        for sector in selected_sectors:
            opps = scan_sector_opportunities(sector, num_picks)
            all_opps.extend(opps)
        render_detailed_charts(all_opps)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8em;">
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational and research purposes only. 
        It does not constitute financial advice. Always conduct your own research before making investment decisions.</p>
        <p>Data sources: Simulated policy data | Market data via yfinance</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
