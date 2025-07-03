import streamlit as st
import pandas as pd
import yfinance as yf
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# --- Tool Functions ---

def load_trades(file):
    df = pd.read_csv(file)
    return df

def query_trades(df, query: str) -> str:
    """
    Query trade data using pandas query syntax or keywords.
    Example: "Show all sell trades for TCS"
    """
    if "sell" in query.lower():
        result = df[df['Transaction Type'].str.lower() == 'sell']
    elif "buy" in query.lower():
        result = df[df['Transaction Type'].str.lower() == 'buy']
    else:
        result = df
    # Add more parsing as needed
    return result.to_string(index=False)

def total_invested(df) -> str:
    buys = df[df['Transaction Type'].str.lower() == 'buy']
    total = buys['Order Value'].sum()
    return f"Total invested amount: ₹{total:,.2f}"

def total_brokerage(df) -> str:
    total = df['Brokerage'].sum()
    return f"Total brokerage paid: ₹{total:,.2f}"

def profit_loss_per_stock(df) -> str:
    summary = []
    for symbol in df['Symbol'].unique():
        trades = df[df['Symbol'] == symbol]
        buy = trades[trades['Transaction Type'].str.lower() == 'buy']['Order Value'].sum()
        sell = trades[trades['Transaction Type'].str.lower() == 'sell']['Order Value'].sum()
        pnl = sell - buy
        summary.append(f"{symbol}: ₹{pnl:,.2f}")
    return "\n".join(summary)

def get_current_price(symbol: str) -> str:
    ticker = yf.Ticker(symbol + ".NS")
    price = ticker.history(period="1d")['Close'].iloc[-1]
    return f"Current price of {symbol}: ₹{price:.2f}"

def missed_profit(df, symbol: str) -> str:
    sells = df[(df['Symbol'] == symbol) & (df['Transaction Type'].str.lower() == 'sell')]
    if sells.empty:
        return f"No sell trades found for {symbol}."
    last_sell_price = sells.iloc[-1]['Average Price']
    qty = sells.iloc[-1]['Quantity']
    current_price = yf.Ticker(symbol + ".NS").history(period="1d")['Close'].iloc[-1]
    missed = (current_price - last_sell_price) * qty
    return f"Missed profit/loss for {symbol}: ₹{missed:,.2f} (Sold at ₹{last_sell_price:.2f}, now ₹{current_price:.2f})"

# --- Streamlit UI ---

st.title("Chat with Your Equity Trade History (Zerodha Format)")

uploaded_file = st.file_uploader("Upload your tradebook CSV", type="csv")

if uploaded_file:
    df = load_trades(uploaded_file)
    st.write("Sample of your trades:", df.head())

    # --- LangChain Agent Setup ---
    tools = [
        Tool(
            name="QueryTrades",
            func=lambda q: query_trades(df, q),
            description="Query trade data. Input: natural language query about trades."
        ),
        Tool(
            name="TotalInvested",
            func=lambda _: total_invested(df),
            description="Calculate total invested amount. Input: any string."
        ),
        Tool(
            name="TotalBrokerage",
            func=lambda _: total_brokerage(df),
            description="Calculate total brokerage paid. Input: any string."
        ),
        Tool(
            name="ProfitLossPerStock",
            func=lambda _: profit_loss_per_stock(df),
            description="Show profit/loss per stock. Input: any string."
        ),
        Tool(
            name="CurrentPrice",
            func=lambda symbol: get_current_price(symbol),
            description="Get current market price of a stock. Input: stock symbol (e.g., TCS)."
        ),
        Tool(
            name="MissedProfit",
            func=lambda symbol: missed_profit(df, symbol),
            description="Compute missed profit/loss after selling a stock. Input: stock symbol."
        ),
    ]

    llm = OpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(
        tools, llm, agent="chat-zero-shot-react-description", memory=memory, verbose=True
    )

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("Ask about your trades:")
    if user_input:
        st.session_state.messages.append(("user", user_input))
        response = agent.run(user_input)
        st.session_state.messages.append(("agent", response))

    for role, msg in st.session_state.messages:
        st.markdown(f"**{role.title()}:** {msg}")

else:
    st.info("Please upload your Zerodha tradebook CSV file to start chatting.")
