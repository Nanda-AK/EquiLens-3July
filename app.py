import pandas as pd
import yfinance as yf
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

# Load CSV data
df = pd.read_csv("tradebook-QK2539-EQ.csv")
df.columns = df.columns.str.strip().str.lower()
print(df.columns.tolist())

# Utility functions
def get_total_sold_quantity(symbol: str) -> str:
    symbol = symbol.upper()
    sell_df = df[df['trade_type'].str.lower() == 'sell']
    sell_df = sell_df[df['symbol'].str.upper() == symbol]

    if sell_df.empty:
        return f"No sell records found for {symbol}."

    total_qty = sell_df['quantity'].sum()
    total_value = (sell_df['quantity'] * sell_df['price']).sum()
    avg_price = total_value / total_qty if total_qty != 0 else 0

    return f"You have sold {int(total_qty)} shares of {symbol} at an average price of ₹{avg_price:.2f}."

def get_total_bought_quantity(symbol: str) -> str:
    symbol = symbol.upper()
    buy_df = df[df['trade_type'].str.lower() == 'buy']
    buy_df = buy_df[buy_df['symbol'].str.upper() == symbol]

    if buy_df.empty:
        return f"No buy records found for {symbol}."

    total_qty = buy_df['quantity'].sum()
    total_value = (buy_df['quantity'] * buy_df['price']).sum()
    avg_price = total_value / total_qty if total_qty != 0 else 0

    return f"You have bought {int(total_qty)} shares of {symbol} at an average price of ₹{avg_price:.2f}."

def calculate_average_cost(_: str = "") -> str:
    buy_df = df[df["trade_type"].str.lower() == "buy"]

    if buy_df.empty:
        return "No buy trades available to calculate average cost."

    grouped = (
        buy_df.groupby("symbol")
        .apply(lambda x: pd.Series({
            "total_quantity": x["quantity"].sum(),
            "average_price": round((x["price"] * x["quantity"]).sum() / x["quantity"].sum(), 2)
        }))
        .reset_index()
    )

    return grouped.to_string(index=False)

def query_trade_data(query: str) -> str:
    action = None
    if "buy" in query.lower():
        action = "buy"
    elif "sell" in query.lower():
        action = "sell"

    if action:
        result_df = df[df["trade_type"].str.lower() == action]
        if "tcs" in query.lower():
            result_df = result_df[df["symbol"].str.upper() == "TCS"]

        if result_df.empty:
            return f"No {action} trades found."

        # LIMIT ROWS
        display_df = result_df[["trade_date", "symbol", "quantity", "price"]].copy()
        display_df = display_df.head(20)  # show only first 20 rows
        return display_df.to_string(index=False)

    return "Query not recognized. Try asking about buy or sell history."



def get_current_price(symbol: str) -> str:
    try:
        ticker = yf.Ticker(symbol + ".NS")
        price = ticker.info['regularMarketPrice']
        return f"Current price of {symbol} is ₹{price}"
    except Exception as e:
        return f"Error fetching price for {symbol}: {str(e)}"

def missed_profit_loss(symbol: str) -> str:
    try:
        sell_df = df[(df['Symbol'].str.upper() == symbol.upper()) & (df['trade_type'].str.lower() == 'sell')]
        if sell_df.empty:
            return f"No sell transactions found for {symbol}."

        last_sell = sell_df.sort_values(by='Date').iloc[-1]
        qty = last_sell['Quantity']
        sell_price = last_sell['Average Price']

        ticker = yf.Ticker(symbol + ".NS")
        current_price = ticker.info['regularMarketPrice']

        missed_pnl = (current_price - sell_price) * qty
        return f"Missed P/L on last sale of {symbol}: ₹{missed_pnl:.2f}"
    except Exception as e:
        return f"Error calculating missed P/L for {symbol}: {str(e)}"

# Define tools for the agent
tools = [
    Tool(
    	name="TotalSoldQuantity",
    	func=get_total_sold_quantity,
    	description="Returns total quantity and average price for stocks sold. Use this when asked how many stocks were sold for a specific symbol."
    ),
    Tool(
        name="QueryTradeData",
        func=query_trade_data,
        description="Use this to get buy or sell history from trade data"
    ),
    Tool(
        name="GetCurrentPrice",
        func=get_current_price,
        description="Fetches the current market price for a given NSE stock symbol"
    ),
    Tool(
        name="MissedProfitLoss",
        func=missed_profit_loss,
        description="Calculate the missed profit or loss if the stock was held instead of sold"
    ),
    Tool(
    	name="CalculateAverageCost",
    	func=calculate_average_cost,
    	description="Calculate total quantity and average buy price per stock symbol from all buy trades."
    ),
    Tool(
    	name="TotalBoughtQuantity",
    	func=get_total_bought_quantity,
    	description="Returns total quantity and average price for stocks bought. Use this when asked how many stocks were bought for a specific symbol."
    )
]

# Initialize LLM and agent
llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key="Your-OpenAI-API-Key")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    #verbose=False,
    handle_parsing_errors=True
)

# Command-line chat loop
print("\n💬 Ask me anything about your equity trades (type 'exit' to quit):")
while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() == "exit":
        break
    response = agent.run(user_input)
    print("\n🤖", response)
