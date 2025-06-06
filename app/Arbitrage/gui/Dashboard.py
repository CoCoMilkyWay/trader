import dash
from dash import html, dcc
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import pandas as pd
import asyncio
from ib_async import IB
import threading
import signal
import atexit

# Global state
data = {
    'account': {},
    'portfolio': pd.DataFrame(columns=['Symbol', 'SecType', 'Qty', 'AvgCost', 'MarketValue', 'UnrealizedPnL']),
    'cash': pd.DataFrame(columns=['Currency', 'Amount'])
}
ib = IB()
loop = asyncio.new_event_loop()

def start_ib_thread():
    threading.Thread(target=ib_worker, daemon=True).start()

def ib_worker():
    asyncio.set_event_loop(loop)
    loop.run_until_complete(connect_and_poll())

async def connect_and_poll():
    try:
        await ib.connectAsync('127.0.0.1', 7497, clientId=2)
        print("✅ Connected to IB Gateway/TWS")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return

    while True:
        account_summary = await ib.accountSummaryAsync()
        account_id = account_summary[0].account if account_summary else 'DU123456'

        def get(tag):
            return next((item.value for item in account_summary if item.tag == tag), 'N/A')

        data['account'] = {
            'Account ID': account_id,
            'Net Liquidation Value': get('NetLiquidation'),
            'Total Cash Value': get('TotalCashValue'),
            'Available Funds': get('AvailableFunds'),
            'Buying Power': get('BuyingPower'),
            'Excess Liquidity': get('ExcessLiquidity'),
            'Equity With Loan Value': get('EquityWithLoanValue'),
            'Maintenance Margin': get('MaintMarginReq'),
            'Initial Margin Requirement': get('InitMarginReq'),
            'Look Ahead Excess Liquidity': get('LookAheadExcessLiquidity'),
            'Look Ahead Margin Req': get('LookAheadMaintMarginReq'),
            'Day Trades Remaining': get('DayTradesRemaining'),
            'Unrealized PnL': get('UnrealizedPnL'),
            'Realized PnL': get('RealizedPnL'),
            'Gross Position Value': get('GrossPositionValue'),
            'Highest Severity': get('HighestSeverity')  # 1 = warning, 2 = liquidation likely
        }

        # Portfolio holdings
        portfolio = []
        for pos in ib.portfolio():
            portfolio.append({
                'Symbol': pos.contract.symbol,
                'SecType': pos.contract.secType,
                'Qty': pos.position,
                'AvgCost': pos.averageCost,
                'MarketValue': pos.marketValue,
                'UnrealizedPnL': pos.unrealizedPNL
            })
        data['portfolio'] = pd.DataFrame(portfolio)

        # Currency holdings
        positions = await ib.reqPositionsAsync()
        cash_positions = [
            {'Currency': pos.contract.currency, 'Amount': pos.position}
            for pos in positions if pos.contract.secType == 'CASH'
        ]
        data['cash'] = pd.DataFrame(cash_positions)

        await asyncio.sleep(5)

def graceful_shutdown(*args):
    print("🔌 Graceful shutdown triggered.")
    if ib.isConnected():
        ib.disconnect()
        print("✅ Disconnected from IB")
    try:
        loop.stop()
    except:
        pass

# Handle shutdown
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)
atexit.register(graceful_shutdown)

# Start IB polling
start_ib_thread()

# Dash App
app = dash.Dash(__name__)
app.title = "IB Live Dashboard"

app.layout = html.Div([
    html.H1("📊 Interactive Brokers Live Dashboard", style={"textAlign": "center", "marginBottom": "40px"}),

    html.H2("🔒 Account Summary", style={"textAlign": "center"}),
    html.Div(id='status-table', style={"width": "80%", "margin": "auto", "marginBottom": "40px"}),

    html.H2("📦 Portfolio Holdings", style={"textAlign": "center"}),
    html.Div(id='portfolio-table', style={"width": "90%", "margin": "auto", "marginTop": "20px"}),

    html.H2("💱 Currency Holdings", style={"textAlign": "center", "marginTop": "40px"}),
    html.Div(id='cash-table', style={"width": "40%", "margin": "auto"}),

    dcc.Graph(id='portfolio-chart'),

    dcc.Interval(id='interval', interval=5000, n_intervals=0)
])

@app.callback(
    Output('status-table', 'children'),
    Output('portfolio-table', 'children'),
    Output('cash-table', 'children'),
    Output('portfolio-chart', 'figure'),
    Input('interval', 'n_intervals')
)
def update_dashboard(n):
    # Account summary table
    account_rows = [html.Tr([html.Td(k), html.Td(v)]) for k, v in data['account'].items()]
    status_table = html.Table([
        html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
        html.Tbody(account_rows)
    ], style={"marginBottom": "30px", "border": "1px solid black", "width": "100%"})

    # Portfolio holdings table
    df = data['portfolio']
    portfolio_table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in df.columns])),
        html.Tbody([html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(len(df))])
    ], style={"marginBottom": "30px", "width": "100%", "border": "1px solid #ccc"})

    # Cash positions table
    cash_df = data['cash']
    cash_table = html.Table([
        html.Thead(html.Tr([html.Th("Currency"), html.Th("Amount")])),
        html.Tbody([html.Tr([html.Td(row['Currency']), html.Td(row['Amount'])]) for _, row in cash_df.iterrows()])
    ], style={"marginBottom": "30px", "width": "100%", "border": "1px solid #ccc"})

    # Portfolio bar chart
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Bar(
            x=df['Symbol'],
            y=df['MarketValue'],
            name='Market Value',
            marker_color='skyblue'
        ))
        fig.update_layout(
            title="📈 Portfolio Market Value",
            xaxis_title="Symbol",
            yaxis_title="USD",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
    return status_table, portfolio_table, cash_table, fig

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)