import dash
from dash import html, dcc
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import asyncio
from ib_async import IB
import threading
import signal
import atexit
import uuid

shutdown_event = asyncio.Event()

# Global state
data = {
    'account': {},
    'portfolio': pd.DataFrame(columns=['Symbol', 'SecType', 'Qty', 'AvgCost', 'MarketValue', 'UnrealizedPnL']),
    'cash': pd.DataFrame(columns=['Currency', 'Amount'])
}
ib = IB()
loop = asyncio.new_event_loop()

# Metric explanations
METRIC_INFO = {
    "Account ID": "你的 IB 账户唯一编号。",
    "Net Liquidation Value": "若全部平仓结汇后账户可拿到的净值。",
    "Gross Position Value": "所有多空头寸绝对市值之和。",
    "Total Cash Value": "折算成基准货币后的现金余额。",
    "Equity With Loan Value": "自有资金 = 账户权益 - 已借保证金。",
    "Excess Liquidity": "Equity With Loan – Maintenance Margin，低于 0 触发强平。",
    "Available Funds": "在维持保证金基础上还能提取/交易的资金。",
    "Buying Power": "可开新仓的最大金额（含杠杆）。",
    "Maintenance Margin": "维持当前仓位必须留存的保证金。",
    "Initial Margin Requirement": "开新仓时需缴的保证金。",
    "Look Ahead Excess Liquidity": "下周期预测的 Excess Liquidity。",
    "Look Ahead Margin Req": "下周期预测的维持保证金需求。",
    "Highest Severity": "风险报警等级：0=正常，1=警告，2=强平在即。",
    "Day Trades Remaining": "当日剩余可做的日内交易次数。",
    "Unrealized PnL": "未平仓头寸浮动盈亏。",
    "Realized PnL": "当日已平仓头寸已实现盈亏。"
}

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

    while not shutdown_event.is_set():
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
            'Highest Severity': get('HighestSeverity')
        }

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

        positions = await ib.reqPositionsAsync()
        cash_positions = [
            {'Currency': pos.contract.currency, 'Amount': pos.position}
            for pos in positions if pos.contract.secType == 'CASH'
        ]
        data['cash'] = pd.DataFrame(cash_positions)

        await asyncio.sleep(5)
        
    if ib.isConnected():
        ib.disconnect()
        print("✅ Disconnected from IB")
            
def graceful_shutdown(*args):
    print("🔌 Graceful shutdown triggered.")
    loop.call_soon_threadsafe(shutdown_event.set)

signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)
atexit.register(graceful_shutdown)

start_ib_thread()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "IB Live Dashboard"

def metric_card(name, value):
    color = "light"
    if name == "Highest Severity":
        color = {"0": "success", "1": "warning", "2": "danger"}.get(str(value), "secondary")
    elif name in ("Unrealized PnL", "Realized PnL"):
        try:
            color = "success" if float(value) >= 0 else "danger"
        except:
            color = "secondary"
    card_id = str(uuid.uuid4())
    return html.Div([
        dbc.Card([
            dbc.CardHeader(name, className="text-center", style={"fontSize": "0.8rem", "padding": "0.3rem"}),
            dbc.CardBody(html.H6(f"{value}", className="card-title text-center mb-0", style={"fontSize": "1rem"})),
        ], color=color, inverse=False, outline=True, id=card_id,
           style={"minWidth": "140px", "padding": "0.2rem"}),
        dbc.Tooltip(METRIC_INFO[name], target=card_id, placement="bottom", style={"fontSize": "0.7rem"})
    ])

app.layout = html.Div([
    html.H1("📊 Interactive Brokers Live Dashboard", style={"textAlign": "center", "marginBottom": "40px"}),
    html.H2("🔒 Account Summary", style={"textAlign": "center"}),
    html.Div(id='status-table', style={"width": "95%", "margin": "auto", "marginBottom": "40px"}),
    html.H2("📦 Portfolio Holdings", style={"textAlign": "center"}),
    html.Div(id='portfolio-table', style={"width": "95%", "margin": "auto", "marginBottom": "30px"}),
    html.H2("💱 Currency Holdings", style={"textAlign": "center", "marginTop": "30px"}),
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
    acct = data['account']
    groups = {
        "Account Core Metrics": [
            'Net Liquidation Value', 
            'Equity With Loan Value',
            'Maintenance Margin',
            'Excess Liquidity'
        ],
        "Funds Availability": [
            'Total Cash Value',
            'Available Funds',
            'Buying Power'
        ],
        "PnL Overview": [
            'Unrealized PnL',
            'Realized PnL',
            'Gross Position Value'
        ],
        "Account Info & Risk": [
            'Account ID',
            'Highest Severity',
            'Day Trades Remaining'
        ]
    }

    status_children = []
    for title, keys in groups.items():
        cards = [metric_card(k, acct.get(k, "N/A")) for k in keys]
        status_children.append(html.H4(title, className="mt-3 mb-2"))
        status_children.append(
            html.Div(cards, style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fill, minmax(140px, 1fr))",
                "gap": "0.5rem"
            })
        )
    status_table = html.Div(status_children)

    df = data['portfolio']
    portfolio_table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in df.columns])),
        html.Tbody([html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(len(df))])
    ], style={"border": "1px solid #ccc", "width": "100%"})

    cash_df = data['cash']
    cash_table = html.Table([
        html.Thead(html.Tr([html.Th("Currency"), html.Th("Amount")])),
        html.Tbody([html.Tr([html.Td(row['Currency']), html.Td(row['Amount'])]) for _, row in cash_df.iterrows()])
    ], style={"border": "1px solid #ccc", "width": "100%"})

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