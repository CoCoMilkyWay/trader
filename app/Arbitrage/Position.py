import asyncio
from ib_async import IB

async def main():
    ib = IB()
    await ib.connectAsync('127.0.0.1', 7497, clientId=2)

    account_summary = await ib.accountSummaryAsync()
    account_id = account_summary[0].account if account_summary else 'DU123456'

    def get_tag(tag):
        return next((item.value for item in account_summary if item.tag == tag), 'N/A')

    print(f"\n🔒 Account ID: {account_id}")

    print("\n📊 Account Financial Overview:")
    print(f"Net Liquidation Value     : {get_tag('NetLiquidation')}")
    print(f"Total Cash Value          : {get_tag('TotalCashValue')}")
    print(f"Available Funds           : {get_tag('AvailableFunds')}")
    print(f"Buying Power              : {get_tag('BuyingPower')}")
    print(f"Excess Liquidity          : {get_tag('ExcessLiquidity')}")
    print(f"Equity With Loan Value    : {get_tag('EquityWithLoanValue')}")
    print(f"Maintenance Margin        : {get_tag('MaintMarginReq')}")
    print(f"Initial Margin Requirement: {get_tag('InitMarginReq')}")
    print(f"Look Ahead Excess Liquidity : {get_tag('LookAheadExcessLiquidity')}")
    print(f"Look Ahead Margin Req       : {get_tag('LookAheadMaintMarginReq')}")
    print(f"Day Trades Remaining         : {get_tag('DayTradesRemaining')}")
    print(f"Unrealized PnL               : {get_tag('UnrealizedPnL')}")
    print(f"Realized PnL                 : {get_tag('RealizedPnL')}")
    print(f"Gross Position Value         : {get_tag('GrossPositionValue')}")
    print(f"Highest Severity             : {get_tag('HighestSeverity')}")  # 1 = warning, 2 = liquidation likely

    print("\n📦 Portfolio Holdings:")
    for pos in ib.portfolio():
        print(f"{pos.contract.symbol} ({pos.contract.secType}) | Qty: {pos.position} | "
              f"Avg Cost: {pos.averageCost:.2f} | Market Value: {pos.marketValue:.2f} | "
              f"Unrealized PnL: {pos.unrealizedPNL:.2f}")

    print("\n💱 Currency Holdings:")
    # while True:
    positions = await ib.reqPositionsAsync()
    cash_positions = [p for p in positions if p.contract.secType == 'CASH']
    for pos in cash_positions:
        print(f"{pos.contract.currency} : {pos.position}")

    print(ib.portfolio())
    print(positions)

    ib.disconnect()

asyncio.run(main())
