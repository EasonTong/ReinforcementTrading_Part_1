import pandas as pd
import numpy as np

df = pd.read_csv('trade_history_output.csv')
print('Total trades:', len(df))
print('Avg net_pips:', df['net_pips'].mean())
print('Sum net_pips:', df['net_pips'].sum())
print('Win rate:', (df['net_pips'] > 0).mean())
print('Max equity:', df['equity_usd'].max())
print('Min equity:', df['equity_usd'].min())
print('Final equity:', df['equity_usd'].iloc[-1])
print()

print('By reason:')
print(df['reason'].value_counts())
print()

print('Avg time_in_trade:', df['time_in_trade'].mean())

wins = df[df['net_pips'] > 0]
losses = df[df['net_pips'] < 0]
if len(wins):
    print(f'Wins: avg realized={wins["realized_pips"].mean():.1f}')
if len(losses):
    print(f'Losses: avg realized={losses["realized_pips"].mean():.1f}')
if len(wins) and len(losses):
    print(f'Profit factor: {wins["net_pips"].sum() / abs(losses["net_pips"].sum()):.2f}')

print()
print('All trades:')
for i, row in df.iterrows():
    print(f"  {i+1:2d}. {row['reason']:30s} net={row['net_pips']:+.1f} pips equity=${row['equity_usd']:,.0f} time={row['time_in_trade']:3d}bars")
