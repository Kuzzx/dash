import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import datetime
import numpy as np
from binancedata import *
from sqlalchemy import create_engine, inspect

REFRESH_TIME = 1    # minutes

engine = create_engine('sqlite:///equity.db')
inspector = inspect(engine)
dropdown_options = inspector.get_table_names()

load_figure_template('DARKLY')
app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])
server = app.server

dropdown = dcc.Dropdown(
    options=sorted(dropdown_options), 
    value='LSR',
    id='algo-dropdown',
    clearable=False,
),

def df_from_sql(engine, table_name, limit=None, start_date=None, end_date=None, interval=None):

    if limit == None and start_date == None and end_date == None:   # No restrictions
        sql_query = f"SELECT * FROM {table_name}"

    elif limit != None and start_date == None and end_date == None: # Limit on # results
        sql_query = f"SELECT * FROM (SELECT * FROM {table_name} ORDER BY Datestamp DESC LIMIT {limit}) ORDER BY Datestamp"

    elif start_date != None and end_date == None:                   # Limit on start date
        sql_query = f"SELECT * FROM {table_name} WHERE Datestamp > '{start_date}'"
    
    elif start_date != None and end_date != None:                   # Limit on start and end dates
        sql_query = f"SELECT * FROM {table_name} WHERE Datestamp > '{start_date}' AND Datestamp < '{end_date}'"

    df = pd.read_sql(sql_query, engine)
    df['Datestamp'] = pd.to_datetime(df['Datestamp'])
    df = df.set_index('Datestamp')

    if interval != None:
        df = df.resample(interval).last()

    return df

@app.callback(
    Output('dd-output-fig', 'figure'),
    Input('algo-dropdown', 'value')
)
def generate_fig(dropdown_value):
    df_data = pd.read_sql(f'SELECT * FROM {dropdown_value}', engine)
    df_data['dateStamp'] = pd.to_datetime(df_data['dateStamp'])
    df_data = df_data.set_index('dateStamp')
    df_data = df_data.resample('1H').mean()

    # create a line chart using plotly
    fig = px.line(df_data, x=df_data.index, y='markToMarket',)
    fig.update_yaxes(title='')
    fig.update_xaxes(title='')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'
    })

    return fig

@app.callback(
    Output('dd-output-table', 'children'),
    Input('algo-dropdown', 'value')
)
def generate_table(dropdown_value):

    df_data = pd.read_sql(f"SELECT * FROM {dropdown_value}", engine)

    start_mtm = float(df_data['markToMarket'].iloc[0])
    end_mtm = float(df_data['markToMarket'].iloc[-1])

    start_date = pd.to_datetime(df_data['dateStamp'].iloc[0])
    end_date = pd.to_datetime(df_data['dateStamp'].iloc[-1])

    if (end_date - start_date).days > 0:
        duration_years = (end_date - start_date).days / 365
    else:
        duration_years = 1 / 365

    raw_return = (end_mtm - start_mtm) / (start_mtm)
    cagr = (end_mtm / start_mtm) ** (1 / duration_years) - 1

    df_data['dateStamp'] = pd.to_datetime(df_data['dateStamp'])
    df_data = df_data.set_index('dateStamp')
    df_data = df_data.resample('1H').mean()
    df_data['returns'] = df_data['markToMarket'].pct_change()

    sharpe_multiple_factor = 365 * 24 # hourly
    sharpe = (df_data['returns'].mean() / df_data['returns'].std()) * (sharpe_multiple_factor ** 0.5)

    roll_max = df_data['markToMarket'].cummax()
    daily_drawdown = df_data['markToMarket'] / roll_max - 1.0
    max_daily_drawdown = daily_drawdown.cummin().iloc[-1]

    df = pd.DataFrame(
        [
            ['MTM', f'{end_mtm:,.2f}'],
            ['P&L', f'{end_mtm - start_mtm:,.2f}'],
            ['Return', f'{100 * raw_return:,.2f}%'],
            ['CAGR',  f'{100 * cagr:,.2f}%'],
            ['Sharpe', f'{sharpe:,.2f}'],
            ['Max DD', f'{100 * max_daily_drawdown:,.2f}%'],
            ['Last update', datetime.datetime.now().strftime('%Y-%m-%d %H:%M')]
        ],
        columns=['Metric', 'Value']
    )

    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, color="dark", size="sm")


def generate_market_table():

    db_path = "binance_data.db"
    market_engine = create_engine(f"sqlite:///{db_path}")

    btc_close = calc_rsi(df_from_sql(engine=market_engine, table_name='BTCPERP', limit=1000), period=14, column='Volume')
    eth_close = calc_rsi(df_from_sql(engine=market_engine, table_name='ETHPERP', limit=1000), period=14, column='Volume')

    btc_lsr = calc_rsi(df_from_sql(engine=market_engine, table_name='LSRBTC', limit=1000, interval='2h'), period=14, column='LSR')
    eth_lsr = calc_rsi(df_from_sql(engine=market_engine, table_name='LSRETH', limit=1000, interval='2h'), period=14, column='LSR')

    btc_funding = calc_rsi(df_from_sql(engine=market_engine, table_name='FUNDINGBTC', limit=2000, interval='2h'), period=14, column='Funding')
    eth_funding = calc_rsi(df_from_sql(engine=market_engine, table_name='FUNDINGETH', limit=2000, interval='2h'), period=14, column='Funding')

    # UPDATE!! -> need to check whether spot and perp data actually properly alligned -> Change to premium index
    btc_basis = calc_rsi(get_binance_perp_basis('BTCUSDT', interval='1h', limit=100), period=14, column='Basis')
    eth_basis = calc_rsi(get_binance_perp_basis('ETHUSDT', interval='1h', limit=100), period=14, column='Basis')

    btc_oi = calc_rsi(df_from_sql(engine=market_engine, table_name='OPENINTBTC', limit=1000, interval='2h'), period=14, column='OI')
    eth_oi = calc_rsi(df_from_sql(engine=market_engine, table_name='OPENINTETH', limit=1000, interval='2h'), period=14, column='OI')

    btc_taker_vol = calc_rsi(calc_rsi(df_from_sql(engine=market_engine, table_name='TAKERVOLBTC', limit=1000, interval='2h'), period=14, column='Perp TBSR'), period=14, column='Spot TBSR')
    eth_taker_vol = calc_rsi(calc_rsi(df_from_sql(engine=market_engine, table_name='TAKERVOLETH', limit=1000, interval='2h'), period=14, column='Perp TBSR'), period=14, column='Spot TBSR')

    df = pd.DataFrame(
        [
            ['BTC', f"{btc_close['Close'].iloc[-1]:,.2f}", f"{btc_close['Volume'].iloc[-2]:,.2f}mm", f"{btc_close['Volume-RSI'].iloc[-2]:,.2f}",
                f"{btc_lsr['LSR'].iloc[-1]:,.4f}", f"{btc_lsr['LSR-RSI'].iloc[-1]:,.2f}", 
                f"{btc_funding['Funding'].iloc[-1]:,.4f}%", f"{btc_funding['Funding-RSI'].iloc[-1]:,.2f}", 
                f"{btc_basis['Basis'].iloc[-1]:,.4f}%", f"{btc_basis['Basis-RSI'].iloc[-1]:,.2f}",
                f"{btc_oi['OI'].iloc[-1]:,.2f}mm", f"{btc_oi['OI-RSI'].iloc[-1]:,.2f}",
                f"{btc_taker_vol['Perp TBSR'].iloc[-1]:,.4f}", f"{btc_taker_vol['Perp TBSR-RSI'].iloc[-1]:,.2f}",
                f"{btc_taker_vol['Spot TBSR'].iloc[-1]:,.4f}", f"{btc_taker_vol['Spot TBSR-RSI'].iloc[-1]:,.2f}",
            ],
            ['ETH', f"{eth_close['Close'].iloc[-1]:,.2f}", f"{eth_close['Volume'].iloc[-2]:,.2f}mm", f"{eth_close['Volume-RSI'].iloc[-2]:,.2f}",
                f"{eth_lsr['LSR'].iloc[-1]:,.4f}", f"{eth_lsr['LSR-RSI'].iloc[-1]:,.2f}", 
                f"{eth_funding['Funding'].iloc[-1]:,.4f}%", f"{eth_funding['Funding-RSI'].iloc[-1]:,.2f}", 
                f"{eth_basis['Basis'].iloc[-1]:,.4f}%", f"{eth_basis['Basis-RSI'].iloc[-1]:,.2f}",
                f"{eth_oi['OI'].iloc[-1]:,.2f}mm", f"{eth_oi['OI-RSI'].iloc[-1]:,.2f}",
                f"{eth_taker_vol['Perp TBSR'].iloc[-1]:,.4f}", f"{eth_taker_vol['Perp TBSR-RSI'].iloc[-1]:,.2f}",
                f"{eth_taker_vol['Spot TBSR'].iloc[-1]:,.4f}", f"{eth_taker_vol['Spot TBSR-RSI'].iloc[-1]:,.2f}",
            ],

        ],
        columns=['Asset', 'Close', 'Volume (5m)', 'RSI', 'LSR', 'RSI', 'Funding', 'RSI', 'Basis', 'RSI', 'Open Int', 'RSI', 'Perp TBSR', 'LSR', 'Spot TBSR', 'LSR']
    )

    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, color="dark", size="sm")

@app.callback(
    Output('dd-orders-table', 'children'),
    Input('algo-dropdown', 'value')
)
def generate_orders_table(dropdown_value):

    engine = create_engine('sqlite:///orders.db')
    inspector = inspect(engine)

    # No existing orders for strategy
    if not inspector.has_table(dropdown_value):
        return

    df = pd.read_sql(f'SELECT * FROM {dropdown_value}', engine)

    # No existing orders for strategy
    if len(df) == 0:
        return

    df.columns = ['Datestamp', 'Asset', 'Side', 'Amount', 'Take-profit', 'Entry', 'Close', 'Elapsed', 'Time-limit']
    df['Take-profit'] = df['Take-profit'].apply(lambda x: f"{x:,.2f}")
    df['Entry'] = df['Entry'].apply(lambda x: f"{x:,.2f}")
    df['Close'] = df['Close'].apply(lambda x: f"{x:,.2f}")
    df['Elapsed'] = df['Elapsed'].apply(lambda x: f"{x:,.2f}")
    df['Time-limit'] = df['Time-limit'].apply(lambda x: f"{x:,.2f}")

    df = df.drop('Datestamp', axis=1)

    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, color="dark", size="sm")

# PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

PLOTLY_LOGO = "assets/mudkip_moon.png"

@app.callback(Output('live-update', 'children'),
              Input('interval-component', 'n_intervals'))
def serve_layout(n):

    return html.Div(children=[
        html.Div(
            html.Img(src=PLOTLY_LOGO, height="50px"),
            style={'width': '5%', 'display': 'inline-block', 'margin-left': '50px'},
        ),
        html.Div(
            html.H2(children="Algo Dashboard"),
            style={'width': '85%', 'display': 'inline-block', 'textAlign': 'center', 'margin-top': '10px'},
        ),
        html.Div(
            dbc.Spinner(color="success"),
            style={'width': '2%', 'display': 'inline-block'},
        ),
        html.Hr(),
        html.Div(
            dropdown,
            style={'margin-left': '20px', 'margin-top': '10px', 'width': '7.5%'},
            className='dash-bootstrap',
        ),
        html.Div(
            dcc.Graph(
                id="dd-output-fig",
                # figure=generate_fig(),
            ),
            style={'width': '80%', 'display': 'inline-block', 'margin-left': '20px'},
        ), 
        html.Div(
            id='dd-output-table',
            # dbc.Table(generate_table()),
            style={'width': '15%', 'display': 'inline-block', 'margin-left': '15px', 'verticalAlign': 'top', 'margin-top': '90px', 'textAlign': 'center'},
        ),
        dbc.Row(
        [
            dbc.Col(
                html.Div(),
                width=3,
            ),
            dbc.Col(
                html.Div(
                    dbc.Table(id='dd-orders-table'),
                    style={'verticalAlign': 'top', 'margin-top': '20px', 'textAlign': 'center'},
                ),
                width=6, # Total width 12
            ),
            dbc.Col(
                html.Div(),
                width=3,
            ),
        ] ,className="g-0" #This allows to have no space between the columns. Delete if you want to have that breathing space
        ),
        dbc.Row(
        [
            dbc.Col(
                html.Div(),
                width=2,
            ),
            dbc.Col(
                html.Div(
                    dbc.Table(generate_market_table()),
                    style={'verticalAlign': 'top', 'margin-top': '20px', 'textAlign': 'center'},
                ),
                width=8, # Total width 12
            ),
            dbc.Col(
                html.Div(),
                width=2,
            ),
        ] ,className="g-0" #This allows to have no space between the columns. Delete if you want to have that breathing space
        ),
    ])

# app.layout = serve_layout

app.layout = html.Div(
    html.Div([
        html.Div(id='live-update'),
        dcc.Interval(
            id='interval-component',
            interval=REFRESH_TIME*1000*60, # in minutes
            n_intervals=0
        )
    ])
)

if __name__ == "__main__":
    app.run_server(debug=False)
