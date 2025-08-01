import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd
from fastapi import FastAPI

# simple global df placeholder
signals_df = pd.DataFrame()


def mount_dash(app: FastAPI, path="/dash"):
    dash_app = dash.Dash(__name__, server=app, routes_pathname_prefix=path + "/")
    dash_app.layout = html.Div([
        html.H2("Nano‑cap Trader Dashboard"),
        dcc.Tabs([
            dcc.Tab(label="Signals", children=[dash_table.DataTable(id="tbl", data=signals_df.to_dict("records"))]),
            dcc.Tab(label="Backtest", children=[dcc.Graph(id="equity")]),
            dcc.Tab(label="Live", children=[html.Div(id="live_status")]),
        ])
    ])

    @dash_app.callback(Output("live_status", "children"), Input("tbl", "data"))
    def _update_status(data):
        return f"{len(data)} signals loaded."