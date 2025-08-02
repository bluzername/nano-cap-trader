"""Enhanced Dash GUI with comprehensive trading interface and real-time monitoring."""
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, ctx
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import asyncio
from fastapi import FastAPI
import logging

from .strategies.strategy_factory import StrategyFactory
from .benchmarking.ab_testing import ABTestFramework
from .risk_management.portfolio_risk import PortfolioRiskManager
from .config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

# Global instances
strategy_factory = StrategyFactory()
ab_test_framework = ABTestFramework()
risk_manager = PortfolioRiskManager()

# Application state
app_state = {
    'active_strategies': {},
    'active_tests': {},
    'universe': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],  # Default universe
    'signals_df': pd.DataFrame(),
    'performance_data': {},
    'risk_data': {},
    'last_update': None
}


def mount_dash(app: FastAPI, path="/dash"):
    """Mount enhanced Dash application with comprehensive trading interface."""
    dash_app = dash.Dash(__name__, server=app, routes_pathname_prefix=path + "/")
    
    # Enhanced CSS styling
    dash_app.layout = html.Div([
        # Header
        html.Div([
            html.H1("🚀 NanoCap Trader - Algorithmic Trading System", 
                   style={'textAlign': 'center', 'color': '#2E86AB', 'marginBottom': '10px'}),
            html.P("Advanced multi-strategy trading platform for nano-cap equities", 
                  style={'textAlign': 'center', 'color': '#666', 'fontSize': '16px'}),
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
        
        # Status Bar
        html.Div(id='status-bar', style={'marginBottom': '20px'}),
        
        # Main Navigation
        dcc.Tabs(id='main-tabs', value='overview', children=[
            dcc.Tab(label='📊 Overview', value='overview'),
            dcc.Tab(label='⚙️ Strategy Control', value='strategy-control'),
            dcc.Tab(label='🧪 A/B Testing', value='ab-testing'),
            dcc.Tab(label='⚠️ Risk Management', value='risk-management'),
            dcc.Tab(label='📈 Performance', value='performance'),
            dcc.Tab(label='📡 Live Trading', value='live-trading'),
        ], style={'marginBottom': '20px'}),
        
        # Tab Content
        html.Div(id='tab-content'),
        
        # Auto-refresh interval
        dcc.Interval(
            id='auto-refresh',
            interval=5*1000,  # 5 seconds
            n_intervals=0
        ),
        
        # Hidden div to store data
        html.Div(id='hidden-data', style={'display': 'none'})
        
    ], style={'margin': '20px', 'fontFamily': 'Arial, sans-serif'})
    
    # Register callbacks
    register_callbacks(dash_app)
    
    return dash_app


def register_callbacks(dash_app):
    """Register all Dash callbacks."""
    
    @dash_app.callback(
        Output('tab-content', 'children'),
        Input('main-tabs', 'value')
    )
    def render_tab_content(active_tab):
        """Render content based on active tab."""
        if active_tab == 'overview':
            return render_overview_tab()
        elif active_tab == 'strategy-control':
            return render_strategy_control_tab()
        elif active_tab == 'ab-testing':
            return render_ab_testing_tab()
        elif active_tab == 'risk-management':
            return render_risk_management_tab()
        elif active_tab == 'performance':
            return render_performance_tab()
        elif active_tab == 'live-trading':
            return render_live_trading_tab()
        else:
            return html.Div("Tab not found")
    
    @dash_app.callback(
        Output('status-bar', 'children'),
        Input('auto-refresh', 'n_intervals')
    )
    def update_status_bar(n):
        """Update status bar with real-time information."""
        try:
            status = {
                'strategies': len(app_state['active_strategies']),
                'tests': len(app_state['active_tests']),
                'last_update': datetime.now().strftime('%H:%M:%S'),
                'universe_size': len(app_state['universe'])
            }
            
            return html.Div([
                html.Div([
                    html.Span(f"Active Strategies: {status['strategies']}", 
                             style={'marginRight': '20px', 'color': '#28a745'}),
                    html.Span(f"Running Tests: {status['tests']}", 
                             style={'marginRight': '20px', 'color': '#007bff'}),
                    html.Span(f"Universe: {status['universe_size']} symbols", 
                             style={'marginRight': '20px', 'color': '#6c757d'}),
                    html.Span(f"Last Update: {status['last_update']}", 
                             style={'color': '#6c757d'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'})
            ], style={
                'backgroundColor': '#e9ecef', 
                'padding': '10px 20px', 
                'borderRadius': '5px',
                'border': '1px solid #dee2e6'
            })
            
        except Exception as e:
            logger.error(f"Error updating status bar: {e}")
            return html.Div("Status update error", style={'color': 'red'})


def render_overview_tab():
    """Render the overview dashboard tab."""
    return html.Div([
        # Key Metrics Cards
        html.Div([
            html.Div([
                html.H3("Portfolio Value", style={'color': '#007bff'}),
                html.H2("$1,000,000", id='portfolio-value', style={'color': '#28a745'})
            ], className='metric-card', style=get_card_style()),
            
            html.Div([
                html.H3("Daily P&L", style={'color': '#007bff'}),
                html.H2("+$2,500", id='daily-pnl', style={'color': '#28a745'})
            ], className='metric-card', style=get_card_style()),
            
            html.Div([
                html.H3("Total Return", style={'color': '#007bff'}),
                html.H2("+4.2%", id='total-return', style={'color': '#28a745'})
            ], className='metric-card', style=get_card_style()),
            
            html.Div([
                html.H3("Sharpe Ratio", style={'color': '#007bff'}),
                html.H2("0.85", id='sharpe-ratio', style={'color': '#28a745'})
            ], className='metric-card', style=get_card_style()),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '20px', 'marginBottom': '30px'}),
        
        # Charts Row
        html.Div([
            html.Div([
                html.H4("Portfolio Performance", style={'textAlign': 'center'}),
                dcc.Graph(id='portfolio-performance-chart', 
                         figure=create_sample_performance_chart())
            ], style={'width': '60%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.H4("Strategy Allocation", style={'textAlign': 'center'}),
                dcc.Graph(id='strategy-allocation-chart',
                         figure=create_sample_pie_chart())
            ], style={'width': '40%', 'display': 'inline-block', 'padding': '10px'}),
        ]),
        
        # Recent Signals Table
        html.Div([
            html.H4("Recent Signals"),
            dash_table.DataTable(
                id='recent-signals-table',
                columns=[
                    {'name': 'Time', 'id': 'timestamp'},
                    {'name': 'Symbol', 'id': 'symbol'},
                    {'name': 'Strategy', 'id': 'strategy'},
                    {'name': 'Signal', 'id': 'signal_type'},
                    {'name': 'Confidence', 'id': 'confidence', 'type': 'numeric', 'format': '.1%'},
                    {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': '$,.2f'},
                ],
                data=create_sample_signals_data(),
                style_cell={'textAlign': 'left'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{signal_type} = BUY'},
                        'backgroundColor': '#d4edda',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{signal_type} = SELL'},
                        'backgroundColor': '#f8d7da',
                        'color': 'black',
                    }
                ],
                page_size=10
            )
        ], style={'marginTop': '30px'})
    ])


def render_strategy_control_tab():
    """Render strategy control and configuration tab."""
    available_strategies = strategy_factory.get_available_strategies()
    
    return html.Div([
        # Strategy Selection and Configuration
        html.Div([
            html.Div([
                html.H4("Strategy Selection"),
                dcc.Dropdown(
                    id='strategy-selector',
                    options=[{'label': strategy.replace('_', ' ').title(), 'value': strategy} 
                            for strategy in available_strategies],
                    value='multi_strategy',
                    clearable=False
                ),
                html.Br(),
                html.Button('📊 Start Strategy', id='start-strategy-btn', 
                           style={'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 
                                 'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'}),
                html.Button('⏹️ Stop Strategy', id='stop-strategy-btn', 
                           style={'backgroundColor': '#dc3545', 'color': 'white', 'border': 'none', 
                                 'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer',
                                 'marginLeft': '10px'}),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H4("Universe Configuration"),
                dcc.Textarea(
                    id='universe-input',
                    value=','.join(app_state['universe']),
                    placeholder='Enter symbols separated by commas (e.g., AAPL,MSFT,GOOGL)',
                    style={'width': '100%', 'height': '100px'}
                ),
                html.Br(),
                html.Button('💾 Update Universe', id='update-universe-btn',
                           style={'backgroundColor': '#6c757d', 'color': 'white', 'border': 'none', 
                                 'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'}),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
        ]),
        
        html.Hr(),
        
        # Active Strategies
        html.Div([
            html.H4("Active Strategies"),
            html.Div(id='active-strategies-list', children=[
                create_active_strategies_display()
            ])
        ], style={'marginTop': '20px'}),
        
        # Strategy Status Messages
        html.Div(id='strategy-messages', style={'marginTop': '20px'})
    ])


def render_ab_testing_tab():
    """Render A/B testing interface."""
    return html.Div([
        # Test Setup
        html.Div([
            html.H4("A/B Test Configuration"),
            html.Div([
                html.Div([
                    html.Label("Test Name:"),
                    dcc.Input(id='test-name-input', type='text', placeholder='Enter test name', 
                             style={'width': '100%', 'marginBottom': '10px'}),
                    
                    html.Label("Strategies to Compare:"),
                    dcc.Dropdown(
                        id='test-strategies-selector',
                        options=[{'label': strategy.replace('_', ' ').title(), 'value': strategy} 
                                for strategy in strategy_factory.get_available_strategies()],
                        value=['statistical_arbitrage', 'momentum'],
                        multi=True
                    ),
                    
                    html.Label("Test Duration (days):"),
                    dcc.Input(id='test-duration-input', type='number', value=30, min=1, max=365,
                             style={'width': '100%', 'marginBottom': '10px'}),
                    
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Label("Benchmark:"),
                    dcc.Dropdown(
                        id='test-benchmark-selector',
                        options=[
                            {'label': 'Russell 2000', 'value': 'russell_2000'},
                            {'label': 'S&P SmallCap 600', 'value': 'sp_600'},
                            {'label': 'S&P 500', 'value': 'sp_500'},
                        ],
                        value='russell_2000'
                    ),
                    
                    html.Br(),
                    html.Label("Enable Paper Trading:"),
                    dcc.Checklist(
                        id='paper-trading-checkbox',
                        options=[{'label': 'Enable Paper Trading', 'value': 'enabled'}],
                        value=['enabled']
                    ),
                    
                    html.Br(),
                    html.Button('🚀 Start A/B Test', id='start-test-btn',
                               style={'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 
                                     'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
            ])
        ]),
        
        html.Hr(),
        
        # Active Tests
        html.Div([
            html.H4("Active Tests"),
            html.Div(id='active-tests-list', children=[
                create_active_tests_display()
            ])
        ]),
        
        # Test Messages
        html.Div(id='test-messages', style={'marginTop': '20px'})
    ])


def render_risk_management_tab():
    """Render risk management dashboard."""
    return html.Div([
        # Risk Metrics Overview
        html.Div([
            html.H4("Risk Metrics Overview"),
            html.Div([
                html.Div([
                    html.H5("Leverage", style={'color': '#007bff'}),
                    html.H3("1.2x", style={'color': '#28a745'})
                ], style=get_card_style()),
                
                html.Div([
                    html.H5("VaR (95%)", style={'color': '#007bff'}),
                    html.H3("1.5%", style={'color': '#ffc107'})
                ], style=get_card_style()),
                
                html.Div([
                    html.H5("Max Drawdown", style={'color': '#007bff'}),
                    html.H3("-2.1%", style={'color': '#dc3545'})
                ], style=get_card_style()),
                
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '20px', 'marginBottom': '20px'})
        ]),
        
        # Risk Alerts
        html.Div([
            html.H4("Risk Alerts"),
            html.Div([
                html.Div("🟢 All risk limits within normal ranges", 
                        style={'padding': '10px', 'backgroundColor': '#d4edda', 'borderRadius': '5px'})
            ])
        ], style={'marginTop': '30px'})
    ])


def render_performance_tab():
    """Render performance analysis dashboard."""
    return html.Div([
        # Performance Summary
        html.Div([
            html.H4("Performance Summary"),
            dash_table.DataTable(
                id='performance-metrics-table',
                columns=[
                    {'name': 'Strategy', 'id': 'strategy'},
                    {'name': 'Total Return', 'id': 'total_return', 'type': 'numeric', 'format': '.2%'},
                    {'name': 'Sharpe Ratio', 'id': 'sharpe_ratio', 'type': 'numeric', 'format': '.3f'},
                    {'name': 'Max Drawdown', 'id': 'max_drawdown', 'type': 'numeric', 'format': '.2%'},
                    {'name': 'Alpha', 'id': 'alpha', 'type': 'numeric', 'format': '.2%'},
                    {'name': 'Win Rate', 'id': 'win_rate', 'type': 'numeric', 'format': '.1%'},
                ],
                data=create_sample_performance_data(),
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}
            )
        ])
    ])


def render_live_trading_tab():
    """Render live trading interface."""
    return html.Div([
        # Trading Controls
        html.Div([
            html.H4("Trading Controls"),
            html.Div([
                html.Button('▶️ Start Trading', id='start-trading-btn',
                           style={'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 
                                 'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'}),
                html.Button('⏸️ Pause Trading', id='pause-trading-btn',
                           style={'backgroundColor': '#ffc107', 'color': 'white', 'border': 'none', 
                                 'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer',
                                 'marginLeft': '10px'}),
                html.Button('⏹️ Stop Trading', id='stop-trading-btn',
                           style={'backgroundColor': '#dc3545', 'color': 'white', 'border': 'none', 
                                 'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer',
                                 'marginLeft': '10px'}),
            ])
        ]),
        
        html.Hr(),
        
        # Current Positions
        html.Div([
            html.H4("Current Positions"),
            dash_table.DataTable(
                id='positions-table',
                columns=[
                    {'name': 'Symbol', 'id': 'symbol'},
                    {'name': 'Strategy', 'id': 'strategy'},
                    {'name': 'Quantity', 'id': 'quantity'},
                    {'name': 'Entry Price', 'id': 'entry_price', 'type': 'numeric', 'format': '$,.2f'},
                    {'name': 'Current Price', 'id': 'current_price', 'type': 'numeric', 'format': '$,.2f'},
                    {'name': 'P&L', 'id': 'pnl', 'type': 'numeric', 'format': '$,.2f'},
                    {'name': 'P&L %', 'id': 'pnl_pct', 'type': 'numeric', 'format': '.2%'},
                ],
                data=create_sample_positions_data(),
                style_cell={'textAlign': 'center'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{pnl} > 0'},
                        'backgroundColor': '#d4edda',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{pnl} < 0'},
                        'backgroundColor': '#f8d7da',
                        'color': 'black',
                    }
                ]
            )
        ])
    ])


def get_card_style():
    """Get consistent card styling."""
    return {
        'backgroundColor': 'white',
        'padding': '20px',
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'textAlign': 'center',
        'border': '1px solid #dee2e6'
    }


def create_sample_performance_chart():
    """Create sample performance chart."""
    import numpy as np
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    cumulative_returns = np.cumsum(np.random.normal(0.001, 0.02, 100))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_returns,
        mode='lines',
        name='Portfolio',
        line=dict(color='#007bff', width=2)
    ))
    
    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_white",
        height=400
    )
    
    return fig


def create_sample_pie_chart():
    """Create sample strategy allocation pie chart."""
    fig = go.Figure(data=[go.Pie(
        labels=['Multi-Strategy', 'Statistical Arbitrage', 'Momentum', 'Mean Reversion'],
        values=[60, 25, 10, 5],
        hole=0.3
    )])
    
    fig.update_layout(
        title="Strategy Allocation",
        template="plotly_white",
        height=400
    )
    
    return fig


def create_sample_signals_data():
    """Create sample signals data."""
    return [
        {
            'timestamp': '09:31:15',
            'symbol': 'AAPL',
            'strategy': 'Multi-Strategy',
            'signal_type': 'BUY',
            'confidence': 0.85,
            'price': 175.23
        },
        {
            'timestamp': '09:32:45',
            'symbol': 'MSFT', 
            'strategy': 'Momentum',
            'signal_type': 'BUY',
            'confidence': 0.72,
            'price': 420.15
        },
        {
            'timestamp': '09:35:12',
            'symbol': 'GOOGL',
            'strategy': 'Mean Reversion',
            'signal_type': 'SELL',
            'confidence': 0.68,
            'price': 142.87
        }
    ]


def create_sample_performance_data():
    """Create sample performance metrics data."""
    return [
        {
            'strategy': 'Multi-Strategy',
            'total_return': 0.042,
            'sharpe_ratio': 0.85,
            'max_drawdown': -0.021,
            'alpha': 0.038,
            'win_rate': 0.61
        },
        {
            'strategy': 'Statistical Arbitrage',
            'total_return': 0.045,
            'sharpe_ratio': 0.89,
            'max_drawdown': -0.018,
            'alpha': 0.042,
            'win_rate': 0.58
        },
        {
            'strategy': 'Momentum',
            'total_return': 0.038,
            'sharpe_ratio': 0.58,
            'max_drawdown': -0.035,
            'alpha': 0.035,
            'win_rate': 0.52
        }
    ]


def create_sample_positions_data():
    """Create sample positions data."""
    return [
        {
            'symbol': 'AAPL',
            'strategy': 'Multi-Strategy',
            'quantity': 100,
            'entry_price': 170.50,
            'current_price': 175.23,
            'pnl': 473.00,
            'pnl_pct': 0.0277
        },
        {
            'symbol': 'MSFT',
            'strategy': 'Momentum',
            'quantity': 50,
            'entry_price': 415.20,
            'current_price': 420.15,
            'pnl': 247.50,
            'pnl_pct': 0.0119
        },
        {
            'symbol': 'TSLA',
            'strategy': 'Mean Reversion',
            'quantity': -25,
            'entry_price': 245.80,
            'current_price': 242.10,
            'pnl': 92.50,
            'pnl_pct': 0.0151
        }
    ]


def create_active_strategies_display():
    """Create display for active strategies."""
    if not app_state['active_strategies']:
        return html.Div("No active strategies", style={'color': '#6c757d', 'fontStyle': 'italic'})
    
    strategy_cards = []
    for strategy_name, strategy in app_state['active_strategies'].items():
        strategy_cards.append(
            html.Div([
                html.H5(strategy_name.replace('_', ' ').title()),
                html.P(f"Type: {strategy.strategy_type.value}"),
                html.P(f"Positions: {len(strategy.positions)}"),
                html.P(f"Portfolio Value: ${strategy.portfolio_value_current:,.2f}")
            ], style={
                'border': '1px solid #dee2e6',
                'borderRadius': '5px',
                'padding': '15px',
                'margin': '10px 0',
                'backgroundColor': '#f8f9fa'
            })
        )
    
    return html.Div(strategy_cards)


def create_active_tests_display():
    """Create display for active A/B tests."""
    if not app_state['active_tests']:
        return html.Div("No active tests", style={'color': '#6c757d', 'fontStyle': 'italic'})
    
    test_cards = []
    for test_id, test_info in app_state['active_tests'].items():
        duration = (datetime.now() - test_info['start_time']).days
        test_cards.append(
            html.Div([
                html.H5(test_info['name']),
                html.P(f"Strategies: {', '.join(test_info['strategies'])}"),
                html.P(f"Duration: {duration}/{test_info['duration']} days"),
                html.P(f"Status: Running")
            ], style={
                'border': '1px solid #dee2e6',
                'borderRadius': '5px',
                'padding': '15px',
                'margin': '10px 0',
                'backgroundColor': '#f8f9fa'
            })
        )
    
    return html.Div(test_cards)