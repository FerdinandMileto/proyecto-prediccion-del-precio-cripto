# frontend.py
import pandas as pd
from dash import Dash, dcc, html, Output, Input
import plotly.graph_objects as go

# Configuración
CSV_FILE = "order_book.csv"
REFRESH_INTERVAL = 2000  # ms

# Inicializar app
app = Dash(__name__)

app.layout = html.Div([
    html.H2("Order Book en Tiempo Real - Bitso"),
    dcc.Graph(id='order-book-graph'),
    dcc.Interval(id='interval-component', interval=REFRESH_INTERVAL, n_intervals=0)
])

@app.callback(
    Output('order-book-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    # Leer CSV
    df = pd.read_csv(CSV_FILE)

    bids = df[df['side'] == 'bid'].sort_values('price')
    asks = df[df['side'] == 'ask'].sort_values('price')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bids['price'],
        y=bids['quantity'],
        name='Bids',
        marker_color='green'
    ))

    fig.add_trace(go.Bar(
        x=asks['price'],
        y=asks['quantity'],
        name='Asks',
        marker_color='red'
    ))

    fig.update_layout(
        barmode='overlay',
        title="Order Book Bitso",
        xaxis_title="Precio",
        yaxis_title="Cantidad",
    )

    return fig
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
