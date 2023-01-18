import streamlit as st
import datetime
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
from model_script import model

header = st.container()
user_input = st.container()
data_table = st.container()
data_graph = st.container()
model_training = st.container()

with header:
    st.title('Stock Forecast App')

with user_input:
    st.header("User Input")
    START = st.date_input("Pick a start date for the dataset", datetime.date(2019, 7, 6))
    TODAY = date.today().strftime("%Y-%m-%d")
    END = st.date_input("Pick an end date for the dataset", date.today())

    selected_stock = st.text_input('Type in a stock ticker symbol (e.x. GOOG)', value="GOOG")
    st.text("Select the data to use for the prediction")

with data_table:
    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, END)
        data = data.sort_index(ascending=False)
        data.reset_index(inplace=True)
        return data


    data_load_state = st.text('Load data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.head(7))

with data_graph:
    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)


    plot_raw_data()

with model_training:
    model(selected_stock, START, END)
