import streamlit as st
import datetime
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
from model_script import model


# st.markdown(
#     """
#     <style>
#     .css-k1vhr4 {
#         background: #ee3876;
#     }
#     .css-1avcm0n {
#         background: black;
#     }
#     .css-91z34k {
#         background: black;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


header = st.container()
user_input = st.container()
stock_close_data = st.container()
model_training = st.container()

with header:
    st.title('Stock Forecast App')

with user_input:
    st.subheader("User Input")
    selected_stock = st.text_input('Type in a stock ticker symbol (e.x. GOOG)', value="GOOG")
    start, end = st.columns(2)
    TODAY = date.today().strftime("%Y-%m-%d")
    with start:
        START = st.date_input("Pick a start date for the dataset", datetime.date(2019, 7, 6))
    with end:
        END = st.date_input("Pick an end date for the dataset", date.today())
    st.text("Select the data to use for the prediction")

with stock_close_data:
    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, END)
        data = data.sort_index(ascending=False)
        data.reset_index(inplace=True)
        return data
    
    data = load_data(selected_stock)
    st.subheader('Stock Closing Price History')
    st.write(data.head(7))

    def plot_raw_data():
        fig = go.Figure()
        #fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text="Closing price of $" + selected_stock, xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_raw_data()


with model_training:
    st.subheader("Predictions")
    model_load_state = st.text('Loading data...')
    model(selected_stock, START, END)
    model_load_state.text('Loading data... done!')




