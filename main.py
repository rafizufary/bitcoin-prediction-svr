import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
from predict import train_and_predict


st.title("Bitcoin Price Prediction")
st.markdown("Web to predict the movement of bitcoin :coin: price based on chosen parameters.")


def fetch_data(start_date, end_date):
    try:
        data = yf.download('BTC-USD', start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return None

def display_data(data):
    st.subheader('Bitcoin Price History')
    # data.reset_index(inplace=True)
    # data.index = data.index + 1
    # data['Date'] = data['Date'].dt.date
    st.dataframe(data, use_container_width=True)
    st.line_chart(data[['Date', 'Close']], x='Date', y='Close', color=["#F3BA2F"])

def main():
    # Memilih tanggal prediksi
    st.subheader("Choose the Date")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start date', datetime(2019, 6, 1))
    with col2:
        end_date = st.date_input('End date', datetime(2024, 6, 1))
    
    # Memilih Paramater C
    st.subheader("Choose the C Parameter")
    c_value = st.selectbox(
    ":orange[C is regularization parameter that serves as the penalty limit for tolerating the error of a prediction.]",
    (0.1, 1, 10))

    # Memilih parameter gamma
    st.subheader("Choose the Gamma Parameter")
    gamma_value = st.selectbox(
    ":orange[Gamma parameter defines how far the influence of a single training example reaches]",
    (0.01, 0.1, 1))

    # Memilih rasio data latih
    st.subheader("Choose the Train Size")
    split_ratio = st.slider("Select the data split ratio", 10, 90, 80, 10, format="%d%%") / 100
    
    if start_date >= end_date:
        st.error("Start date must be before end date!")
        return

    # Konversi tanggal menjadi string
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Crawling data
    if st.button('Generate Data'):
        st.markdown('')
        with st.spinner("Downloading data..."):
            end_date_str = end_date + timedelta(days=1)
            data = fetch_data(start_date_str, end_date_str)
            if data is not None:
                display_data(data)
                
            # Proses data
        with st.spinner("Processing Data..."):
            st.markdown('')
            start_time = time.time()
            predictions, y_test, accuracy, mape, future_predictions, future_dates, price_movement = train_and_predict(data, split_ratio, gamma_value, c_value)
            st.success(f"Processing complete in {time.time() - start_time:.2f} seconds")

            

            #Evaluasi Model
            st.subheader('Parameters Use')
            col3, col4, col5 = st.columns(3)
            col3.metric(":orange[C Value]", f"{c_value}")
            col4.metric(":orange[Gamma Parameter]", f"{gamma_value}")
            col5.metric(":orange[Train Size]", f"{split_ratio * 100:.0f}%")

            st.subheader("Model Accuracy")
            col1, col2 = st.columns(2)
            col1.metric(":blue[MAPE]", f"{mape:.4f}%")
            col2.metric(":blue[Accuracy]", f"{accuracy:.2f}%")
            st.markdown('')

            # Konversi ke index yang sesuai
            predictions = pd.Series(predictions, index=pd.RangeIndex(len(predictions)))
            y_test = pd.Series(y_test.flatten(), index=pd.RangeIndex(len(y_test)))
                    
            # Inisialisasi tanggal untuk perbandingan data
            start_index_for_dates = len(data) - len(y_test)  # Index awal untuk tanggal data pengujian
            test_dates = data['Date'][start_index_for_dates:].reset_index(drop=True)  # Reset index untuk menghindari masalah indeks

            # Tampilkan perbandingan
            st.subheader('Actual and Prediction Comparison')
            comparison_df = pd.DataFrame({
                'Date': test_dates, 
                'Actual': y_test, 
                'Predicted': predictions})
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            # Tampilkan grafik
            df = pd.DataFrame(comparison_df)
            df.set_index('Date', inplace=True)
            st.line_chart(df, color=["#F3BA2F", "#00aced"])

            # Tampilkan peramalan di masa depan
            future_data = pd.DataFrame({
                'Date': future_dates, 
                'Predicted Close': future_predictions})
            st.subheader("Predicted Bitcoin Close Prices for the Next 7 Days")
            st.dataframe(future_data, use_container_width=True, hide_index=True)
            st.subheader(f"Price will likely to: :blue[{price_movement}]")

                    
if __name__ == '__main__':
    main()
