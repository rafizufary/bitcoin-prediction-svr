import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
from predict import train_and_predict

# Tampilkan judul dengan rata tengah dan ukuran font yang lebih besar
# st.markdown("<h1 style='text-align: center; font-size: 64px;'>Bitcoin Price Prediction</h1>", unsafe_allow_html=True)

st.title("Bitcoin Price Prediction")
st.markdown("Web to predict the movement of bitcoin :coin: price based on date you choose.")


def fetch_data(ticker_symbol, start_date, end_date):
    
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return None

def display_data(data):
    st.subheader('Bitcoin Price History')
    data.reset_index(inplace=True)
    data.index = data.index + 1
    data['Date'] = data['Date'].dt.date
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
    ":orange[C is regularization parameter that indicates how much you want to avoid misclassifying each training example]",
    (1, 10, 100))

    # Memilih parameter gamma
    st.subheader("Choose the Gamma Parameter")
    gamma_value = st.selectbox(
    ":orange[Gamma parameter defines how far the influence of a single training example reaches]",
    (0.1, 0.01, 0.001))

    st.subheader("Choose the Train Size")
    split_ratio = st.slider("Select the data split ratio", 10, 90, 80, 10, format="%d%%") / 100
    
    if start_date >= end_date:
        st.error("Start date must be before end date!")
        return

    ticker_symbol = "BTC-USD"
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')


    if st.button('Generate Data'):
        with st.spinner("Downloading data..."):
            end_date_str = end_date + timedelta(days=1)
            data = fetch_data(ticker_symbol, start_date_str, end_date_str)
            if data is not None:
                display_data(data)
                csv = data.to_csv().encode('utf-8')
                st.download_button(label="Download data as CSV", data=csv, file_name='bitcoin_data.csv', mime='text/csv')
                
                # Prediction and further processing
                with st.spinner("Processing Data..."):
                    start_time = time.time()
                    predictions, y_test, accuracy, rmse, future_predictions, future_dates = train_and_predict(data, split_ratio, gamma_value, c_value)
                    # predictions, y_test, accuracy, rmse = train_and_predict(data, split_ratio)
                    st.success(f"Processing complete in {time.time() - start_time:.2f} seconds")

            

                    #Evaluate the Model
                    st.subheader('Parameters Use')
                    col3, col4, col5 = st.columns(3)
                    col3.metric(":blue[C Value]", f"{c_value}")
                    col4.metric(":blue[Gamma Parameter]", f"{gamma_value}")
                    col5.metric(":blue[Train Size]", f"{split_ratio * 100:.0f}%")

                    st.subheader("Model Accuracy")
                    col1, col2 = st.columns(2)
                    col1.metric(":green[RMSE]", f"{rmse:.5f}")
                    col2.metric(":green[Accuracy]", f"{accuracy:.2f}%")



                    # st.subheader(f":red[RMSE:] {rmse:.5f}")
                    # st.subheader(f":red[Accuracy:] {accuracy:.2f}%")
                    # st.subheader(f":red[Test Size:] {split_ratio * 100:.0f}%")
                    # st.subheader(f":red[C Value:] {c_value}")
                    # st.subheader(f":red[Gamma Parameter:] {gamma_value}")

                    # Convert to right index
                    predictions = pd.Series(predictions, index=pd.RangeIndex(len(predictions)))
                    y_test = pd.Series(y_test.flatten(), index=pd.RangeIndex(len(y_test)))
                    
                    # Set Dates for Comparison  
                    start_index_for_dates = len(data) - len(y_test)  # Index awal untuk tanggal data pengujian
                    test_dates = data['Date'][start_index_for_dates:].reset_index(drop=True)  # Reset index untuk menghindari masalah indeks

                    # Display the results
                    comparison_df = pd.DataFrame({
                        'Date': test_dates, 
                        'Actual': y_test, 
                        'Predicted': predictions})
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                    # Plotting data
                    df = pd.DataFrame(comparison_df)
                    df.set_index('Date', inplace=True)
                    st.line_chart(df, color=["#F3BA2F", "#00aced"])

                    # Display future predictions
                    future_data = pd.DataFrame({
                        'Date': future_dates, 
                        'Predicted Close': future_predictions})
                    st.subheader("Predicted Bitcoin Prices for the Next 7 Days")
                    st.dataframe(future_data, use_container_width=True, hide_index=True)

                    
if __name__ == '__main__':
    main()
