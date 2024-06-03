import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import time
from predict import train_and_predict


# Tampilkan judul dengan rata tengah dan ukuran font yang lebih besar
st.markdown("<h1 style='text-align: center; font-size: 64px;'>Bitcoin Price Prediction</h1>", unsafe_allow_html=True)


def main():
    # Define the ticker symbol for Bitcoin
    ticker_symbol = "BTC-USD"

    # Streamlit widgets to select start and end date
    start_date = st.date_input('Start date', datetime(2024, 1, 1))
    end_date = st.date_input('End date', datetime.now())

    # Convert dates to string format required by yfinance
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Download data from yfinance
    if st.button('Fetch Data'):
        with st.status("Downloading data...", expanded=True) as status:
            data = yf.download(ticker_symbol, start=start_date_str, end=end_date_str)
            status.update(label="Download complete!", state="complete", expanded=False)
        
        # Set 'Date' not as index and only show the date
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].dt.date

        # Display the dataframe
        st.dataframe(data, use_container_width=True, hide_index=True)

        # Display a line chart
        st.line_chart(data[['Date', 'Close']], x='Date', y='Close')

        # # Save the data to session state for further use
        # st.session_state['data'] = data

    

        # Perform prediction
        with st.status("Processing Data...", expanded=True) as status:
            start_time = time.time()  # Waktu mulai
            predictions, y_test, accuracy, rmse = train_and_predict(data) #Panggil variabel yang dibutuhkan
            end_time = time.time()    # Waktu selesai

            elapsed_time = end_time - start_time  # Waktu yang dibutuhkan

            status.update(label=f"Process Complete! (Took {elapsed_time:.2f} seconds)", state="complete", expanded=False)

            #Evaluate the Model
            st.subheader('Model Accuracy')
            st.subheader(f":red[RMSE:] {rmse}")
            st.subheader(':red[Accuracy:] {:.2f}%'.format(accuracy))

            # Convert to right index
            predictions = pd.Series(predictions, index=pd.RangeIndex(len(predictions)))
            y_test = pd.Series(y_test, index=pd.RangeIndex(len(y_test)))
            
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
            st.line_chart(df)


if __name__ == '__main__':
    main()
