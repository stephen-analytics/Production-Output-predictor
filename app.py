import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle
from utils import *

# model = pickle.load(open('model.pkl', 'rb'))
# encoder_dict = pickle.load(open('encoder.pkl', 'rb')) 
cols= []
cat_columns = ["Labour", "Transactions", "Capital", "Store Cluster", "Date", "Holiday"]

  
def main(): 
    st.title("Production Output Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"></h2>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html = True)
    
    Date = st.date_input("Production Date", value=None) 
    holiday = st.selectbox("For the given date selected, select if it is a holiday or not",["Yes", "No"])
    Cluster = st.selectbox("For the given store, select the cluster it belongs to",[1,2,3,4,5])
    Transactions = st.number_input("Amount of Transactions") 
    Capital = st.number_input("Amount of Capital") 
    Labour = st.number_input("Labour Cost") 
    Prev_1_month_sales = st.number_input("Total sales recorded in planning period 1") 
    Prev_2_month_sales = st.number_input("Total sales recorded in planning period 2") 
    Prev_3_month_sales = st.number_input("Total sales recorded in planning period 3") 
    Prev_4_month_sales = st.number_input("Total sales recorded in planning period 4") 
    model_type = st.selectbox("Select the Model you want to use",["Random Forests", "Decision Trees"]) 

    if st.button("Predict"): 
        pd_date = pd.to_datetime(Date)
        if holiday == "Yes":
            h_value = 1
        else:
            h_value = 0
        features = [[Labour,Transactions,Capital, int(Cluster), pd_date, h_value ,Prev_1_month_sales ,Prev_2_month_sales,
                    Prev_3_month_sales,Prev_4_month_sales]]
        new_df = preprocess_data(features)
        if model_type == "Random Forests":
            model = pickle.load(open('models/RandomForest.pkl', 'rb'))
            sales_prediction = model.predict(new_df)
        elif model_type == "Decision Trees":
            model = pickle.load(open('models/DecisionTree.pkl', 'rb'))
            sales_prediction = model.predict(new_df)

        st.success(f"Production Output for date: {str(new_df['month'].values[0])}/{str(new_df['year'].values[0])} is {sales_prediction}")
      
if __name__=='__main__': 
    main()

#cd project_lin/Scripts/

#activate.bat

#cd ../..

#streamlit run app.py

