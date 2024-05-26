# ------------------------------------------------------------------------------------------------------------------------------------

#                                          INDUSTRIAL COPPER MODELLING   -   CAPSTONE POJECT


# ------------------------------------------------------------------------------------------------------------------------------------

#                                                      Import the Packages

import streamlit as st
import pickle
import numpy as np
import sklearn
from streamlit_option_menu import option_menu

# ------------------------------------------------------------------------------------------------------------------------------------

#                                             "Parameters" for predict_status() Function

# ctry    -> country
# itmtp   -> item type
# aplcn   -> application
# wth     -> width
# prdrf   -> product reference
# qtlg    -> quantity tons
# cstlg   -> customer log
# tknslg  -> thickness log

# slgplg  -> selling price log
# itmdt   -> Day of item date 
# itmmn   -> Month of Item date
# itmyr   -> Year of Item Date
# deldtdy -> Day of Delivery date
# deldtmn -> Month of Delivery date
# deldtyr -> Year of Delivery date

# ------------------------------------------------------------------------------------------------------------------------------------

#                                                Functions for  "Predict_status()"

def predict_status(ctry,itmtp,aplcn,wth,prdrf,qtlg,cstlg,tknslg,slgplg,itmdt,itmmn,itmyr,deldtdy,deldtmn,deldtyr):

    # change the datatypes "string" to "int"
    itdd = int(itmdt)
    itdm = int(itmmn)
    itdy = int(itmyr)

    dydd = int(deldtdy)
    dydm = int(deldtmn)
    dydy = int(deldtyr)

    # modelfile of the classification
    with open("D://ICM_Project//Classification_model.pkl","rb") as f:

        model_class=pickle.load(f)

    user_data = np.array([[ctry,itmtp,aplcn,wth,prdrf,qtlg,cstlg,tknslg,
                       slgplg,itdd,itdm,itdy,dydd,dydm,dydy]])
    
    y_pred= model_class.predict(user_data)

    if y_pred == 1:

        return 1

    else:

        return 0

# ------------------------------------------------------------------------------------------------------------------------------------

#                                                Functions for  "Predict_selling_price()"

def predict_selling_price(ctry,sts,itmtp,aplcn,wth,prdrf,qtlg,cstlg,
                   tknslg,itmdt,itmmn,itmyr,deldtdy,deldtmn,deldtyr):

    # change the datatypes "string" to "int"
    itdd = int(itmdt)
    itdm = int(itmmn)
    itdy = int(itmyr)

    dydd = int(deldtdy)
    dydm = int(deldtmn)
    dydy = int(deldtyr)

    # modelfile of the classification
    with open("D://ICM_Project//Regression_Model.pkl","rb") as f:

        model_regg=pickle.load(f)

    user_data= np.array([[ctry,sts,itmtp,aplcn,wth,prdrf,qtlg,cstlg,tknslg,
                       itdd,itdm,itdy,dydd,dydm,dydy]])
    
    y_pred= model_regg.predict(user_data)

    ac_y_pred= np.exp(y_pred[0])

    return ac_y_pred

# ------------------------------------------------------------------------------------------------------------------------------------

#                                                       Streamlit UI Part

st.set_page_config(layout= "wide")

st.title(":blue[**INDUSTRIAL COPPER MODELING**]")

with st.sidebar:

    option = option_menu('HARI YOGESH RAM', options = ["HOME", "PREDICT SELLING PRICE", "PREDICT STATUS"])

# ------------------------------------------------------------------------------------------------------------------------------------

if option == 'HOME':

    col1, col2 = st.columns([2, 2], gap="large")

    with col1:

        st.write("---")
        st.markdown("""
            <h2 style='color: red;'>Problem Statement</h2>
        """, unsafe_allow_html=True)
        st.write("""
            <div style='text-align: justify;'>
                <h3 style='font-size: 20px;'>
                    The copper industry faces the following challenges:
                    <br><br>
                    <h4 style='color: green;'> Pricing Predictions : </h4> 
                    <br>  
                    <p><b> Manual predictions can be inaccurate due to skewness and noisy data.</p>
                    <br><br>
                    <h4 style='color: green;'> Lead Classification : </h4>
                    <br>
                    <p><b> Difficulty in capturing and classifying leads effectively. </p>
                </h3>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <h2 style='color: blue;'>Objective</h2>
        """, unsafe_allow_html=True)   
        st.write("""
            <div style='text-align: justify;'>
                <h3 style='font-size: 20px;'>
                    <b>Regression Model</b>: Utilize advanced techniques such as data normalization, feature scaling, and outlier detection to build a robust model for predicting copper prices.
                    <br><br>
                    <b>Classification Model</b>: Build a lead classification system to evaluate and classify leads based on the likelihood of them becoming customers.
                </h3>
            </div>
        """, unsafe_allow_html=True)

    #--------------------------------------------------------------------------------------------------------------------------------
    

    with col2:

        st.write("---")
        st.markdown("""
            <h2 style='color: green;'>Model</h2>
        """, unsafe_allow_html=True)
            
        st.write("""
            <div style='text-align: justify;'>
                <h3 style='font-size: 20px;'>
                    <b>1. Regression Model</b><br>
                    Purpose: Predict the selling price of copper.<br>
                    Techniques Used: Data normalization, feature scaling, outlier detection.<br>
                    Algorithm: Random Forest Regression.
                    <br><br>
                    <b>2. Classification Model</b><br>
                    Purpose: Classify leads as 'Won' or 'Lost'.<br>
                    Techniques Used: Data normalization, feature scaling, handling class imbalance.<br>
                    Algorithm: Random Forest Classifier .
                </h3>
            </div>
        """, unsafe_allow_html=True)

        st.write("---")
        st.write("---")

# ------------------------------------------------------------------------------------------------------------------------------------


if option == "PREDICT STATUS":

    st.header(":red[**PREDICT STATUS (Won / Lose)**]")
    st.write(" ")

    col1,col2= st.columns(2)

    with col1:

        country = st.number_input(label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0")
        item_type = st.number_input(label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0")
        application = st.number_input(label="**Enter the Value for APPLICATION**/ Min:2.0, Max:87.5")
        width = st.number_input(label="**Enter the Value for WIDTH**/ Min:700.0, Max:1980.0")
        product_ref = st.number_input(label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579")
        quantity_tons_log = st.number_input(label="**Enter the Value for QUANTITY_TONS (Log Value)**/ Min:-0.322, Max:6.924",format="%0.15f")
        customer_log = st.number_input(label="**Enter the Value for CUSTOMER (Log Value)**/ Min:17.21910, Max:17.23015",format="%0.15f")
        thickness_log = st.number_input(label="**Enter the Value for THICKNESS (Log Value)**/ Min:-1.71479, Max:3.28154",format="%0.15f")
    
    with col2:

        selling_price_log = st.number_input(label="**Enter the Value for SELLING PRICE (Log Value)**/ Min:5.97503, Max:7.39036",format="%0.15f")
        item_date_day = st.selectbox("**Select the Day for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        item_date_month = st.selectbox("**Select the Month for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        item_date_year = st.selectbox("**Select the Year for ITEM DATE**",("2020","2021"))
        delivery_date_day = st.selectbox("**Select the Day for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        delivery_date_month = st.selectbox("**Select the Month for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        delivery_date_year = st.selectbox("**Select the Year for DELIVERY DATE**",("2020","2021","2022"))
        

    button = st.button(":violet[***PREDICT THE STATUS***]",use_container_width=True)

    if button:

        status = predict_status(country, item_type, application, width, product_ref, quantity_tons_log,
                               customer_log, thickness_log, selling_price_log, item_date_day,
                               item_date_month, item_date_year, delivery_date_day, delivery_date_month,
                               delivery_date_year)
        
        if status == 1:

            st.write("## :green[**The Status is WON**]")

        else:

            st.write("## :red[**The Status is LOSE**]")

# ------------------------------------------------------------------------------------------------------------------------------------

if option == "PREDICT SELLING PRICE":

    st.header("**PREDICT SELLING PRICE**")
    st.write(" ")

    col1,col2= st.columns(2)

    with col1:

        country = st.number_input(label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0")
        status = st.number_input(label="**Enter the Value for STATUS**/ Min:0.0, Max:8.0")
        item_type = st.number_input(label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0")
        application = st.number_input(label="**Enter the Value for APPLICATION**/ Min:2.0, Max:87.5")
        width = st.number_input(label="**Enter the Value for WIDTH**/ Min:700.0, Max:1980.0")
        product_ref = st.number_input(label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579")
        quantity_tons_log = st.number_input(label="**Enter the Value for QUANTITY_TONS (Log Value)**/ Min:-0.3223343801166147, Max:6.924734324081348",format="%0.15f")
        customer_log = st.number_input(label="**Enter the Value for CUSTOMER (Log Value)**/ Min:17.21910565821408, Max:17.230155364880137",format="%0.15f")
        
    
    with col2:

        thickness_log = st.number_input(label="**Enter the Value for THICKNESS (Log Value)**/ Min:-1.7147984280919266, Max:3.281543137578373",format="%0.15f")
        item_date_day = st.selectbox("**Select the Day for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        item_date_month = st.selectbox("**Select the Month for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        item_date_year = st.selectbox("**Select the Year for ITEM DATE**",("2020","2021"))
        delivery_date_day = st.selectbox("**Select the Day for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        delivery_date_month = st.selectbox("**Select the Month for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        delivery_date_year = st.selectbox("**Select the Year for DELIVERY DATE**",("2020","2021","2022"))
        

    button= st.button(":violet[***PREDICT THE SELLING PRICE***]",use_container_width=True)

    if button:

        price= predict_selling_price(country,status, item_type, application, width, product_ref, quantity_tons_log,
                               customer_log, thickness_log, item_date_day,
                               item_date_month, item_date_year, delivery_date_day, delivery_date_month,
                               delivery_date_year)
        
        
        st.write("## :green[**The Selling Price is :**]",price)


# ------------------------------------------------------            END                -----------------------------------------------------------------------





