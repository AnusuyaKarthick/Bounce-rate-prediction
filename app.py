# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 07:08:55 2023

@author: KarthickAnu
"""

import streamlit as st
import numpy as np
import pickle


file =open('model.pkl','rb')
bounce_project =pickle.load(file)




def main():
    st.title('Bounce rate Prediction App')
    st.markdown('Just Enter the following details and we will predict type of sales(Medicine) whether it is sold or bouned')
    a = st.slider("Specialisation",0,55)
    b = st.selectbox("Dept",('Department1','Department2','Department3'))
    if b == 'Department1':
       b=0
    elif b == 'Department2':
       b=1
    else:
       b=2
    c = st.number_input("Quantity",min_value=0, max_value=150)
    d = st.slider("ReturnQuantity",0,50)
    e = st.number_input("Final_Cost",min_value=40.00, max_value=5000.00)
    f = st.number_input("Final_Sales",min_value=0.00, max_value=5000.00)
    g = st.number_input("RtnMRP",min_value=0.00, max_value=10000.00)
    h = st.selectbox("Formulation",('Form1','Form2','Form3','Patent'))
    if h == 'Form1':
       h=0
    elif h == 'Form2':
       h=1
    elif h == 'Form3':
       h=2
    else:
       h=3
    i = st.slider("DrugName",0,741)
    j = st.slider("SubCat",0,16)
    k = st.slider("SubCat1",0,20)
    
    submit = st.button('Predict Type of sales')
    if submit: 
       prediction = bounce_project.predict([[a,b,c,d,e,f,g,h,i,j,k]])
       st.write('Bounce rate', prediction)
if __name__ == '__main__':
    main()