import numpy as np
import pandas as pd
import streamlit as st
import pickle
import datetime
from datetime import date
import math
st.header("Flight Price Prediction Model")
trans=pickle.load(open('Column transformer2.pkl','rb'))
df=pickle.load(open('DataSet.pkl','rb'))
xgb=pickle.load(open('XGBRegressor.pkl','rb'))
def transforms(arr):
    if arr[0]=='one':
        arr[0]=1
    if arr[0]=='zero':
        arr[0]=0
    if arr[0]=='two_or_more':
        arr[0]=2
    if arr[1]=='Economy':
        arr[1]=0
    if arr[1]=='Business':
        arr[1]=1
    return arr.reshape(1,2)
def options(ar, sc, dc,cls):
    d = df.loc[(df['airline'] == ar) & (df['source_city'] == sc) & (df['destination_city'] == dc)&(df['class']==cls), ['flight', 'stops','departure_time','arrival_time','duration']]
    d = d.groupby(['flight','departure_time','stops','arrival_time'])['duration'].mean()
    d = pd.DataFrame(d)
    d.reset_index(inplace=True)
    return d
def convert(a):
    a=math.modf(a)
    st=(str)((int)(a[1]))+" hours "+(str)((int)(a[0]*60))+" minutes"
    return st
airline=st.selectbox("Select Airline",df['airline'].unique())
sc=st.selectbox("Select Source City",df['source_city'].unique())
dc=st.selectbox("Select Destination City",df['destination_city'].unique())
da=st.date_input("Enter Departure date")
cls=st.selectbox("Select class",df[df['airline']==airline]['class'].unique())
tod=date.today()
df1=options(airline,sc,dc,cls)
days_left=(datetime.date(da.year,da.month,da.day)-tod).days
l1=[]
for i in range(df1.shape[0]):
    l = list(df1.iloc[i, 1:])
    test_input = np.array([airline,sc,l[0],l[1],l[2],dc,cls,l[3],days_left], dtype=object).reshape( 1, 9)
    test_input[[0, 0], [3, 6]] =transforms(test_input[[0, 0], [3, 6]])
    test_input = trans.transform(test_input)
    t=round(xgb.predict(test_input)[0])
    l1.append(t)
l1=pd.DataFrame(l1)
l1.columns=['Predicted Price']
df1["duration"]=df1["duration"].apply(convert)
df1=pd.concat([df1,l1],axis=1)
st.dataframe(df1)