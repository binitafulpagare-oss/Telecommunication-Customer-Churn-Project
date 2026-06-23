import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('churn_model.pkl', 'rb'))
cols  = pickle.load(open('columns.pkl', 'rb'))

st.set_page_config(page_title="Churn Predictor", page_icon="📱")
st.title("📱 Telecom Churn Predictor")
st.divider()

c1, c2 = st.columns(2)
with c1:
    st.subheader("Plan Details")
    intl_plan      = st.selectbox("International Plan?", ["no","yes"])
    voice_plan     = st.selectbox("Voice Plan?", ["no","yes"])
    account_length = st.number_input("Account Length", 1, 250, 100)
    voice_messages = st.number_input("Voice Messages", 0, 60, 0)
with c2:
    st.subheader("Usage Details")
    day_mins   = st.number_input("Day Minutes",   0.0, 400.0, 200.0)
    eve_mins   = st.number_input("Eve Minutes",   0.0, 400.0, 200.0)
    night_mins = st.number_input("Night Minutes", 0.0, 400.0, 200.0)
    intl_mins  = st.number_input("Intl Minutes",  0.0,  60.0,  10.0)

st.divider()
customer_calls = st.slider("Customer Service Calls", 0, 10, 1)
st.divider()

if st.button("Predict Churn Risk", use_container_width=True):
    row = {c: 0 for c in cols}
    row['account_length']  = account_length
    row['voice_plan']      = 1 if voice_plan  == "yes" else 0
    row['voice_messages']  = voice_messages
    row['intl_plan']       = 1 if intl_plan   == "yes" else 0
    row['intl_mins']       = intl_mins
    row['day_mins']        = day_mins
    row['eve_mins']        = eve_mins
    row['night_mins']      = night_mins
    row['customer_calls']  = customer_calls
    input_df = pd.DataFrame([row])[cols]
    prob = model.predict_proba(input_df)[0][1]
    pct  = round(prob * 100, 1)
    st.write("Churn Probability:" , pct,"%")
    if prob > 0.3:
        st.error(f"🚨 HIGH RISK — {pct}% churn probability!")
    else:
        st.success(f"✅ LOW RISK — {pct}% churn probability.")