import pickle
import streamlit as st
import pandas as pd
from PIL import Image

# Load model
model_file = 'model_C=1.0.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

def main():
    # App Header
    st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
    
    image = Image.open('images/icone.png')
    st.image(image, width=700)  # Updated: use width instead of use_column_width
    st.title("Customer Churn Prediction")
    st.markdown("Predict if a customer is likely to churn and get a risk score.")

    # Sidebar
    add_selectbox = st.sidebar.selectbox(
        "Prediction Mode",
        ("Online", "Batch")
    )
    st.sidebar.info("This app predicts Customer Churn using ML models.")
    st.sidebar.image(Image.open('images/image.png'), width=200)  # Updated: width

    if add_selectbox == "Online":
        with st.form("churn_form"):
            st.subheader("Customer Details")
            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox('Gender', ['male', 'female'])
                seniorcitizen = st.selectbox('Senior Citizen', [0, 1])
                partner = st.selectbox('Partner', ['yes', 'no'])
                dependents = st.selectbox('Dependents', ['yes', 'no'])
                phoneservice = st.selectbox('Phone Service', ['yes', 'no'])
                multiplelines = st.selectbox('Multiple Lines', ['yes', 'no', 'no_phone_service'])

            with col2:
                internetservice = st.selectbox('Internet Service', ['dsl', 'no', 'fiber_optic'])
                onlinesecurity = st.selectbox('Online Security', ['yes', 'no', 'no_internet_service'])
                onlinebackup = st.selectbox('Online Backup', ['yes', 'no', 'no_internet_service'])
                deviceprotection = st.selectbox('Device Protection', ['yes', 'no', 'no_internet_service'])
                techsupport = st.selectbox('Tech Support', ['yes', 'no', 'no_internet_service'])
                streamingtv = st.selectbox('Streaming TV', ['yes', 'no', 'no_internet_service'])
                streamingmovies = st.selectbox('Streaming Movies', ['yes', 'no', 'no_internet_service'])

            st.subheader("Billing & Contract")
            contract = st.selectbox('Contract', ['month-to-month', 'one_year', 'two_year'])
            paperlessbilling = st.selectbox('Paperless Billing', ['yes', 'no'])
            paymentmethod = st.selectbox('Payment Method', [
                'bank_transfer_(automatic)',
                'credit_card_(automatic)',
                'electronic_check',
                'mailed_check'
            ])
            tenure = st.number_input('Tenure (months)', min_value=0, max_value=240, value=0)
            monthlycharges = st.number_input('Monthly Charges', min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
            totalcharges = tenure * monthlycharges

            submitted = st.form_submit_button("Predict")
            if submitted:
                input_dict = {
                    "gender": gender,
                    "seniorcitizen": seniorcitizen,
                    "partner": partner,
                    "dependents": dependents,
                    "phoneservice": phoneservice,
                    "multiplelines": multiplelines,
                    "internetservice": internetservice,
                    "onlinesecurity": onlinesecurity,
                    "onlinebackup": onlinebackup,
                    "deviceprotection": deviceprotection,
                    "techsupport": techsupport,
                    "streamingtv": streamingtv,
                    "streamingmovies": streamingmovies,
                    "contract": contract,
                    "paperlessbilling": paperlessbilling,
                    "paymentmethod": paymentmethod,
                    "tenure": tenure,
                    "monthlycharges": monthlycharges,
                    "totalcharges": totalcharges
                }
                X = dv.transform([input_dict])
                y_pred = model.predict_proba(X)[0, 1]
                churn = y_pred >= 0.5

                st.metric(label="Churn Risk", value=f"{y_pred:.2%}", delta="High" if churn else "Low")
                st.success(f"Predicted Churn: {churn}")

    elif add_selectbox == "Batch":
        file_upload = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
        if file_upload:
            data = pd.read_csv(file_upload)
            X = dv.transform(data.to_dict(orient="records"))
            y_pred = model.predict_proba(X)[:, 1]
            results = pd.DataFrame({"Churn_Probability": y_pred, "Churn_Prediction": y_pred >= 0.5})
            st.dataframe(results)

if __name__ == "__main__":
    main()