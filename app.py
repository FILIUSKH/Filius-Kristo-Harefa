import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dashboard Machine Learning Pasien",
    page_icon="🏥",
    layout="centered"
)

@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)

linreg_model = load_pickle("linear_regression_model.pkl")
linreg_scaler = load_pickle("linear_regression_scaler.pkl")
linreg_features = load_pickle("linear_regression_features.pkl")

nb_model = load_pickle("naive_bayes_smote_model.pkl")
nb_scaler = load_pickle("naive_bayes_scaler.pkl")
nb_features = load_pickle("naive_bayes_features.pkl")

st.sidebar.title("Menu Aplikasi")

page = st.sidebar.radio(
    "Pilih halaman:",
    [
        "Prediksi Cholesterol",
        "Klasifikasi Penyakit Jantung"
    ]
)

st.title("Dashboard Machine Learning Pasien")


def numeric_input_by_feature(feature_name):
    default_values = {
        "age": 50.0,
        "sex": 1.0,
        "cp": 0.0,
        "trestbps": 120.0,
        "chol": 200.0,
        "fbs": 0.0,
        "restecg": 1.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 1.0,
        "slope": 1.0,
        "ca": 0.0,
        "thal": 2.0
    }

    return st.number_input(
        label=feature_name,
        value=float(default_values.get(feature_name, 0.0))
    )


if page == "Prediksi Cholesterol":
    st.subheader("Prediksi Cholesterol menggunakan Linear Regression")

    st.info(
        "Dataset tidak memiliki kolom Glucose. "
        "Karena itu, target regresi disesuaikan menjadi kolom chol."
    )

    input_data = {}

    with st.form("form_regression"):
        for feature in linreg_features:
            input_data[feature] = numeric_input_by_feature(feature)

        submit = st.form_submit_button("Prediksi Cholesterol")

    if submit:
        input_df = pd.DataFrame([input_data], columns=linreg_features)
        input_scaled = linreg_scaler.transform(input_df)
        prediction = linreg_model.predict(input_scaled)[0]

        st.success(f"Hasil prediksi cholesterol: {prediction:.2f}")


elif page == "Klasifikasi Penyakit Jantung":
    st.subheader("Klasifikasi Penyakit Jantung menggunakan Naive Bayes")

    input_data = {}

    with st.form("form_classification"):
        for feature in nb_features:
            input_data[feature] = numeric_input_by_feature(feature)

        submit = st.form_submit_button("Klasifikasi")

    if submit:
        input_df = pd.DataFrame([input_data], columns=nb_features)
        input_scaled = nb_scaler.transform(input_df)

        prediction = nb_model.predict(input_scaled)[0]
        probability = nb_model.predict_proba(input_scaled)[0]

        if prediction == 1:
            st.error("Hasil klasifikasi: Terindikasi penyakit jantung")
        else:
            st.success("Hasil klasifikasi: Tidak terindikasi penyakit jantung")

        st.write("Probabilitas kelas 0:", round(float(probability[0]), 4))
        st.write("Probabilitas kelas 1:", round(float(probability[1]), 4))
