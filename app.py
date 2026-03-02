import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="SVC V3 - Predicción SAIDI", layout="centered")

st.title("Predicción SAIDI — SVC V5")
st.markdown("Carga el modelo y los transformadores desde la carpeta Joblib y completa los campos para predecir SAIDI.")

# rutas esperadas
MODEL_PATH = os.path.join('Joblib', 'best_SVC_model_V5.joblib')
ENCODER_PATH = os.path.join('Joblib', 'One_encoder.joblib')
SCALER_PATH = os.path.join('Joblib', 'MinMax_escaler.joblib')

def load_artifacts():
    missing = []
    if not os.path.exists(MODEL_PATH):
        missing.append(MODEL_PATH)
    if not os.path.exists(ENCODER_PATH):
        missing.append(ENCODER_PATH)
    if not os.path.exists(SCALER_PATH):
        missing.append(SCALER_PATH)
    if missing:
        st.error(f"Faltan archivos: {missing}. Colócalos en la carpeta Joblib.")
        return None, None, None
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, encoder, scaler

model, encoder, scaler = load_artifacts()
if model is None:
    st.stop()

# Variables usadas en el notebook (entrada)
variables_categoricas = ['Empresa ID', 'dia_semana', 'mes', 'viento_HoraLocal', 'Comuna']
variables_escalar = ['Clientes Instalados', 'viento_Viento', 'viento_Rafagas', 'viento_Nubosidad', 'viento_Precipitacion', 'viento_PresionAtm', 'viento_Temperatura']

st.header('Entrada de datos')
col1, col2 = st.columns(2)

with col1:
    empresa = st.selectbox('Empresa ID', encoder.categories_[variables_categoricas.index('Empresa ID')].tolist())
    dia_semana = st.selectbox('Día de la semana', ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    mes = st.number_input('Mes (1-12)', min_value=1, max_value=12, value=1)
    hora_local = st.number_input('Hora local (0-23)', min_value=0, max_value=23, value=12)
    comuna = st.selectbox('Comuna', encoder.categories_[variables_categoricas.index('Comuna')].tolist())

with col2:
    clientes = st.number_input('Clientes Instalados', min_value=0, value=1000)
    viento_viento = st.number_input('viento_Viento (m/s)', value=2.5, format="%.2f")
    viento_rafagas = st.number_input('viento_Rafagas (m/s)', value=3.0, format="%.2f")
    viento_nubosidad = st.number_input('viento_Nubosidad (%)', min_value=0.0, max_value=100.0, value=20.0, format="%.1f")
    viento_precip = st.number_input('viento_Precipitacion (mm)', min_value=0.0, value=0.0, format="%.2f")
    viento_presion = st.number_input('viento_PresionAtm (hPa)', value=1013.0, format="%.2f")
    viento_temp = st.number_input('viento_Temperatura (°C)', value=15.0, format="%.2f")
    dir_deg = st.number_input('viento_DireccionViento (grados 0-360)', min_value=0.0, max_value=360.0, value=90.0, format="%.1f")

if st.button('Predecir SAIDI'):
    # construir DataFrame de entrada
    # crear componente sen y cos desde grados
    rad = np.deg2rad(dir_deg)
    viento_dir_sin = np.sin(rad)
    viento_dir_cos = np.cos(rad)

    # dataframe categórico y numérico
    df_cat = pd.DataFrame([[empresa, dia_semana, mes, hora_local, comuna]], columns=variables_categoricas)
    df_num = pd.DataFrame([[clientes, viento_viento, viento_rafagas, viento_nubosidad, viento_precip, viento_presion, viento_temp]], columns=variables_escalar)

    # aplicar encoder a categóricas
    try:
        cat_enc = encoder.transform(df_cat)
        try:
            cat_cols = encoder.get_feature_names_out(variables_categoricas)
        except Exception:
            # fallback: nombres genéricos
            cat_cols = [f"cat_{i}" for i in range(cat_enc.shape[1])]
        df_cat_enc = pd.DataFrame(cat_enc, columns=cat_cols, index=[0])
    except Exception as e:
        st.error(f"Error al transformar variables categóricas: {e}")
        st.stop()

    # escalar numéricas
    try:
        df_num_scaled = pd.DataFrame(scaler.transform(df_num), columns=variables_escalar, index=[0])
    except Exception as e:
        st.error(f"Error al escalar variables numéricas: {e}")
        st.stop()

    # añadir componentes direccionales
    df_num_scaled['viento_dir_sin'] = viento_dir_sin
    df_num_scaled['viento_dir_cos'] = viento_dir_cos

    # concatenar
    X_input = pd.concat([df_num_scaled.reset_index(drop=True), df_cat_enc.reset_index(drop=True)], axis=1)

    # Asegurar orden de columnas esperado por el modelo es dependiente del entrenamiento
    try:
        y_pred = model.predict(X_input)
        st.success(f'Predicción SAIDI: {y_pred[0]:.4f}')
    except Exception as e:
        st.error(f"Error al predecir: {e}")
        st.write('Columnas de entrada que se entregaron al modelo:')
        st.write(list(X_input.columns))

st.markdown("---")
st.info('Asegúrate de que los archivos Joblib (modelo, encoder y scaler) estén en la carpeta Joblib.')