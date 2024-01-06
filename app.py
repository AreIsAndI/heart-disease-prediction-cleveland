# import necessary libraries
import streamlit as st
import numpy as np
import joblib as jb
import pandas as pd
import warnings
from numpy.random import randint
warnings.filterwarnings('ignore')

def main():
    # configure the page information
    st.set_page_config(
        page_title= 'Predict your Heart Disease by Irfan!',
        page_icon= ':sparkling_heart:'
    )

    # configure the title
    st.header('Predict your Heart Disease by Irfan! :sparkling_heart:')

    # initialize and load the data, scaler, and model
    loaded_data = pd.read_csv('Include/final-cleveland-data.csv')
    scaler_filename = 'Include/scaler_cleveland.save' 
    knn_filename = 'Include/knn_model_cleveland.sav' 
    rf_filename = 'Include/rf_model_cleveland.sav'
    gbc_filename = 'Include/gbc_model_cleveland.sav'
    knn_loaded_model = jb.load(knn_filename) 
    rf_loaded_model = jb.load(rf_filename) 
    gbc_loaded_model = jb.load(gbc_filename)

    # create two tabs
    tab1, tab2 = st.tabs(['Single Predict', 'Multiple Predict'])

    with tab1:
        # initialize the user inputs
        st.subheader('Input Single Health Status')

        col1, col2 = st.columns([2, 2]) # create two columns
        with col1:
            v_age = st.number_input(label='Age (in years)',
                                    min_value=int(loaded_data['age'].min()),
                                    max_value=int(loaded_data['age'].max()))
            
            sex_dict = {'Female':0, 'Male':1}
            def sex_choice(option):
                return sex_dict[option]

            v_sex = st.selectbox(label='Sex',
                                options=list(sex_dict.keys()))
            
            cp_dict = {'Typical Angina':1, 'Atypical Angina':2, 'Non-anginal Pain':3, 'Asymptomatic':4}
            def cp_choice(option):
                return cp_dict[option]

            v_cp = st.selectbox(label='Chest Pain',
                                options=list(cp_dict.keys()))

            v_trestbps = st.number_input(label='Resting Blood Pressure (in mmHg)',
                                        min_value=loaded_data['trestbps'].min(),
                                        max_value=loaded_data['trestbps'].max())

            v_chol = st.number_input(label='Cholestrol (in mg/dl)',
                                    min_value=loaded_data['chol'].min(),
                                    max_value=loaded_data['chol'].max())
            
            slope_dict = {'Unsloping':1, 'Flat':2, 'Downslaping':3}
            def slope_choice(option):
                return slope_dict[option]

            v_slope = st.selectbox(label='Slope',
                                options=list(slope_dict.keys()))
        
        with col2:
            fbs_dict = {'False':0, 'True':1}
            def fbs_choice(option):
                return fbs_dict[option]

            v_fbs = st.selectbox(label='Fasting Blood Sugar',
                                options=list(fbs_dict.keys()))
            
            restecg_dict = {'Normal':0, 'Having ST-T Wave Abnormality':1, 'Left Ventricular Hypertrophy':2}
            def restecg_choice(option):
                return restecg_dict[option]

            v_restecg = st.selectbox(label='Resting Electrocardiographic Results',
                                options=list(restecg_dict.keys()))

            v_thalach = st.number_input(label='Maximum Heart Rate',
                                        min_value=loaded_data['thalach'].min(),
                                        max_value=loaded_data['thalach'].max())
            
            exang_dict = {'No':0, 'Yes':1}
            def exang_choice(option):
                return exang_dict[option]

            v_exang = st.selectbox(label='Exercise Induced Angina',
                                options=list(exang_dict.keys()))

            v_oldpeak = st.number_input(label='ST Depression Induced by Exercise',
                                        min_value=loaded_data['oldpeak'].min(),
                                        max_value=loaded_data['oldpeak'].max())
            
            ca_dict = {'0':0, '1':1, '2':2, '3':3}
            def ca_choice(option):
                return ca_dict[option]

            v_ca = st.selectbox(label='Number of Major Vessels',
                                options=list(ca_dict.keys()))

        thal_dict = {'Normal':3, 'Fixed Defect':6, 'Reversable Defect':7}
        def thal_choice(option):
            return thal_dict[option]

        v_thal = st.selectbox(label='Thal',
                            options=list(thal_dict.keys()))

        # configure the model prediction
        st.divider()
        st.subheader('Heart Disease Prediction')
        col1, col2 = st.columns([2, 2]) # create two columns
        with col1:
            model_dict = {'K-Nearest Neighbor':knn_loaded_model, 'Random Forest':rf_loaded_model, 'Gradient Boosting':gbc_loaded_model}
            def model_choice(option):
                return model_dict[option]

            v_model = st.selectbox(label='Model',
                                options=list(model_dict.keys()),
                                key='mdl1')
            but_predict = st.button("Predict!", type='primary', key='pred1')

        with col2:
            if but_predict:
                model = model_choice(v_model)
                scaler = jb.load(scaler_filename)
                feature = [v_age, sex_choice(v_sex), cp_choice(v_cp), v_trestbps, v_chol,
                        fbs_choice(v_fbs), restecg_choice(v_restecg), v_thalach, exang_choice(v_exang), v_oldpeak,
                        slope_choice(v_slope), ca_choice(v_ca), thal_choice(v_thal), ]
                feature = np.reshape(feature, (1, -1))
                feature = scaler.transform(feature)

                pred = model.predict([feature[0]])[0]
                if pred == 0:
                    st.header(':green[Healthy!] :sparkling_heart:', anchor=False)
                    st.write('You are free from heart disease! But do not forget to keep your healthy, yes?')
                elif pred == 1:
                    st.header(':orange[Warning!] :heartbeat:', anchor=False)
                    st.write('You have symptoms of heart disease! Contact your doctor for further treatment!')
                else:
                    st.header(':red[Danger!] :broken_heart:', anchor=False)
                    st.write('You have heart disease! Contact your doctor immediately!')
    
    with tab2:
        st.subheader('Input Multiple Health Status')
        # uploaded csv
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])

        col1, col2 = st.columns([2, 2]) # create two columns
        with col1:
            ran_num = randint(len(loaded_data), size=(10))
            sample_csv = loaded_data.iloc[ran_num,:-1].to_csv(index=False).encode('utf-8')
            st.write('See this examples')
            st.download_button('Download the sample.csv', data=sample_csv,
                               file_name='sample.csv', mime='text/csv')

        with col2:
            # choose model
            v_model2 = st.selectbox(label='Model',
                                    options=list(model_dict.keys()),
                                    key='mdl2')
            
            if uploaded:
                st.session_state.disabled = False
            else:
                st.session_state.disabled = True
            but_predict2 = st.button("Predict!", type='primary', key='pred2', disabled=st.session_state.disabled)
        
        if but_predict2:
            st.divider()
            st.subheader('Heart Disease Prediction')
            model2 = model_choice(v_model2)
            scaler2 = jb.load(scaler_filename)
            uploaded_data2 = pd.read_csv(uploaded)
            feature2 = scaler2.transform(uploaded_data2)
            pred2 = model2.predict(feature2)
            
            temp = []
            color = []
            for i in pred2:
                if i == 0:
                    temp.append('Healthy!')
                    color.append('color: green')
                elif i == 1:
                    temp.append('Warning!')
                    color.append('color: orange')
                else:
                    temp.append('Danger!')
                    color.append('color: red')
            
            def color_red_column(col):
                return color
            
            uploaded_data2['Prediction'] = temp
            styled_temp = uploaded_data2.style.apply(color_red_column, subset=['Prediction'])
            st.dataframe(styled_temp.format(precision=0), use_container_width=True)

    # copyright claim
    st.divider()
    st.caption('*Copyright (c) Muhammad Irfan Arisani 2024*')


if __name__ == '__main__':
    main()
