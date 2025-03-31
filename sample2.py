import pandas as pd
import numpy as np

import joblib
import streamlit as st
import warnings
import io
warnings.filterwarnings('ignore')

@st.cache_resource
def predict(df, model_file='C://Users//User//Documents//work//Database//modeling//model v.3//XGBoostModel_91%.pkl'):
    with open(model_file, 'rb') as file:
        loaded_model = joblib.load(file)

    model = loaded_model['model']
    cv2 = loaded_model['count_vectorizer']
    
    if 'gender' not in df.columns:
        df['gender'] = np.nan
    null_gender_rows = df['gender'].isnull()

    if 'Name' in df.columns:
        name = 'Name'
    elif 'FirstName' in df.columns:
        name = 'FirstName'
    else:
        st.exception(RuntimeError("No Name Column Existing"))

    if null_gender_rows.any():
        x_new_transformed = cv2.transform(df.loc[null_gender_rows, str(name)].fillna(''))
        y_pred = model.predict(x_new_transformed)
        gender_map = {'F': 0, 'M': 1}
        inv_gender_map = {v: k for k, v in gender_map.items()}
        y_pred_labels = [inv_gender_map.get(y, y) for y in y_pred]
        df.loc[null_gender_rows, 'gender'] = y_pred_labels

    return df

def saving_excel(buffer, df):
        with pd.ExcelWriter(buffer,engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
        buffer.seek(0)

        st.download_button(
            label="Download Excel File",
            data= buffer,
            file_name = "pred_val.xlsx",
            mime = "application/vnd.ms_excel",
            on_click='ignore'
        )


# st.dialog("Log in to Account!")
# def log_in():
#     st.)
def creds_entered():
    if st.session_state["user"].strip() == "admin" and st.session_state['passwd'].strip() == 'admin':
        st.session_state['authenticated'] = True
    else:
        st.session_state['authenticated'] = False
        st.error('Invalid Username/Password . . . ')
    
def authenticate_user():
    if "authenticated" not in st.session_state:
        st.text_input(label="Username: ", value="", key="user", on_change=creds_entered)
        st.text_input(label="Password: ", value="", key="pswswd", type="password", on_change=creds_entered)
        return False
    else:
        if st.session_state['authenticated']:
            return True
        else:
            st.text_input(label="Username: ", value="", key="user", on_change=creds_entered)
            st.text_input(label="Password: ", value="", key="psswd", type="password", on_change=creds_entered)
            return False
def main():
    if not authenticate_user():
        st.warning("Please log in to continue.")
        return

    st.title("Gender Prediction 🧔‍♂️👩‍🦰")
    st.info("This app tries to predict a person's gender using machine learning")
    
    upload_file = st.file_uploader("Upload Excel File")
    dataframe = None
    predicted_df = None  

    if upload_file is not None:
        if upload_file.name.endswith('.csv'):
            dataframe = pd.read_csv(upload_file)
        elif upload_file.name.endswith('.xlsx'):
            dataframe = pd.read_excel(upload_file)
    with st.expander('Uploaded Data'):
        st.dataframe(dataframe, use_container_width=True, height=300)
        
    if upload_file is not None:
        if st.button('Predict'):
            if dataframe is not None:
                try:
                    predicted_df = predict(dataframe)
                    st.success("Predicted Values")
                    # st.dataframe(predicted_df, use_container_width=True, height=300)  # Display updated data
                    # dataframe = predicted_df
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Please upload a file first.")
    with st.sidebar:
        st.header('Gender Visualizations')
        gender_cat = st.selectbox('Gender Count:', ('Male', 'Female', 'All'))
        

    tab1, tab2 = st.tabs(['Null Gender', 'Predicted Gender'])

    tab1.subheader("Rows with Missing Gender")
    if dataframe is not None:
        tab1.dataframe(dataframe[dataframe['gender'].isnull()], use_container_width=True,height=300)

    tab2.subheader("Rows with Predicted Gender")
    if predicted_df is not None:
        tab2.dataframe(predicted_df[predicted_df['gender'].notnull()], use_container_width=True,height=300)
    elif dataframe is not None:
        tab2.dataframe(dataframe[dataframe['gender'].notnull()],use_container_width=True, height=300)

    if predicted_df is not None:
       saving_excel(buffer, predicted_df)


if __name__ =='__main__':
    buffer = io.BytesIO()
    main()