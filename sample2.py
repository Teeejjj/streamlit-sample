import pandas as pd
import numpy as np
from database import DBConnect
DBConnect = DBConnect()
import joblib
import streamlit as st
import warnings
import io
import plotly.express as px
import time
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

    # if 'gender' in df.columns:
    #     gen_col = 'gender'
    # elif 'Sex' in df.columns:
    #     gen_col = 'Sex'
    # else:
    #     st.error("Gender/Sex Column does not exist!")
    # st.session_state.gen_col = gen_col
    # null_gender_rows = df[gen_col].isnull()

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

@st.dialog("Gender Prediction")
def for_funsies(model_file='C://Users//User//Documents//work//Database//modeling//model v.3//XGBoostModel_91%.pkl'):
    text = st.text_input("Enter a Name:", placeholder="Any Name")
    with open(model_file, 'rb') as file:
        loaded_model = joblib.load(file)

    model = loaded_model['model']
    cv2 = loaded_model['count_vectorizer']
    if text:
        try:
            prog_bar = st.progress(0, text="Analyzing Name . . .")
            for comp in range(100):
                time.sleep(0.01)
                prog_bar.progress(comp + 1, text= f"Predicting Name. . . {comp + 1}%")
            text_transform = cv2.transform([text])
            pred_text = model.predict(text_transform)
            gend_map = {0:'Female', 1:'Male'}
            pred_gend = gend_map.get(pred_text[0], None)
            prog_bar.empty()
            st.markdown(f'The predicted gender for {text} is <code><b>{pred_gend}</b></code>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Predition Erro:{e}")

def authenticate_user():
    def database_conn():
        DBConnect.db_connect()
        # DBConnect.cnx_79
        db_df = DBConnect.users_conn()
        return db_df
    
    db_df = database_conn()
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.header("Enter your Account")
        username = st.text_input("Username:")
        password = st.text_input("Password:", type="password")

        if st.button("Login 🙂‍↕️"):
            if not username:
                st.warning("Please enter username")
            elif not password:
                st.warning("Please enter password")
            else:
                valid_user = not db_df[
                    (db_df["username"] == username) & 
                    (db_df["pass"] == password)
                ].empty
                if valid_user:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid Username/Password")
        return False
    return True

#######################_____________MAIN WINDOW_____________#######################
def main():
    if not authenticate_user():
        # st.warning("Please log in to continue.")
        return
    
    #### -~-~ Creating our session states -~-~ ####
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    if 'last_uploaded' not in st.session_state:
        st.session_state.last_uploaded = None

    st.title("Gender Prediction 🧔‍♂️👩‍🦰")
    st.info("This app tries to predict a person's gender using machine learning")
    
    upload_file = st.file_uploader("Upload Excel File", key='upload_file')

    #### -~-~ Creating our session states -~-~ ####
    if upload_file is None and st.session_state.get('data_source') == 'upload':
        st.session_state.update({
            'data_source': None,
            'current_df': None,
            'last_uploaded': None
        })
        st.rerun()

    # ---- Getting our Data via Upload ---- #
    if upload_file:
        if upload_file != st.session_state.last_uploaded:
            st.session_state.data_source = 'upload' # New file detected - reset state
            st.session_state.last_uploaded = upload_file
            
            try:
                if upload_file.name.endswith('.csv'):                               #     TRY
                    st.session_state.current_df = pd.read_csv(upload_file)          #
                elif upload_file.name.endswith(('.xlsx', '.xls')):                  #
                    st.session_state.current_df = pd.read_excel(upload_file)        #               CONDITIONING :) BETTER ERROR HANDLING
                else:                                                               #
                    st.error("Unsupported file format")                             #
                    st.session_state.current_df = None                              #
            except Exception as e:                                                  #     CATCH
                st.error(f"Error reading file: {str(e)}")                           #
                st.session_state.current_df = None                                  #      this is one complicated blocks of comment

# ---- Sidebar ---- #
    with st.sidebar:
        with st.expander("Try the model out!", icon="👶"):
            if st.button("➡️"):
                for_funsies()
        st.header('Gender Visualizations')
        gender_cat = st.selectbox('Gender Count:', ('Male', 'Female', 'All'))
        if st.session_state.current_df is not None and 'gender' in st.session_state.current_df.columns:
            df = st.session_state.current_df
            
            gender_counts = df['gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            
            color_map = {'M': '#1f77b4', 'F': '#ff69b4'}  # Blue for Male, Pink for Female
            
            if gender_cat == 'Male':
                filtered_counts = gender_counts[gender_counts['Gender'] == 'M']
            elif gender_cat == 'Female':
                filtered_counts = gender_counts[gender_counts['Gender'] == 'F']
            else:
                filtered_counts = gender_counts
            
            ## Plotly Visualization ##
            fig = px.bar(
                filtered_counts,
                x='Gender',
                y='Count',
                color='Gender',
                color_discrete_map=color_map,
                title=f"Gender Distribution - {gender_cat}",
                text_auto=True
            )
            
            fig.update_layout(
                xaxis_title=None,
                yaxis_title="Count",
                showlegend=False,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Upload data to see visualizations")
    
        if st.button('Log out 📤'):
            # return to Log in / def authenticate_user
            st.session_state.clear()
            st.rerun()
        if st.button("Try this sampled data! 👍", use_container_width=True):
            st.session_state.data_source = 'sample'
            st.session_state.current_df = pd.DataFrame({"Name":["Josh", "Kristian", "Shiela", "Mark", "Jenny", "Delro", "Chin"],
                                    "gender": [None, None, None, None, None, None, None]})
            
            st.session_state.last_uploaded = None  # This will clear the upload history
            st.rerun()

    # ---- Dataframes ---- #
    dataframe = st.session_state.get('current_df')
    predicted_df = None

    # ------ expander for uploaded data ------- #        
    with st.expander('Uploaded Data'):
        if dataframe is not None:
            st.dataframe(dataframe, use_container_width=True, height=300)
        else:
            st.image("https://media.istockphoto.com/id/637743724/vector/dont-know-emoticon.jpg?s=1024x1024&w=is&k=20&c=7OfBxiNChgyDEEN7Dq_6wNB66LFNZO2E52djPvgSWHw=")
        
    if dataframe is not None:
        if st.button('Predict'):
            if dataframe is not None:
                try:
                    predicted_df = predict(dataframe.copy(deep=True))
                    st.session_state.current_df = predicted_df
                    st.success("Predicted Values")
                    # st.dataframe(predicted_df, use_container_width=True, height=300)  # Display updated data
                    # dataframe = predicted_df
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Please upload a file first.")

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

#######################_____________INITIALIZE APP_____________#######################
if __name__ =='__main__':
    buffer = io.BytesIO()
    main()

# # # # # # --ALL COMMENTS ARE A REMINDER BECAUSE IM A FORGTFUL BITHC-- # # # # # #