# from xml.parsers.expat import model
# import joblib
# import streamlit as st    
# from constants import DataManager
# import transform_page as transform_page 
# import pandas as pd
# import numpy as np
# from models import models_execution
# data_manager = DataManager()
# import machine
# from models import  model_components
# from transformations import transformation_execution
# import joblib, base64, json
# from io import BytesIO
# import pickle
# from reporting_lib import model_reporting_setup

# def show_model_page(edit,model_data,model_name,data_uploaded,reset):
#             if 'show_ml_dialogue' not in st.session_state:
#                 st.session_state.show_ml_dialogue= False
#             """Detailed model implementation page"""
            
#             title = f"{model_data['model name'] if edit else model_name} Implementation"
#             st.markdown(data_manager.label_style, unsafe_allow_html=True)
#             st.markdown("""
#             <div class="custom-container">
#                 <h2 class="custom-header">🚀"""+title+"""</h2>
#             </div>
#             """, unsafe_allow_html=True)
#             if reset:
#                 # if 'selected_data' in st.session_state:
#                 #     del st.session_state['selected_data']
#                 # if 'selected_trans' in st.session_state:
#                 #      del st.session_state['selected_trans']
#                 if 'model_data' in st.session_state:
#                     del st.session_state['model_data']
#                 # if 'comments' in st.session_state:
#                 #     del st.session_state['comments']
#             # INITIALIZATION
     
 
#             # if 'selected_trans' in st.session_state:
#             #     del st.session_state['selected_trans']
#                 # st.write(st.session_state.selected_data)
#             if 'selected_trans' not in st.session_state :
#                 st.session_state.selected_trans = []
#             if data_uploaded:
#                 st.write(edit)
#                 if edit :
                    
#                     # st.session_state.comments=model_data['comments']
#                     if 'selected_trans' not in st.session_state :
#                         st.session_state.selected_trans = model_data['transformations']
#                     if 'selected_data' not in st.session_state  :
#                         # st.session_state.selected_data = transform.apply_selected_transformations(model_data['transformations'])
#                         st.session_state.selected_data = st.session_state.df_original.copy()
#                         pipeline_steps = st.session_state.pipeline['transformations']
#                         # Create quick lookup dictionary
#                         pipeline_dict = {step['name']: step for step in pipeline_steps}
#                         for step_name in model_data['transformations']:
#                             step = pipeline_dict.get(step_name)

#                             if step:
#                                 st.session_state.selected_data = transformation_execution.execute_transformation(
#                                     step['type'],
#                                     "execution",
#                                     {
#                                         "df": st.session_state.selected_data,
#                                         "step": step
#                                     }
#                                 )
#                         # st.dataframe(st.session_state.selected_data)
#                     default_trans =  model_data['transformations']
#                 else :
#                      if 'selected_data' not in st.session_state :
#                         st.session_state.selected_data = st.session_state.df_original
#                     # else : st.session_state.selected_trans = []
#                      default_trans =  []
#             else : 
#                 st.session_state.selected_data = pd.DataFrame()
#                 # st.session_state.selected_trans = []
#                 default_trans = []
#             if 'show_data_transformations' not in st.session_state:
#                 st.session_state.show_data_transformations = False
#             # if st.session_state.current_page == 'main':
#             #    print('main')
#             #    del  st.session_state['model_data']
#             #    del st.session_state['selected_data']
#             # Back Button
#             back_col , download_Col = st.columns([5,1])
#             with back_col:
#                 if st.button("Back to Main"):
#                     st.session_state.current_page = "main"
#                     if 'selected_data' in st.session_state:
#                         del st.session_state['selected_data']
#                     if 'selected_trans' in st.session_state:
#                         del st.session_state['selected_trans']
#                     st.rerun()
#             with download_Col:
#                 if st.button("Download Model",key="download_model"):
#                     # st.write(st.session_state.model_results['model'])
#                     if edit and 'model' in model_data:
#                         model_b64 = model_data['model']
#                         # st.session_state.model_results['model']
#                         model_bytes = pickle.dumps(model_b64)
#                     st.download_button(
#                         label="Download "+ model_data['model name'],
#                         data=model_bytes,
#                         file_name=model_data['model name'] + ".pkl",
#                         mime="application/octet-stream"
#                     )
# # ####################################################################
#             #  MODEL DESCRIPTION
#             st.markdown(models_execution.get_model_data(model_name,"model_description") , unsafe_allow_html=True)
#             #  MODEL REFERENCE CODE 
#             with st.expander("reference code"):
#                 st.markdown(
#                     models_execution.get_model_data(model_name,"model_reference") , unsafe_allow_html=True)
# ######################################################################
#             # Data transformations dialog
#             if st.button("Choose Your Data Transformations"):
#                 st.session_state.show_data_transformations = not st.session_state.show_data_transformations
#                 st.write(st.session_state.show_data_transformations,"show")

#             if st.session_state.show_data_transformations:
            
#                 with st.expander("Data Transformation Options", expanded=True):
#                     trans =st.session_state.pipeline['transformations']
#                     selected_trans = st.multiselect(
#                     "Select columns to include:",
#                     options=[step["name"] for step in trans]if data_uploaded else [],
#                     default= default_trans,
#                     key="data_transform_columns2" 
                    
#                     ) 
#                     # st.write(st.session_state.selected_data)
                    
#                     if not st.session_state.selected_data.empty:
#                         st.session_state.temp_df = st.session_state.df_original.copy()
#                         if selected_trans:
#                             for step in st.session_state.pipeline['transformations']:
#                                 if step['name'] in selected_trans:
#                             # Apply selected transformations     
#                                     st.write(step['type'],"step type")
#                                     st.session_state.temp_df = transformation_execution.execute_transformation(step['type'],"execution",{"df":st.session_state.temp_df,"step":step})
#                         st.dataframe(st.session_state.temp_df)
#                     else :
#                         # st.write("else")
#                         st.error('Please Load you data first', icon="🚨")
       

#                     if st.button("Apply Transformations", key="apply_transformations"):
#                         st.write(selected_trans)
#                         st.session_state.selected_trans = selected_trans 
#                         st.session_state.selected_data =st.session_state.temp_df
#                         st.success("Data transformations applied successfully!")
#                         # st.session_state.show_data_transformations = False
#                         st.rerun()
                        
            
#             # try:
#             # Model configuration section
#             st.markdown(data_manager.label_style, unsafe_allow_html=True)
#             st.markdown("""
#             <div class="custom-container">
#                 <h2 class="custom-header">🚀 Model configuration</h2>
#             </div>
#             """, unsafe_allow_html=True)
#             config_params ={'model_data':model_data,"edit":edit}
#             validate_params = {"params":models_execution.execute_model(model_name,"config",config_params)}
#             script_params=validate_params["params"]
#             # st.write(script_params)
#             col3,col4 = st.columns([1,1])
#             with col3:
#                 if st.button("Create Model"if not edit else "Update Model",):
#                     if models_execution.execute_model(model_name,"validate",validate_params)  :
#                         ## Model Execution 
#                         # params = {"df":st.session_state.selected_data,"features":features,"target":target,"edit":edit}
#                         st.session_state.model_results = models_execution.execute_model(model_name,"script",script_params)
#             if 'model_results' in st.session_state or edit:      
#             #     x_test =st.text_input("Input your values separated by commas", key="input_values")  
#                 # if st.button("Predict results", key="predict_results"):
#             #         try:
#             #             if x_test:
#             #                 x_test = np.array([float(x.strip()) for x in x_test.split(',')]).reshape(1, -1)
#             #                 result = return_model(model_data['model']).predict(x_test)
    
#             #                 st.write(result)
#             #         except ValueError:
#             #             st.error("Please enter valid numeric values separated by commas.")
#                 prediction_setup(model_data['model'])
#             if st.button("prepare model report", key="prepare_model_report"):
#                 st.session_state.show_ml_dialogue= True
#             if st.session_state.show_ml_dialogue:
#                 if 'model_results' in st.session_state or edit:
#                         st.success("Model report is ready!")
#                         st.session_state.show_ml_dialogue=  True
#                         st.session_state.model_results = model_data
#                         # st.session_state.model_results if  'model_results' in st.session_state else 
#                         model_index=0
#                         for index, model_info in enumerate(st.session_state.pipeline["ML"]):
#                             if model_info.get("model name") == model_name:
#                                 model_index = index
#                                 break
#                         model_reporting_setup.display_ml_reporting_ui(model_name,model_index)
#             if edit: 
#                 with col4:
#                     if st.button("Delete Model") :
#                         machine.delete_model(model_name=model_data['model name'])
#                         st.success('Model deleted')
                
#             # Show report if model was created
#             if 'model_data' not in st.session_state and edit:
#                 st.session_state.model_data = model_data
                
#             # if 'model_results' in st.session_state:
#             if st.button("Show Report", key="show_report_button"):
#                 st.markdown("""  <style>
#                     .metric-card {
#                     background-color: #0F1533; /* set background color */
#                     }
#                 </style>""", 
#                 unsafe_allow_html=True)
#                 models_execution.execute_model(model_name,"report",{})
#                     # model_report()
#         ## Here is the comments section 
#         ## same with every model
#             model_components.comments_section(edit)
#             # except Exception as e:
#             #     st.error(f"An error occurred: {e}")
#             #     st.stop()

# # def return_model(model_b64):
# #     model_bytes = base64.b64decode(model_b64)
# #     buffer = BytesIO(model_bytes)
# # # Load model from memory
# #     model = joblib.load(buffer)
# #     return model

# def prediction_setup(model):
#     with st.expander("Model Prediction Setup", expanded=True):
#         uploaded_file = st.file_uploader("upload prediction data here", type="csv")
#         if uploaded_file is not None:
#             try:
#                 st.session_state.pred_df = pd.read_csv(uploaded_file)
#                 st.success("Prediction data loaded successfully!")
#             except Exception as e:
#                 st.error(f"Error loading prediction data: {e}")
#                 return
            
#         pred_cols = st.multiselect(
#             "Select columns for prediction:",
#             options=st.session_state.df_original.columns.tolist(),
#             key="prediction_columns"
#         )
#         trans =st.session_state.pipeline['transformations']
#         selected_trans = st.multiselect(
#                     "Select transformations to include:",
#                     options=[step["name"] for step in trans],
#                     key="data_transform_columns" 
#                     ) 
#         if selected_trans:
#                 for step in st.session_state.pipeline['transformations']:
#                     if step['name'] in selected_trans:
#                 # Apply selected transformations     
#                         st.write(step['type'],"step type")
#                         st.session_state.pred_df = transformation_execution.execute_transformation(step['type'],"execution",{"df":st.session_state.pred_df,"step":step})
#         if uploaded_file is not None:
#             st.dataframe(st.session_state.pred_df)
#         if st.button("Predict results", key="predict_results_button"):
#             try:
#                 if st.session_state.pred_df is not None:
#                     result = model.predict(st.session_state.pred_df[pred_cols])
#                     st.session_state.pred_df['Prediction'] = result
#                     st.dataframe(st.session_state.pred_df)

#             except ValueError:
#                 st.error("Please enter valid numeric values separated by commas.")

from xml.parsers.expat import model
import joblib
import streamlit as st    
from constants import DataManager
import transform_page as transform_page 
import pandas as pd
import numpy as np
from models import models_execution
data_manager = DataManager()
import machine
from models import  model_components
from transformations import transformation_execution
import joblib, base64, json
from io import BytesIO
import pickle
from reporting_lib import model_reporting_setup

def show_model_page(edit,model_data,model_name,data_uploaded,reset):
            if 'show_ml_dialogue' not in st.session_state:
                st.session_state.show_ml_dialogue= False
            """Detailed model implementation page"""
            
            title = f"{model_data['model name'] if edit else model_name} Implementation"
            st.markdown(data_manager.label_style, unsafe_allow_html=True)
            st.markdown("""
            <div class="custom-container">
                <h2 class="custom-header">🚀"""+title+"""</h2>
            </div>
            """, unsafe_allow_html=True)
            if reset:
                # if 'selected_data' in st.session_state:
                #     del st.session_state['selected_data']
                # if 'selected_trans' in st.session_state:
                #      del st.session_state['selected_trans']
                if 'model_data' in st.session_state:
                    del st.session_state['model_data']
                # if 'comments' in st.session_state:
                #     del st.session_state['comments']
            # INITIALIZATION
     
 
            # if 'selected_trans' in st.session_state:
            #     del st.session_state['selected_trans']
                # st.write(st.session_state.selected_data)
            if 'selected_trans' not in st.session_state :
                st.session_state.selected_trans = []
            if data_uploaded:
                st.write(edit)
                if edit :
                    
                    # st.session_state.comments=model_data['comments']
                    if 'selected_trans' not in st.session_state :
                        st.session_state.selected_trans = model_data['transformations']
                    if 'selected_data' not in st.session_state  :
                        # st.session_state.selected_data = transform.apply_selected_transformations(model_data['transformations'])
                        st.session_state.selected_data = st.session_state.df_original.copy()
                        pipeline_steps = st.session_state.pipeline['transformations']
                        # Create quick lookup dictionary
                        pipeline_dict = {step['name']: step for step in pipeline_steps}
                        for step_name in model_data['transformations']:
                            step = pipeline_dict.get(step_name)

                            if step:
                                st.session_state.selected_data = transformation_execution.execute_transformation(
                                    step['type'],
                                    "execution",
                                    {
                                        "df": st.session_state.selected_data,
                                        "step": step
                                    }
                                )
                        # st.dataframe(st.session_state.selected_data)
                    default_trans =  model_data['transformations']
                else :
                     if 'selected_data' not in st.session_state :
                        st.session_state.selected_data = st.session_state.df_original
                    # else : st.session_state.selected_trans = []
                     default_trans =  []
            else : 
                st.session_state.selected_data = pd.DataFrame()
                # st.session_state.selected_trans = []
                default_trans = []
            if 'show_data_transformations' not in st.session_state:
                st.session_state.show_data_transformations = False
            # if st.session_state.current_page == 'main':
            #    print('main')
            #    del  st.session_state['model_data']
            #    del st.session_state['selected_data']
            # Back Button
            back_col , download_Col = st.columns([5,1])
            with back_col:
                if st.button("Back to Main"):
                    st.session_state.current_page = "main"
                    if 'selected_data' in st.session_state:
                        del st.session_state['selected_data']
                    if 'selected_trans' in st.session_state:
                        del st.session_state['selected_trans']
                    st.rerun()
            with download_Col:
                if st.button("Download Model",key="download_model"):
                    # st.write(st.session_state.model_results['model'])
                    if edit and 'model' in model_data:
                        model_b64 = model_data['model']
                        # st.session_state.model_results['model']
                        model_bytes = pickle.dumps(model_b64)
                    st.download_button(
                        label="Download "+ model_data['model name'],
                        data=model_bytes,
                        file_name=model_data['model name'] + ".pkl",
                        mime="application/octet-stream"
                    )
# ####################################################################
            #  MODEL DESCRIPTION
            st.markdown(models_execution.get_model_data(model_name,"model_description") , unsafe_allow_html=True)
            #  MODEL REFERENCE CODE 
            with st.expander("reference code"):
                st.markdown(
                    models_execution.get_model_data(model_name,"model_reference") , unsafe_allow_html=True)
######################################################################
            # Data transformations dialog
            if st.button("Choose Your Data Transformations"):
                st.session_state.show_data_transformations = not st.session_state.show_data_transformations
                st.write(st.session_state.show_data_transformations,"show")

            if st.session_state.show_data_transformations:
            
                with st.expander("Data Transformation Options", expanded=True):
                    trans =st.session_state.pipeline['transformations']
                    selected_trans = st.multiselect(
                    "Select columns to include:",
                    options=[step["name"] for step in trans]if data_uploaded else [],
                    default= default_trans,
                    key="data_transform_columns2" 
                    
                    ) 
                    # st.write(st.session_state.selected_data)
                    
                    if not st.session_state.selected_data.empty:
                        st.session_state.temp_df = st.session_state.df_original.copy()
                        if selected_trans:
                            for step in st.session_state.pipeline['transformations']:
                                if step['name'] in selected_trans:
                            # Apply selected transformations     
                                    st.write(step['type'],"step type")
                                    st.session_state.temp_df = transformation_execution.execute_transformation(step['type'],"execution",{"df":st.session_state.temp_df,"step":step})
                        st.dataframe(st.session_state.temp_df)
                    else :
                        # st.write("else")
                        st.error('Please Load you data first', icon="🚨")
       

                    if st.button("Apply Transformations", key="apply_transformations"):
                        st.write(selected_trans)
                        st.session_state.selected_trans = selected_trans 
                        st.session_state.selected_data =st.session_state.temp_df
                        st.success("Data transformations applied successfully!")
                        # st.session_state.show_data_transformations = False
                        st.rerun()
                        
            
            # try:
            # Model configuration section
            st.markdown(data_manager.label_style, unsafe_allow_html=True)
            st.markdown("""
            <div class="custom-container">
                <h2 class="custom-header">🚀 Model configuration</h2>
            </div>
            """, unsafe_allow_html=True)
            try:
                config_params ={'model_data':model_data,"edit":edit}
                validate_params = {"params":models_execution.execute_model(model_name,"config",config_params)}
                script_params=validate_params["params"]
            # st.write(script_params)
            except Exception as e:
                 st.error(f"An error occurred during model configuration: {e}")
            col3,col4 = st.columns([1,1])
            with col3:
                if st.button("Create Model"if not edit else "Update Model",):
                    if models_execution.execute_model(model_name,"validate",validate_params)  :
                        ## Model Execution 
                        # params = {"df":st.session_state.selected_data,"features":features,"target":target,"edit":edit}
                        st.session_state.model_results = models_execution.execute_model(model_name,"script",script_params)

            # --- FIX: resolve the active model object from the freshest available source ---
            # Priority: 1) just-trained result in session_state  2) existing model_data (edit mode)
            # This avoids the one-rerun delay where model_data['model'] doesn't exist yet
            # but model_results already holds the trained model.
            if 'model_results' in st.session_state and 'model' in st.session_state.model_results:
                active_model = st.session_state.model_results['model']
            elif edit and 'model' in model_data:
                active_model = model_data['model']
            else:
                active_model = None

            if active_model is not None:
                prediction_setup(active_model)
            elif 'model_results' in st.session_state or edit:
                st.info("Model object not available yet. Please train or reload the model.")
            # ---------------------------------------------------------------------------------

            if st.button("prepare model report", key="prepare_model_report"):
                st.session_state.show_ml_dialogue= True
            if st.session_state.show_ml_dialogue:
                if 'model_results' in st.session_state or edit:
                        st.success("Model report is ready!")
                        st.session_state.show_ml_dialogue=  True
                        st.session_state.model_results = model_data
                        # st.session_state.model_results if  'model_results' in st.session_state else 
                        model_index=0
                        for index, model_info in enumerate(st.session_state.pipeline["ML"]):
                            if model_info.get("model name") == model_name:
                                model_index = index
                                break
                        model_reporting_setup.display_ml_reporting_ui(model_name,model_index)
            if edit: 
                with col4:
                    if st.button("Delete Model") :
                        machine.delete_model(model_name=model_data['model name'])
                        st.success('Model deleted')
                
            # Show report if model was created
            if 'model_data' not in st.session_state and edit:
                st.session_state.model_data = model_data
                
            # if 'model_results' in st.session_state:
            if st.button("Show Report", key="show_report_button"):
                st.markdown("""  <style>
                    .metric-card {
                    background-color: #0F1533; /* set background color */
                    }
                </style>""", 
                unsafe_allow_html=True)
                models_execution.execute_model(model_name,"report",{})
                    # model_report()
        ## Here is the comments section 
        ## same with every model
            model_components.comments_section(edit)
            # except Exception as e:
            #     st.error(f"An error occurred: {e}")
            #     st.stop()

# def return_model(model_b64):
#     model_bytes = base64.b64decode(model_b64)
#     buffer = BytesIO(model_bytes)
# # Load model from memory
#     model = joblib.load(buffer)
#     return model

def prediction_setup(model):
    with st.expander("Model Prediction Setup", expanded=True):
        uploaded_file = st.file_uploader("upload prediction data here", type="csv")
        if uploaded_file is not None:
            try:
                st.session_state.pred_df = pd.read_csv(uploaded_file)
                st.success("Prediction data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading prediction data: {e}")
                return
            
        pred_cols = st.multiselect(
            "Select columns for prediction:",
            options=st.session_state.df_original.columns.tolist(),
            key="prediction_columns"
        )
        trans =st.session_state.pipeline['transformations']
        selected_trans = st.multiselect(
                    "Select transformations to include:",
                    options=[step["name"] for step in trans],
                    key="data_transform_columns" 
                    ) 
        if selected_trans:
                for step in st.session_state.pipeline['transformations']:
                    if step['name'] in selected_trans:
                # Apply selected transformations     
                        st.write(step['type'],"step type")
                        st.session_state.pred_df = transformation_execution.execute_transformation(step['type'],"execution",{"df":st.session_state.pred_df,"step":step})
        if uploaded_file is not None:
            st.dataframe(st.session_state.pred_df)
        if st.button("Predict results", key="predict_results_button"):
            try:
                if st.session_state.pred_df is not None:
                    result = model.predict(st.session_state.pred_df[pred_cols])
                    st.session_state.pred_df['Prediction'] = result
                    st.dataframe(st.session_state.pred_df)

            except ValueError:
                st.error("Please enter valid numeric values separated by commas.")