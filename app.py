import streamlit as st
import pandas as pd
import pickle


st.title('SteamFlow Prediction APP for a Papermil ')
from PIL import Image
image = Image.open('papermil1000.jpg')
st.image(image, caption='Paper mill- Paper production (Own image)', width=400)

st.write("""
# Description

This app predicts the **SteamFlow** of a PaperRoll in PaperMill!

During the production of a paper roll in the paper mill, it takes around **17-20** tons of water/hour(Depends on the size of the roll).
There are 403 parameters nobs are all around the paper machine, which measures different parameters during the production. Among them 23 of them are
more important that has good effect on the water uses(SteamFlow) parapeter.

With this app production manager can check how much water will be used to produce a paper roll and set the parameters to those 23 nobs.

#  Tools used:

We have used Random Forest regressor to make the model with hyperparameter tuning. 

Data we had collected from a paper mill open data. The data consists of 3941 records with 24 features. 

Used scikit learn.

""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        starch_mixing_tank23top_ply = st.sidebar.slider('Starch mixing in TANK23 top Ply', 21.1,61.9,34.2,0.2)
        pulp_to_mixing_tank23 = st.sidebar.slider('Pulp to mixing TANK23', 131.0,223.0,175.0,0.2)
        power_power_station24 = st.sidebar.slider('Power of POWERSTATION24', 13.0,342.0,250.0,0.2)
                   
              
        recycled_liquid_top_ply = st.sidebar.slider('Recycled liquid of Top_ply', 0.0,2.0,1.0,0.1)
        sum_bleached_pulp = st.sidebar.slider('Sum of Bleached Pulp', 6.0,10.0,7.0,0.2)
        flow_after_machine_chest21 = st.sidebar.slider('Flow after Machine_Chest21', 2148.0,3700.0,2888.0,0.5)
        steam_pressure_group1 = st.sidebar.slider('Steam pressure of group1', 6.0,104.0,80.0,0.2)
        steam_pressure_group2 = st.sidebar.slider('Steam pressure of group2', 39.0,176.0,156.0,0.2)
        level_condensate_bucket1 = st.sidebar.slider('level Condensate Bucket1', 26.0,531.0,330.0,0.2)
        level_condensate_bucket1_valve_position = st.sidebar.slider('level Condensate_bucket1_valve_position', 26.0,531.0,330.0,0.2)  
        level_condensate_bucket2_valve_position = st.sidebar.slider('level condensate_bucket2_valve_position', 26.0,531.0,330.0,0.2) 
        steam_group3_pressure = st.sidebar.slider('Steam group3 pressure', 26.0,531.0,330.0,0.2)                
        steam_group4under_pressure = st.sidebar.slider('Steam group4 under_pressure', 26.0,531.0,330.0,0.2)                
        steam_pressure4_over = st.sidebar.slider('Steam pressure4 over', 142.0,294.0,238.0,0.2)
        pressure_yankee_cylinder = st.sidebar.slider('Pressure of yankee cylinder', 143.0,295.0,237.0,0.2)
        steam_group5under_pressure = st.sidebar.slider('Steam Group5under Pressure', 143.0,287.0,237.0,0.2)
        steam_pressure5_over = st.sidebar.slider('Steam pressure5 over', 143.0,290.0,237.0,0.2)
        pressure_condensate_bucket2 = st.sidebar.slider('Pressure_condensate_bucket2', 39.0,188.0,158.0,0.2)
        pressure_condensate_bucket5 = st.sidebar.slider('Pressure_condensate_bucket5', 2.0,100.0,75.0,0.2)
        production_paper_machine2 = st.sidebar.slider('Production PAPERMACHINE2', 0.0,14.0,10.0,0.2)
        dry_production_vira = st.sidebar.slider('Dry production Vira', 115.0,205.0,160.0,0.2)
        total_starch_paper_machine2 = st.sidebar.slider('Total starch of PAPERMACHINE2', 37.0,95.0,57.0,0.2)
       
       
        
        data = {'starch_mixing_tank23top_ply': starch_mixing_tank23top_ply,
                'pulp_to_mixing_tank23': pulp_to_mixing_tank23,
                'power_power_station24': power_power_station24,
                
                'recycled_liquid_top_ply': recycled_liquid_top_ply,
                'sum_bleached_pulp': sum_bleached_pulp,
                'flow_after_machine_chest21': flow_after_machine_chest21,
                'steam_pressure_group1': steam_pressure_group1,
                'steam_pressure_group2': steam_pressure_group2,
                'level_condensate_bucket1': level_condensate_bucket1,
                'steam_pressure4_over': steam_pressure4_over,
                'pressure_yankee_cylinder': pressure_yankee_cylinder,
                'steam_group5under_pressure': steam_group5under_pressure,
                'steam_pressure5_over': steam_pressure5_over,
                'pressure_condensate_bucket2': pressure_condensate_bucket2,
                'pressure_condensate_bucket5': pressure_condensate_bucket5,
                'production_paper_machine2': production_paper_machine2,
                'dry_production_vira': dry_production_vira,
                'total_starch_paper_machine2': total_starch_paper_machine2,
                'level_condensate_bucket1_valve_position': level_condensate_bucket1_valve_position,
                'level_condensate_bucket2_valve_position': level_condensate_bucket2_valve_position,
                'steam_group3_pressure': steam_group3_pressure,
                'steam_group4under_pressure': steam_group4under_pressure
                
                
                
               }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
# =============================================================================
# papermil_raw = pd.read_csv('papermil_water_flow.csv')
# papermil = papermil_raw.drop(columns=['steamflow'])
# df = pd.concat([input_df,papermil],axis=0)
# =============================================================================


# Displays the user input features
#st.subheader('User Input features')

# =============================================================================
# if uploaded_file is not None:
#     st.write(input_df)
# else:
#     st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
#     st.write(input_df)
# =============================================================================

# Reads in saved classification model
load_rf_model = pickle.load(open('papermil_rf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_rf_model.predict(input_df)
#prediction_proba = load_rf_model.predict_proba(df)


st.write("""
# Prediction of SteamFlow(tons/hour)
""")
st.write(prediction)

#st.subheader('Prediction Probability')
#st.write(prediction_proba)



st.subheader('Model and UI developed by : Ricky D Cruze')
