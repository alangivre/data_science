
# Function to make predictions
def predict_diarrheal_diseases(pca0, meningitis):
    # Make prediction
    prediction = lm_ols_2.predict({"pca0": pca0, "Meningitis": meningitis})
    return prediction.iloc[0]

# Streamlit app
st.title("Predict Diarrheal Diseases using OLS Model")
st.write("Enter the following details:")

# Collect user input
pca0 = st.number_input("PCA0 Value", min_value=data_inclusivo['pca0'].min(), max_value=data_inclusivo['pca0'].max(), value=np.median(data_inclusivo['pca0']))
meningitis = st.number_input("Meningitis Value", min_value=data_inclusivo['Meningitis'].min(), max_value=data_inclusivo['Meningitis'].max(), value=np.median(data_inclusivo['Meningitis']))

# Make prediction
if st.button("Predict"):
    prediction = predict_diarrheal_diseases(pca0, meningitis)
    st.write(f"Predicted Diarrheal Diseases: {prediction}")
