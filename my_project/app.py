import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("😊 Sentiment Analysis App")
st.write("Type a sentence below to check if the sentiment is Positive or Negative!")

# Text input box for the user
user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Transform the input text using the trained vectorizer
        input_vectorized = vectorizer.transform([user_input])
        # Predict the sentiment
        prediction = model.predict(input_vectorized)[0]
        
        # Display the result
        if prediction == "positive":
            st.success(f"Prediction: **{prediction.upper()}** 😊")
        else:
            st.error(f"Prediction: **{prediction.upper()}** 😞")