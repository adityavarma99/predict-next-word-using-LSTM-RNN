import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model = load_model('next_word_lstm.keras')

# Load the tokenizer
with open('lstm_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    # Convert text to token sequence
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Check if the sequence is empty
    if not token_list:
        return "Invalid input. Please enter meaningful text."

    # Adjust the sequence length to match max_sequence_len-1
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # Use only the last (max_sequence_len - 1) tokens
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    # Predict the next word
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    # Retrieve the predicted word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    
    return "Prediction failed. Unable to find the next word."

# Streamlit app
st.title("Next Word Prediction with LSTM (50 Epochs)")

# Input text from the user
input_text = st.text_input("Enter a sequence of words:")

# Predict button
if st.button("Predict Next Word"):
    if input_text.strip():  # Ensure input is not empty
        max_sequence_len = model.input_shape[1] + 1  # Retrieve max sequence length from model
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f"**Next word:** {next_word}")
    else:
        st.write("Please enter a valid sequence of words.")
