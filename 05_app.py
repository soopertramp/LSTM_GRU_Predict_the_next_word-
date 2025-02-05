import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set Streamlit page configuration at the very beginning
st.set_page_config(page_title="LSTM Next Word Predictor", layout="wide")

# Load the LSTM Model
@st.cache_resource
def load_lstm_model():
    return load_model('04_next_word_lstm.h5')

@st.cache_resource
def load_tokenizer():
    with open('03_tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)

model = load_lstm_model()
tokenizer = load_tokenizer()

# Function to predict next words
def predict_next_words(model, tokenizer, text, max_sequence_len, num_words=1):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]

        predicted_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                predicted_word = word
                break
        
        if predicted_word is None:
            return text  # Return original text if no prediction is made
        
        text += " " + predicted_word
    return text

# Streamlit App UI
st.title("🧠 Next Word Prediction with LSTM")
st.markdown("Enter a sequence of words, and the LSTM model will predict the next word(s).")

# Sidebar for instructions and model info
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.write("1. Enter a sentence in the text box below.")
    st.write("2. Choose how many words you want to predict.")
    st.write("3. Click the **Predict** button.")
    st.write("4. The predicted words will appear below.")
    st.markdown("---")
    st.write("🔬 Model: LSTM trained on the The Tragedie of Hamlet by William Shakespeare 1599.")

# User input
input_text = st.text_input("Enter a sequence of words", "To be or not to")
num_words = st.slider("Number of words to predict", 1, 5, 1)

if st.button("🔮 Predict Next Word(s)"):
    max_sequence_len = model.input_shape[1] + 1  # Get max sequence length from the model input shape
    predicted_text = predict_next_words(model, tokenizer, input_text, max_sequence_len, num_words)
    
    if predicted_text == input_text:
        st.warning("⚠️ Unable to generate the next word. Try a different input.")
    else:
        st.success(f"**Predicted Text:** {predicted_text}")
