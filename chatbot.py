import tkinter as tk
from tkinter import scrolledtext
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('first_aid_chatbot_model.h5')

# Vocabulary and word-to-index mapping (replace with your actual vocabulary)
vocab = ['burn', 'cut', 'choke', 'heart attack', 'bleeding']
word_to_int = {word: i for i, word in enumerate(vocab)}

# Function to preprocess user input
def preprocess_input(input_text):
    vector = np.zeros(len(vocab))
    for word in input_text.split():
        if word in word_to_int:
            vector[word_to_int[word]] = 1
    return vector.reshape(1, -1)

# Function to get response from the model
def get_response(input_text):
    preprocessed_input = preprocess_input(input_text)
    predictions = model.predict(preprocessed_input)
    predicted_tag_index = np.argmax(predictions)
    # Get the tag associated with the predicted index (replace with your actual tags)
    tags = ['burn', 'cut', 'choke', 'heart_attack', 'bleeding']
    predicted_tag = tags[predicted_tag_index]
    return predicted_tag

# Function to handle sending a message
def send_message(event=None):
    message = entry.get()
    if message.strip() == '':
        return
    conversation.config(state=tk.NORMAL)
    conversation.insert(tk.END, "You: " + message + '\n')
    conversation.see(tk.END)
    entry.delete(0, tk.END)
    response = get_response(message)
    conversation.insert(tk.END, "Chatbot: " + response + '\n')
    conversation.see(tk.END)
    conversation.config(state=tk.DISABLED)

# Create the main window
root = tk.Tk()
root.title("Chatbot UI")

# Create a scrolled text widget to display the conversation
conversation = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=15, state=tk.DISABLED)
conversation.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Create an entry widget for user input
entry = tk.Entry(root, width=30)
entry.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
entry.bind("<Return>", send_message)

# Create a button to send the message
send_button = tk
