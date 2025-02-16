import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Load the dataset
file_path = "dataset.csv"
data = pd.read_csv(file_path)
text = "\n".join(data["Poetry"].dropna().tolist())

# Tokenize text with newlines included
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts([text])
vocab_size = len(tokenizer.word_index) + 1

# Load the trained model
model_save_path = "roman_urdu_poetry_model.keras"
model = tf.keras.models.load_model(model_save_path)

# Determine the maximum verse length
max_verse_length = data['Poetry'].dropna().apply(
    lambda poem: max([len(verse.strip().split()) for verse in poem.split('\n')], default=0)
).max()
seq_length = max_verse_length

# Text generation function
def generate_text(seed_text, num_generate=50, temperature=1.0):
    generated_text = seed_text
    for _ in range(num_generate):
        token_list = tokenizer.texts_to_sequences([generated_text])[0][-seq_length:]
        token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')
        predictions = model.predict(token_list, verbose=0)
        predictions = predictions / temperature
        predicted_id = tf.argmax(predictions, axis=-1).numpy()[0]
        if predicted_id == 0:
            continue
        predicted_word = tokenizer.index_word.get(predicted_id, '')
        if predicted_word == "\n":
            generated_text += "\n"
        else:
            generated_text += " " + predicted_word
    return generated_text

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Seed Text", placeholder="Enter a seed word for poetry"),
        gr.Slider(minimum=10, maximum=200, step=10, value=50, label="Number of Words"),
        gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Poetry"),
    title="Roman Urdu Poetry Generator",
    description="Enter a seed word and generate poetry in Roman Urdu with newline support."
)

iface.launch()
