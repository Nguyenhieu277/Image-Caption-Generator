import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import nltk
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from tensorflow.keras.applications.densenet201 import DenseNet201, preprocess_input
from tensorflow.keras.models import Model


@st.cache_resource
def load_files():
    with open('trained_model/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    model = load_model('trained_model/caption_imgs.keras')
    with open('trained_model/features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    return tokenizer, model, features

tokenizer, model, features = load_files()

@st.cache_resource
def load_extract_features():
    model = DenseNet201()
    model = Model(inputs = model.input, outputs = model.layers[-2].output)
    return model

feature_extractor = load_extract_features()
def extract_features(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = feature_extractor.predict(img_array)
    return feature.reshape((1, 1920))

def beam_search(model, tokenizer, feature, beam_width = 3, max_length = 34):
    start_seq = "<start>"
    sequences = [(start_seq, 0.0)]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            sequence = tokenizer.texts_to_sequences([seq])[0]
            sequence = pad_sequences([sequence], maxlen = max_length, padding = 'post')

            pred = model.predict([feature, sequence], verbose = 0)[0]

            top_indices = np.argsort(pred)[-beam_width:]

            for word_idx in top_indices:
                word = None
                for w, index in tokenizer.word_index.items():
                    if index == word_idx:
                        word = w
                        break
                
                if word is None or word == "<end>":
                    continue
                
                new_seq = seq + " " + word
                new_score = score + np.log(pred[word_idx])
                all_candidates.append((new_seq, new_score))
        
        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]
    
    best_seq = sequences[0][0]
    return best_seq.replace("<start>", "").replace("<end>", "").strip()

st.title("Image Caption Generator")

uploaded_file = st.file_uploader('Upload an Image', type = ['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_name = uploaded_file.name

    # Check if features exist or extract new ones
    if image_name in features:
        image_feature = features[image_name].reshape((1, 1920))
    else:
        image_feature = extract_features(image)

    # Generate caption
    caption = beam_search(model, tokenizer, image_feature, beam_width=5)
    st.subheader("Generated Caption:")
    st.write(caption)   