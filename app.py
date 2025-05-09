import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import os
os.environ["TORCH_HOME"] = os.getcwd() 
from transformers import BertTokenizer, BertForSequenceClassification

@st.cache_resource
def load_bert_classifier():
    tokenizer = BertTokenizer.from_pretrained(r"Shunmugapriya1612/sports-bert-classifier")
    model = BertForSequenceClassification.from_pretrained(r"Shunmugapriya1612/sports-bert-classifier")
    model.eval()
    return tokenizer, model

# Load BERT text generation pipeline
@st.cache_resource
def load_text_generator():
    generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2", framework="pt")
    return generator
# Preprocessing function
def preprocess(text, tokenizer):
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

def classify(transcript, tokenizer, model, label_encoder):
    inputs = tokenizer(transcript, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_idx = outputs.logits.argmax(dim=1).item()

    print("Type of predicted_idx:", type(predicted_idx))
    print("Value of predicted_idx:", predicted_idx)
    print("Classes:", label_encoder.classes_)

    return label_encoder.inverse_transform([predicted_idx])[0]




# App title
st.title("🎙️ Sports Interview NLP Suite (BERT-AUG)")

# Tabs
tab1, tab2, tab3 = st.tabs(["🧾 Transcript Classification", "🤖 Q&A Generator", "📊 Embedding Explorer"])

# Label encoder (use same order as training)
label_encoder = LabelEncoder()
label_encoder.fit(['pre_game', 'in_game_analysis', 'post_game_reaction', 'injury_update',
                   'player_motivation', 'performance_review', 'team_strategy', 'off_field'])


# Tab 1: Classification
with tab1:
    st.header("Classify Interview Transcript")
    transcript = st.text_area("Paste full interview transcript:")
    if st.button("Classify"):
        tokenizer, model = load_bert_classifier()
        label = classify(transcript, tokenizer, model, label_encoder)
        st.success(f"Predicted Category: **{label}**")

# Tab 2: Q&A Generator
with tab2:
    st.header("Generate AI Interview Response")
    category = st.selectbox("Choose Interview Category", label_encoder.classes_)
    question = st.text_input("Enter question:")
    if st.button("Generate Response"):
        generator = load_text_generator()
        prompt = f"Category: {category}\nQuestion: {question}\nAnswer:"
        response = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        st.markdown("**AI Response:**")
        st.write(response.split("Answer:")[-1].strip())

# Tab 3: UMAP/t-SNE Visualization
with tab3:
    st.header("Explore Topic Clusters")
    try:
        embed_df = pd.read_csv(r"C:\Users\91735\OneDrive\Documents\Data Science\NLP\Exam\Exam\Input\Project\embeddings.csv")  # x, y, label, text
        fig = px.scatter(embed_df, x="x", y="y", color="label", hover_data=["text"], title="Transcript Embeddings")
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.warning("Upload a precomputed `embeddings.csv` with columns: x, y, label, text.")

