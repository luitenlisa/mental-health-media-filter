import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_text(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).tolist()[0]
    sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    max_index = probs.index(max(probs))
    sentiment = sentiment_labels[max_index]

    return {"Sentiment": sentiment, "Confidence": probs[max_index]}

def detect_triggers(text):
    trigger_words = ["suicide", "self-harm", "panic", "hopeless", "anxious", "trauma", "violence"]
    found_triggers = [word for word in trigger_words if word in text.lower()]
    return found_triggers if found_triggers else None

# Streamlit UI
st.title("Mental Health Media Filter")
user_text = st.text_area("Enter text to analyze:", "")

if st.button("Analyze"):
    sentiment_result = analyze_text(user_text)
    trigger_result = detect_triggers(user_text)

    st.write("### Sentiment Analysis")
    st.write(f"**Sentiment:** {sentiment_result['Sentiment']} ({sentiment_result['Confidence']:.2f})")

    st.write("### Trigger Detection")
    if trigger_result:
        st.write(f"🚨 **Triggers Found:** {', '.join(trigger_result)}")
    else:
        st.write("✅ No triggers detected.")
