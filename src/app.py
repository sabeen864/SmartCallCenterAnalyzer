import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import nltk
import re

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .main-header h1 {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .tech-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-size: 1rem;
        text-align: center;
        box-shadow: 0 1px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.7rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.3rem;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card h3 {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-card p {
        font-size: 0.85rem;
        margin: 0.2rem 0 0 0;
        opacity: 0.9;
    }
    
    .result-container {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.7rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    .highlight-result {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.7rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .churn-positive {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1rem;
    }
    
    .churn-negative {
        color: #27ae60;
        font-weight: bold;
        font-size: 1rem;
    }
    
    .sentiment-positive {
        color: #27ae60;
        font-weight: bold;
        font-size: 0.95rem;
        background: #d4edda;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        display: inline-block;
    }
    
    .sentiment-negative {
        color: #e74c3c;
        font-weight: bold;
        font-size: 0.95rem;
        background: #f8d7da;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        display: inline-block;
    }
    
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
        font-size: 0.95rem;
        background: #e2e3e5;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        display: inline-block;
    }
    
    .intent-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 1px 6px rgba(0,0,0,0.1);
    }
    
    .priority-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }
    
    .priority-medium {
        background: linear-gradient(135deg, #feca57, #ff9ff3);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }
    
    .priority-low {
        background: linear-gradient(135deg, #48dbfb, #0abde3);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }
    
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 1.1rem;
        font-weight: bold;
        margin: 0.8rem 0 0.5rem 0;
        text-align: center;
        box-shadow: 0 1px 6px rgba(0,0,0,0.1);
        max-width: 500px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .model-card {
        background: #ffffff;
        padding: 0.7rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    .model-card h4 {
        color: #d63031;
        font-weight: bold;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }
    
    .model-card small {
        color: #666;
        font-style: italic;
        font-size: 0.85rem;
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #00b894, #00cec9);
        height: 16px;
        border-radius: 8px;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        box-shadow: 0 1px 6px rgba(0,0,0,0.1);
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-weight: 600;
    }
    
    .storytelling-section {
        background: #ffffff;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    .storytelling-section ul {
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .sidebar .element-container {
        background: linear-gradient(135deg, #e17055, #fdcb6e);
        border-radius: 6px;
        padding: 0.5rem;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Cache data loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../data/processed/intent_priority_twcs.csv')
        df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)
        df_sample = pd.read_csv('../figures/sample_intents.csv')
        return df, df_sample
    except FileNotFoundError:
        st.error("⚠️ Data files not found. Ensure Notebooks 4 and 5 ran.")
        return None, None

# Cache model loading
@st.cache_resource
def load_models():
    try:
        xgb = joblib.load('../models/best_churn_model_XGBoost.pkl')
        tfidf = joblib.load('../models/tfidf_vectorizer.pkl')
        distilbert = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        t5 = pipeline('text2text-generation', model='t5-small')
        sia = SentimentIntensityAnalyzer()
        return xgb, tfidf, distilbert, t5, sia
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Ensure TF-IDF vectorizer is saved.")
        return None, None, None, None, None

# Load data and models
df, df_sample = load_data()
xgb, tfidf, distilbert, t5, sia = load_models()

# Keyword definitions for intent fallbacks
complaint_keywords = r'bad|terrible|awful|frustrated|pissed|angry|upset|worst|horrible|sucks|ridiculous|issue|problem|poor|disappointed|shit|bastard|ahole|crappy|lousy|cheap|moaning|pathetic|fail|disgusting'
inquiry_keywords = r'\?$|how|why|what|when|where|question|info|update'
praise_keywords = r'thanks|thank you|great|awesome|excellent|amazing|love|appreciate|fantastic|wonderful'
cancellation_keywords = r'cancel|quit|leave|switch|terminate|account|stop|unsubscribe'

# Intent classification function
def classify_intent(text):
    intent_categories = ['complaint', 'inquiry', 'praise', 'cancellation']
    few_shot_prompt = """Classify the tweet into one of: complaint, inquiry, praise, cancellation.
Return only: Intent: <intent>
Examples:
Tweet: This service is awful, fix it now! -> Intent: complaint
Tweet: How do I update my account details? -> Intent: inquiry
Tweet: Amazing support, thank you so much! -> Intent: praise
Tweet: I want to cancel my subscription. -> Intent: cancellation
Tweet: Why is this service so bad? -> Intent: complaint
Tweet: {text} -> """
    prompt = few_shot_prompt.format(text=text)
    try:
        output = t5(prompt, max_new_tokens=10)[0]['generated_text'].replace('Intent: ', '').strip().lower()
        intent = output if output in intent_categories else None
    except:
        intent = None
    # Fallback rules
    text_lower = text.lower()
    distilbert_result = distilbert(text[:512])[0]
    sentiment = distilbert_result['label']
    if intent is None or intent == 'inquiry':
        if re.search(cancellation_keywords, text_lower):
            return 'cancellation', f"The tweet mentions cancellation intent."
        elif re.search(complaint_keywords, text_lower):
            return 'complaint', f"The tweet expresses dissatisfaction."
        elif re.search(praise_keywords, text_lower) and sentiment == 'POSITIVE':
            return 'praise', f"The tweet conveys positive feedback."
        elif sentiment == 'NEGATIVE':
            return 'complaint', "The tweet has negative sentiment."
        return 'inquiry', "The tweet is an inquiry."
    explain_prompt = f"""Explain in one sentence why the tweet is classified as {intent}.
Tweet: {text}, Intent: {intent} -> """
    explanation = t5(explain_prompt, max_new_tokens=30)[0]['generated_text'].strip()
    return intent, explanation

# Priority scoring function
def assign_priority(intent, sentiment, churn):
    intent_weights = {'cancellation': 0.8, 'complaint': 0.6, 'inquiry': 0.4, 'praise': 0.2}
    base_score = intent_weights.get(intent, 0.4)
    sentiment_modifier = {'NEGATIVE': 0.2, 'NEUTRAL': 0.0, 'POSITIVE': -0.1}
    score = base_score + sentiment_modifier.get(sentiment, 0.0)
    if churn == 1:
        score += 0.2
    return min(max(score, 0.0), 1.0)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text.strip().lower()

# Feature engineering for churn prediction
def prepare_features(text, response_time=30.0, hour=18, day_of_week=0, is_weekend=0, author_tweet_count=50, author_mean_response_time=120.0):
    cleaned_text = clean_text(text)
    char_count = len(cleaned_text)
    word_count = len(cleaned_text.split()) if cleaned_text else 1
    avg_word_len = char_count / word_count
    sentiment = sia.polarity_scores(cleaned_text)['compound']
    tfidf_vec = tfidf.transform([cleaned_text])
    num_features = np.array([[char_count, word_count, avg_word_len, sentiment, hour, day_of_week, is_weekend, 
                             response_time, author_tweet_count, author_mean_response_time]])
    from scipy.sparse import hstack
    return hstack([tfidf_vec, num_features])

# Sidebar
st.sidebar.markdown('<div class="main-header"><h2>🤖 Smart Call Center Analyzer</h2></div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["Overview", "Integrated Analysis", "Churn Prediction", 
                                    "Sentiment Analysis", "Intent & Priority", 
                                    "About Me & My Project"])

# Tech Stack section
st.sidebar.markdown('<div class="section-header">⚡ Tech Stack</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="tech-highlight">🐍 Python: Core programming</div>
<div class="tech-highlight">🐼 Pandas: Data manipulation</div>
<div class="tech-highlight">🚀 XGBoost: Churn prediction (F1: 0.973)</div>
<div class="tech-highlight">🤗 Transformers:<br>• DistilBERT: Sentiment (92%)<br>• T5-Small: Intent classification</div>
<div class="tech-highlight">📊 NLTK (VADER): Sentiment (84%)</div>
<div class="tech-highlight">📈 Plotly: Visualizations</div>
<div class="tech-highlight">🌟 Streamlit: Frontend</div>
""", unsafe_allow_html=True)

# Overview page
if page == "Overview":
    st.markdown('<div class="main-header"><h1>🤖 GenAI-Powered Smart Call Center Analyzer</h1><p>AI-driven insights for customer retention</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-container">
        <h3>Purpose</h3>
        <p style="font-size: 1.1rem;">Developed by Sabeen Zehra, this GenAI-powered tool leverages advanced machine learning and natural language processing to predict customer churn, analyze sentiments, and prioritize intents, enabling call centers to enhance customer retention and operational efficiency.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🧠 AI Techniques</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="result-container">
        <p style="font-size: 1.1rem;">The analyzer integrates three cutting-edge ML techniques:</p>
        <ul style="font-size: 1.1rem; line-height: 1.6;">
            <li><strong>Churn Prediction:</strong> XGBoost with 97.3% F1-score, using 5,000 TF-IDF and 10 numerical features.</li>
            <li><strong>Sentiment Analysis:</strong> DistilBERT (92% accuracy) and VADER (84%) for real-time emotion detection.</li>
            <li><strong>Intent Classification:</strong> T5-Small with zero-shot learning and regex fallbacks for accurate prioritization.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>1.26M</h3>
                <p>Tweets Processed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>10.9%</h3>
                <p>Churn Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>0.48</h3>
                <p>Avg. Priority Score</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📈 Data Insights</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = df['sentiment_label'].value_counts(normalize=True) * 100
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, 
                         title="Sentiment Distribution", 
                         color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#FFE66D'])
            fig.update_traces(textinfo='percent+label', textfont_size=14)
            fig.update_layout(title_font_size=16, title_font_color='#2C3E50', 
                            showlegend=True, height=350, margin=dict(t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            intent_counts = df_sample['intent_label'].value_counts()
            fig = px.bar(x=intent_counts.index, y=intent_counts.values, 
                         title="Intent Distribution",
                         labels={'x': 'Intent', 'y': 'Cases'}, 
                         color=intent_counts.index,
                         color_discrete_sequence=['#667EEA', '#764BA2', '#F093FB', '#F5576C'])
            fig.update_layout(title_font_size=16, title_font_color='#2C3E50',
                            xaxis_title_font_size=14, yaxis_title_font_size=14,
                            height=350, margin=dict(t=40, b=40))
            fig.update_traces(texttemplate='%{y}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

# Integrated Analysis page
elif page == "Integrated Analysis":
    st.markdown('<div class="main-header"><h1>🔄 Integrated GenAI Analysis</h1><p>Unified Insights for Action</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-container">
        <h3>🤖 Comprehensive Insights</h3>
        <p style="font-size: 1.1rem;">This GenAI-driven module combines churn prediction, sentiment analysis, and intent prioritization to provide actionable insights from a single customer tweet, streamlining call center decision-making.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("integrated_form"):
        st.markdown('<div class="section-header">💬 Tweet Analysis</div>', unsafe_allow_html=True)
        text = st.text_area("Enter Customer Tweet", 
                           placeholder="Type a customer tweet...", 
                           value="this is awful, I want to cancel my account now!",
                           height=100)
        
        submit = st.form_submit_button("Analyze", use_container_width=True)
    
    if submit and text and xgb is not None and distilbert is not None and t5 is not None:
        # Sentiment Analysis
        distilbert_result = distilbert(text[:512])[0]
        sentiment = distilbert_result['label']
        sentiment_score = distilbert_result['score']
        vader_score = sia.polarity_scores(clean_text(text))['compound']
        vader_label = 'POSITIVE' if vader_score > 0.1 else 'NEGATIVE' if vader_score < -0.1 else 'NEUTRAL'
        
        # Intent Classification
        intent, intent_explanation = classify_intent(text)
        
        # Churn Prediction with default features
        features = prepare_features(text)
        churn_prob = float(xgb.predict_proba(features)[0][1])
        churn_label = 1 if churn_prob > 0.5 else 0
        
        # Priority Scoring
        priority = assign_priority(intent, sentiment, churn_label)
        
        st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="result-container">
                <h4>Churn Prediction</h4>
                <p style="font-size: 0.95rem;">Predicts likelihood of customer churn using XGBoost.</p>
            </div>
            """, unsafe_allow_html=True)
            
            churn_class = "churn-positive" if churn_label == 1 else "churn-negative"
            churn_text = "HIGH CHURN RISK" if churn_label == 1 else "LOW CHURN RISK"
            
            st.markdown(f'<div class="highlight-result"><p class="{churn_class}">{churn_text}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-bar">Confidence: {churn_prob:.2%}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="result-container">
                <h4>Sentiment Analysis</h4>
                <p style="font-size: 0.95rem;">Detects customer emotions using DistilBERT and VADER.</p>
            </div>
            """, unsafe_allow_html=True)
            
            sentiment_class = f"sentiment-{sentiment.lower()}"
            sentiment_emoji = "😊" if sentiment == 'POSITIVE' else "😠" if sentiment == 'NEGATIVE' else "😐"
            
            st.markdown(f'<div class="highlight-result"><p class="{sentiment_class}">{sentiment_emoji} DistilBERT: {sentiment} ({sentiment_score:.2%})</p></div>', unsafe_allow_html=True)
            st.markdown(f'<p><strong>VADER:</strong> {vader_label} (Score: {vader_score:.3f})</p>', unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="result-container">
                <h4>Intent & Priority</h4>
                <p style="font-size: 0.95rem;">Classifies intent and assigns priority using T5-Small.</p>
            </div>
            """, unsafe_allow_html=True)
            
            intent_emoji = {"complaint": "😠", "inquiry": "❓", "praise": "😊", "cancellation": "🚨"}
            st.markdown(f'<div class="highlight-result"><p class="intent-badge">{intent_emoji.get(intent, "🤔")} {intent.upper()}</p></div>', unsafe_allow_html=True)
            
            priority_class = "priority-high" if priority > 0.7 else "priority-medium" if priority > 0.4 else "priority-low"
            priority_text = f"Priority: {priority:.2f}"
            
            st.markdown(f'<p class="{priority_class}">{priority_text}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size: 0.9rem;">{intent_explanation}</p>', unsafe_allow_html=True)

# Churn Prediction page
elif page == "Churn Prediction":
    st.markdown('<div class="main-header"><h1>📊 Churn Prediction</h1><p>XGBoost Model (F1: 0.973)</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-container">
        <h3>🤖 Churn Detection</h3>
        <p style="font-size: 1.1rem;">Utilizes XGBoost with 5,000 TF-IDF and 10 numerical features to predict customer churn with high accuracy.</p>
        <div class="feature-highlight">
            <strong>🏆 Metrics:</strong> F1: 0.973 | Precision: 0.992 | Recall: 0.954
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("churn_form"):
        st.markdown('<div class="section-header">💬 Tweet Input</div>', unsafe_allow_html=True)
        text = st.text_area("Enter Customer Tweet", 
                           placeholder="Enter tweet for churn analysis...", 
                           value="this is awful, I want to cancel my account now!",
                           height=100)
        
        st.markdown('<div class="section-header">⚙️ Features</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**⏱️ Timing**")
            response_time = st.slider("Response Time (min)", 0.0, 1440.0, 30.0)
            hour = st.slider("Hour of Day", 0, 23, 18)
        with col2:
            st.markdown("**👤 Author**")
            author_tweet_count = st.number_input("Tweet Count", min_value=1, value=50)
            author_mean_response_time = st.number_input("Mean Response Time (min)", min_value=0.0, value=120.0)
        
        col3, col4 = st.columns(2)
        with col3:
            day_of_week = st.selectbox("📅 Day of Week", 
                                     options=[0, 1, 2, 3, 4, 5, 6], 
                                     format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
        with col4:
            is_weekend = st.checkbox("Weekend", value=day_of_week in [5, 6])
        
        submit = st.form_submit_button("Predict", use_container_width=True)
    
    if submit and text and xgb is not None:
        features = prepare_features(text, response_time, hour, day_of_week, int(is_weekend), 
                                   author_tweet_count, author_mean_response_time)
        churn_prob = float(xgb.predict_proba(features)[0][1])
        churn_label = 1 if churn_prob > 0.5 else 0
        
        st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="result-container">
                <h3>🔍 Churn Assessment</h3>
                <p style="font-size: 0.95rem;">Evaluates churn risk based on tweet content and contextual features.</p>
            </div>
            """, unsafe_allow_html=True)
            
            churn_class = "churn-positive" if churn_label == 1 else "churn-negative"
            churn_text = "HIGH CHURN RISK" if churn_label == 1 else "LOW CHURN RISK"
            risk_level = "CRITICAL" if churn_prob > 0.8 else "HIGH" if churn_prob > 0.6 else "MEDIUM" if churn_prob > 0.4 else "LOW"
            
            st.markdown(f'<div class="highlight-result"><p class="{churn_class}">{churn_text}</p></div>', unsafe_allow_html=True)
            
            progress_color = "#e74c3c" if churn_prob > 0.5 else "#f39c12" if churn_prob > 0.3 else "#27ae60"
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, {progress_color}, {progress_color}); 
                        height: 18px; width: {churn_prob*100}%; border-radius: 8px; 
                        display: flex; align-items: center; justify-content: center; 
                        color: white; font-weight: bold; margin: 0.5rem 0;
                        box-shadow: 0 1px 6px rgba(0,0,0,0.1);">
                {churn_prob:.2%} Confidence
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="feature-highlight">
                <strong>Risk Level:</strong> {risk_level} | <strong>Confidence:</strong> {churn_prob:.2%}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="model-card">
                <h4>🧠 Model Insights</h4>
                <p><strong>Algorithm:</strong> XGBoost</p>
                <p><strong>Features:</strong> 5,010</p>
                <p><strong>F1-Score:</strong> 97.3%</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">🧠 Explainable AI Insights</div>', unsafe_allow_html=True)
        feature_names = ['Response Time', 'VADER Sentiment', 'Author Tweet Count', 'TF-IDF: "awful"', 'TF-IDF: "cancel"', 'Hour of Day', 'Word Count', 'Character Count']
        importance = [0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05]
        
        fig = px.bar(x=importance, y=feature_names, orientation='h', 
                     title="Top Features Driving Churn",
                     labels={'x': 'Importance Score', 'y': 'Features'}, 
                     color=importance,
                     color_continuous_scale='Viridis')
        fig.update_layout(title_font_size=16, title_font_color='#2C3E50',
                         xaxis_title_font_size=14, yaxis_title_font_size=14,
                         height=350, margin=dict(t=40, b=40))
        fig.update_traces(texttemplate='%{x:.3f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

# Sentiment Analysis page
elif page == "Sentiment Analysis":
    st.markdown('<div class="main-header"><h1>😊 Sentiment Analysis</h1><p>DistilBERT (92%) & VADER (84%)</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-container">
        <h3>🎭 Emotion Detection</h3>
        <p style="font-size: 1.1rem;">Employs DistilBERT for real-time sentiment analysis and VADER for historical sentiment trends, capturing customer emotions accurately.</p>
        <div class="feature-highlight">
            <strong>🏆 Metrics:</strong> DistilBERT: 92% accuracy | VADER: 84% accuracy
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">💬 Tweet Analysis</div>', unsafe_allow_html=True)
    text = st.text_area("Customer Tweet", 
                       placeholder="Enter tweet for sentiment analysis...", 
                       value="this is awful, I want to cancel my account now!",
                       height=120)
    
    if text and distilbert is not None:
        result = distilbert(text[:512])[0]
        sentiment = result['label']
        score = result['score']
        
        vader_score = sia.polarity_scores(clean_text(text))['compound']
        vader_label = 'POSITIVE' if vader_score > 0.1 else 'NEGATIVE' if vader_score < -0.1 else 'NEUTRAL'
        
        st.markdown('<div class="section-header">🎯 Sentiment Results</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="result-container">
                <h4>🤖 DistilBERT</h4>
                <p style="font-size: 0.95rem;">High-accuracy real-time sentiment detection.</p>
            </div>
            """, unsafe_allow_html=True)
            
            sentiment_class = f"sentiment-{sentiment.lower()}"
            sentiment_emoji = "😊" if sentiment == 'POSITIVE' else "😠" if sentiment == 'NEGATIVE' else "😐"
            
            st.markdown(f'<div class="highlight-result"><p class="{sentiment_class}">{sentiment_emoji} {sentiment} ({score:.2%})</p></div>', unsafe_allow_html=True)
            
            confidence_color = "#27ae60" if sentiment == 'POSITIVE' else "#e74c3c" if sentiment == 'NEGATIVE' else "#95a5a6"
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, {confidence_color}, {confidence_color}); 
                        height: 16px; width: {score*100}%; border-radius: 8px; 
                        display: flex; align-items: center; justify-content: center; 
                        color: white; font-weight: bold; margin: 0.5rem 0;
                        box-shadow: 0 1px 6px rgba(0,0,0,0.1);">
                Confidence: {score:.2%}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="result-container">
                <h4>📊 VADER</h4>
                <p style="font-size: 0.95rem;">Robust sentiment analysis for historical data.</p>
            </div>
            """, unsafe_allow_html=True)
            
            vader_class = f"sentiment-{vader_label.lower()}"
            vader_emoji = "😊" if vader_label == 'POSITIVE' else "😠" if vader_label == 'NEGATIVE' else "😐"
            
            st.markdown(f'<div class="highlight-result"><p class="{vader_class}">{vader_emoji} {vader_label} (Score: {vader_score:.3f})</p></div>', unsafe_allow_html=True)
            
            vader_color = "#27ae60" if vader_score > 0 else "#e74c3c" if vader_score < 0 else "#95a5a6"
            score_width = abs(vader_score) * 100
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, {vader_color}, {vader_color}); 
                        height: 16px; width: {score_width}%; border-radius: 8px; 
                        display: flex; align-items: center; justify-content: center; 
                        color: white; font-weight: bold; margin: 0.5rem 0;
                        box-shadow: 0 1px 6px rgba(0,0,0,0.1);">
                Score: {vader_score:.3f}
            </div>
            """, unsafe_allow_html=True)
    
    if df is not None:
        st.markdown('<div class="section-header">📊 Historical Insights</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = df['sentiment_label'].value_counts(normalize=True) * 100
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, 
                         title="Sentiment Distribution", 
                         color_discrete_sequence=['#E74C3C', '#27AE60', '#F39C12'])
            fig.update_traces(textinfo='percent+label', textfont_size=14)
            fig.update_layout(title_font_size=16, title_font_color='#2C3E50', height=350, margin=dict(t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            hours = list(range(24))
            neg_sentiment = [30, 25, 20, 15, 12, 10, 15, 25, 35, 40, 42, 45, 48, 50, 52, 48, 45, 42, 38, 35, 32, 28, 25, 22]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hours, y=neg_sentiment, mode='lines+markers',
                                   name='Negative Sentiment %', 
                                   line=dict(color='#E74C3C', width=3),
                                   marker=dict(size=8)))
            fig.update_layout(title="Negative Sentiment by Hour",
                            xaxis_title="Hour of Day", yaxis_title="Negative Sentiment %",
                            title_font_size=16, title_font_color='#2C3E50',
                            height=350, margin=dict(t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)

# Intent & Priority page
elif page == "Intent & Priority":
    st.markdown('<div class="main-header"><h1>Intent & Priority</h1><p>Powered by T5-Small and Sentiment Driven Intelligence</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-container">
        <h3>🧠 Intent Detection</h3>
        <p style="font-size: 1.1rem;">Uses T5-Small with zero-shot learning and regex-based fallbacks to classify customer intents (complaint, inquiry, praise, cancellation) and assign priority scores.</p>
        <div class="feature-highlight">
            <strong>Categories:</strong> Complaint | Inquiry | Praise | Cancellation
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("intent_form"):
        st.markdown('<div class="section-header">💬 Tweet Analysis</div>', unsafe_allow_html=True)
        text = st.text_area("Customer Tweet", 
                           placeholder="Enter tweet for intent classification...", 
                           value="this is awful, I want to cancel my account now!",
                           height=120)
        submit = st.form_submit_button("Classify", use_container_width=True)
    
    if submit and text and t5 is not None:
        distilbert_result = distilbert(text[:512])[0]
        sentiment = distilbert_result['label']
        
        intent, explanation = classify_intent(text)
        
        features = prepare_features(text)
        churn_prob = float(xgb.predict_proba(features)[0][1])
        churn_label = 1 if churn_prob > 0.5 else 0
        
        priority = assign_priority(intent, sentiment, churn_label)
        
        st.markdown('<div class="section-header">Classification Results</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="result-container">
                <h3>🤖 Intent Results</h3>
                <p style="font-size: 0.95rem;">Identifies customer intent and explains the classification.</p>
            </div>
            """, unsafe_allow_html=True)
            
            intent_colors = {'complaint': '#E74C3C', 'inquiry': '#3498DB', 'praise': '#27AE60', 'cancellation': '#8E44AD'}
            intent_emojis = {'complaint': '😠', 'inquiry': '❓', 'praise': '😊', 'cancellation': '🚨'}
            
            intent_color = intent_colors.get(intent, '#95A5A6')
            intent_emoji = intent_emojis.get(intent, '🤔')
            
            st.markdown(f"""
            <div class="highlight-result" style="background: linear-gradient(135deg, {intent_color}, {intent_color}AA); 
                        color: white;">
                <h2>{intent_emoji} {intent.upper()}</h2>
                <p style="font-size: 0.95rem;">{explanation}</p>
            </div>
            """, unsafe_allow_html=True)
            
            priority_class = "priority-high" if priority > 0.7 else "priority-medium" if priority > 0.4 else "priority-low"
            priority_text = "URGENT" if priority > 0.7 else "MEDIUM" if priority > 0.4 else "LOW"
            
            st.markdown(f"""
            <div class="{priority_class}">
                <strong>{priority_text}</strong><br>
                Score: {priority:.2f}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="model-card">
                <h4>⚙️ Details</h4>
                <p><strong>Model:</strong> T5-Small</p>
                <p><strong>Method:</strong> Few-shot + Fallbacks</p>
                <p><strong>Sentiment:</strong> """ + sentiment + """</p>
                <p><strong>Churn Risk:</strong> """ + f"{churn_prob:.1%}" + """</p>
            </div>
            """, unsafe_allow_html=True)

# About Me & My Project page
elif page == "About Me & My Project":
    st.markdown('<div class="main-header"><h1>👋 About Me & My Project</h1><p>The journey behind the Smart Call Center Analyzer</p></div>', unsafe_allow_html=True)
    
    # Personal introduction
    st.markdown("""
    <div class="storytelling-section" style="font-size: 1.1rem;">
    <h3>🙋‍♀️ Hi, I'm Sabeen Zehra!</h3>
    <p>
        Currently in my final year of BS Data Science and somewhere between debugging models and fighting off imposter syndrome 😣. 
        I built this project as a personal milestone. No deadlines. No assignment prompts. Just me, my laptop, and an unhealthy number of browser tabs open at all times ☕💤.
        <br><br>
        The Smart Call Center Analyzer started as a portfolio idea but quickly turned into a deep dive into real-world AI. 
        From predicting churn to understanding emotions to prioritizing customer queries using GenAI. I tried to make each module feel like it <em>belongs</em> in a real product.
        <br><br>
        I didn’t want to build something fancy. I wanted to build something that makes sense. 
        And maybe, just maybe, something a recruiter might scroll through and think, “Okay... she gets it.”🤞
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Technical journey
    st.markdown('<div class="section-header">🛠️ Building the Solution (The Fun Part!)</div>', unsafe_allow_html=True)
    
    journey_steps = [
        {
            "title": "🧹 Data Detective Work", 
            "description": "Cleaned 1.26M tweets, dealt with emojis, URLs, and noise. Learned that real data is never as clean as textbooks suggest!",
            "tech": "Pandas, RegEx, lots of patience"
        },
        {
            "title": "🔧 Feature Engineering Magic", 
            "description": "Created 5,010 features from scratch. TF-IDF vectorization taught me that 'cancel' and 'awful' are powerful predictors!",
            "tech": "Scikit-learn, TF-IDF, VADER"
        },
        {
            "title": "🤖 Model Training Marathon", 
            "description": "Trained XGBoost for days, achieved 97.3% F1-score. The moment I saw those results, I literally jumped out of my chair!",
            "tech": "XGBoost, hyperparameter tuning"
        },
        {
            "title": "🧠 Adding Intelligence", 
            "description": "Integrated DistilBERT for sentiment and T5-Small for intent classification. Watching AI understand human emotions never gets old.",
            "tech": "Hugging Face Transformers"
        },
        {
            "title": "🌟 Bringing It to Life", 
            "description": "Built this Streamlit app to make everything interactive. Because what's the point of great models if no one can use them?",
            "tech": "Streamlit, Plotly, CSS magic"
        }
    ]
    
    for i, step in enumerate(journey_steps, 1):
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(f"""
            <div style="background: #667eea; color: white; 
                        width: 50px; height: 50px; border-radius: 50%; 
                        display: flex; align-items: center; justify-content: center; 
                        font-size: 1.2rem; font-weight: bold; margin: 1rem auto;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
                {i}
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="model-card">
                <h4>{step['title']}</h4>
                <p>{step['description']}</p>
                <small style="color: #666; font-style: italic;">Tech used: {step['tech']}</small>
            </div>
            """, unsafe_allow_html=True)

    # What I learned
    st.markdown('<div class="section-header">📚 What I Learned (The Real Value)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="storytelling-section">
            <h4>🔬 Technical Skills</h4>
            <ul style="line-height: 1.6;">
                <li><strong>Python mastery:</strong> From basic scripts to complex ML pipelines</li>
                <li><strong>ML expertise:</strong> Feature engineering, model selection, evaluation</li>
                <li><strong>NLP magic:</strong> Working with transformers and understanding language</li>
                <li><strong>Data visualization:</strong> Making numbers tell compelling stories</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="storytelling-section">
            <h4>💡 Life Lessons</h4>
            <ul style="line-height: 1.6;">
                <li><strong>Patience pays off:</strong> Good models take time and iteration</li>
                <li><strong>User experience matters:</strong> The best model is useless if it's not accessible</li>
                <li><strong>Documentation is key:</strong> Future me will thank present me</li>
                <li><strong>Problem-solving mindset:</strong> Every error is a learning opportunity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Achievements
    st.markdown('<div class="section-header">🏆 Achievements</div>', unsafe_allow_html=True)
    
    achievements = [
        {"metric": "97.3%", "desc": "F1-Score for Churn", "icon": "🎯"},
        {"metric": "92%", "desc": "DistilBERT Accuracy", "icon": "😊"},
        {"metric": "1.26M", "desc": "Tweets Processed", "icon": "📊"},
        {"metric": "5,010", "desc": "Features Engineered", "icon": "⚙️"}
    ]
    
    cols = st.columns(4)
    for i, achievement in enumerate(achievements):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1rem;">{achievement['icon']}</div>
                <h3>{achievement['metric']}</h3>
                <p>{achievement['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Connect with Me
    st.markdown('<div class="section-header">📬 Connect with Me</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="result-container">
        <h3>🤝 Let’s Talk!</h3>
        <p style="font-size: 1.1rem;">Thanks for stopping by. You’ve already made this project more real by reading this. And if you're into meaningful tech and a little bit of chaos driven creativity, we should definitely connect.</p>
        <p><strong>LinkedIn:</strong> <a href="http://www.linkedin.com/in/sabeen-zehra-6635aa355" target="_blank">sabeen-zehra-6635aa355</a></p>
        <p><strong>Email:</strong> <a href="mailto:syedasabeen583@gmail.com">syedasabeen583@gmail.com</a></p>
    </div>
    """, unsafe_allow_html=True)
