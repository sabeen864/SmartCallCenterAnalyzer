# **Smart Call Center Analyzer**

**Author:** Sabeen Zehra  
**GitHub:** [sabeen864](https://github.com/sabeen864)

---

## **Project Overview**

The **Smart Call Center Analyzer** is an AI-powered web application that optimizes call center operations by predicting customer churn, analyzing sentiments, and prioritizing intents. Built by **Sabeen Zehra** as a portfolio project, it processes **1.26 million** customer service tweets from the **TWCS dataset** to deliver actionable insights, enhancing customer retention and agent efficiency through advanced machine learning and NLP.

---

## **Key Features**

- **Churn Prediction**: XGBoost model (F1-score: 0.973) with 5,010 features (5,000 TF-IDF + 10 numerical) to identify at-risk customers.  
- **Sentiment Analysis**: DistilBERT (92% accuracy) for real-time emotion detection and VADER (84% accuracy) for historical trends.  
- **Intent Classification & Priority Scoring**: T5-Small with zero-shot learning and regex fallbacks to classify intents (complaint, inquiry, praise, cancellation) and assign priority scores.  
- **Interactive Frontend**: Streamlit dashboard with Plotly visualizations (e.g., sentiment distribution, intent vs. priority).  
- **End-to-End Pipeline**: Data cleaning, feature engineering, modeling, and deployment, showcasing comprehensive data science skills.

This project demonstrates proficiency in **Python**, **Pandas**, **scikit-learn**, **Hugging Face Transformers**, **NLTK**, **Streamlit**, and **Plotly**, with a focus on business impact and user experience.

---

## **Directory Structure**

```
SmartCallCenterAnalyzer/
├── data/
│   ├── features/
│   │   ├── X_features.pkl
│   │   ├── y_labels.pkl
│   ├── processed/  (Generated after running notebooks)
│   │   ├── cleaned_twcs.csv
│   │   ├── cleaned_uci.csv
│   │   ├── intent_priority_twcs.csv
│   │   ├── labeled_intents.csv (for debugging)
│   │   ├── sentiment_twcs.csv
│   ├── raw/  (Download manually)
│   │   ├── amazon_cells_labelled.txt
│   │   ├── twcs.csv
├── figures/
│   ├── *.png
│   ├── sample_intents.csv
├── models/
│   ├── best_churn_model_XGBoost.pkl
│   ├── tfidf_vectorizer.pkl
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_Modeling_Customer_Churn.ipynb
│   ├── 04_sentiment_analysis.ipynb
│   ├── 05_intent_classification.ipynb
├── src/
│   ├── app.py
├── index.html  (Static portfolio page - Quick Review Link)
├── .gitignore
├── README.md
├── requirements.txt
```

---

## **Prerequisites**

- **Operating System**: Windows, macOS, or Linux  
- **Python**: Version 3.8 or higher  
- **Git**: For cloning the repository  
- **Internet Connection**: Required for downloading GenAI models (DistilBERT, T5-Small) and raw data  
- **Hardware**: Minimum 8GB RAM; GPU recommended for faster inference  
- **Disk Space**: 1GB for TWCS dataset (500MB) and model weights (~500MB)  

---

## **Setup Instructions**

### **Clone the Repository**
```bash
git clone https://github.com/sabeen864/SmartCallCenterAnalyzer.git
cd SmartCallCenterAnalyzer
```

### **Set Up a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **Download Raw Data**

The `.csv` files are not included in the repository. Download them manually:

- **TWCS Dataset**: Kaggle - Customer Support on Twitter  
  Use the file `twcs.csv`.  

- **UCI Dataset**: UCI Machine Learning Repository - Sentiment Labelled Sentences  
  Use the file `amazon_cells_labelled.txt` (one of the three provided datasets; select this for consistency with the project).

Place the downloaded files in `data/raw/`.

---

## **Verify Data and Models**

- **Pre-included Files**:  
  `data/features/X_features.pkl`, `data/features/y_labels.pkl`,  
  `models/best_churn_model_XGBoost.pkl`, `models/tfidf_vectorizer.pkl`  

- **Generated Files**:  
  Run the notebooks to generate `data/processed/` files (e.g., `cleaned_twcs.csv`) directly in the specified paths.

---

## **Run the Notebooks**

Execute the notebooks in order to process data and save outputs:
```bash
jupyter notebook
```

Open each notebook (`01_data_cleaning.ipynb` to `05_intent_classification.ipynb`) and **run all cells**. Outputs will be saved automatically to the `data/processed/` and `figures/` directories as specified in the code.

---

## **Download GenAI Models**

The app automatically downloads **DistilBERT** and **T5-Small** on first run. To pre-download:

```python
from transformers import pipeline
import nltk
nltk.download('vader_lexicon')

pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
pipeline('text2text-generation', model='t5-small')
```

This caches models in `~/.cache/huggingface/hub` (~500MB).

---

## **Run the Streamlit Application**

```bash
cd src
streamlit run app.py
```

Access the app at: [http://localhost:8501](http://localhost:8501)

---

## **Quick Review Option**

Cloning the repository and running the notebooks or Streamlit app can be a lot of work just to review the project. To provide a **quick look**, visit:

🔗 **[Static Preview Page](https://sabeen864.github.io/SmartCallCenterAnalyzer/)**

This includes:

- Tabbed interface with screenshots of the Streamlit app
- Live-rendered "About Me & My Project" section
- Custom CSS with gradients, cards, and badges  

---

## **Verify Setup**

Test with a sample tweet:

> `"this is awful, I want to cancel my account now!"`

Expected output:

- **Churn**: High risk (>50% probability)  
- **Sentiment**: NEGATIVE (DistilBERT, VADER)  
- **Intent**: Cancellation (T5-Small)  
- **Priority**: High (>0.7)

---

## **Technical Details**

### **Data Pipeline**

- **Data Cleaning**: Processed 1.26M tweets with Pandas and RegEx to remove noise (Notebook 1).  
- **Feature Engineering**: Generated 5,010 features (5,000 TF-IDF + 10 numerical, e.g., sentiment, response time) (Notebook 2).  
- **Churn Modeling**: Trained XGBoost with hyperparameter tuning (F1: 0.973) (Notebook 3).  
- **Sentiment Analysis**: Combined VADER (84% accuracy) and DistilBERT (92% accuracy) (Notebook 4).  
- **Intent Classification**: Used T5-Small with zero-shot learning and regex fallbacks to classify intents, addressing inquiry bias (Notebook 5).  

---

### **Model Performance**

- **XGBoost (Churn)**: F1: 0.973, Precision: 0.992, Recall: 0.954  
- **DistilBERT (Sentiment)**: Accuracy: 92%  
- **VADER (Sentiment)**: Accuracy: 84%  
- **T5-Small (Intent)**: Validated on a small ground-truth set with regex fallbacks  

---

## **Visualizations**

- **Sentiment Distribution**: Pie chart of positive, negative, neutral sentiments  
- **Intent Distribution**: Bar chart of intent categories  
- **Feature Importance**: Bar chart of top churn predictors  
- **Priority Scores**: Box plot of priority by intent  

---

## **Challenges Overcome**

- Pivoted from text summarization to intent classification in Notebook 5 due to model performance and business relevance, showcasing adaptability.  
- Mitigated inquiry bias in intent classification with regex fallbacks, with plans for T5-Small fine-tuning.

---

## **Future Enhancements**

- Integrate realtime Twitter API for live analysis  
- Fine-tune T5-Small on domain-specific intents  
- Add a downloadable report feature for managers  
- Optimize inference for low-resource systems

---

## **Additional Files**

**index.html**: A static HTML portfolio page created to showcase the project.  
It includes a tabbed interface with screenshots of the Streamlit app (e.g., Overview, Churn Prediction) and a live-rendered "About Me & My Project" section, matching the Streamlit design.  
The page features custom CSS with gradients, cards, and badges, and addresses issues like scrollable screenshots by using multiple image placeholders.

---

## **Acknowledgments**

- **Dataset**: Customer Support on Twitter  
- **Libraries**: Streamlit, Pandas, scikit-learn, Hugging Face Transformers, NLTK, Plotly  
- **Inspiration**: Real-world AI applications in customer service

---

## **Connect**

Thanks for stopping by. You’ve made this project real by reading this. If you’re into meaningful tech and a bit of chaotic creativity, let’s connect!

- **GitHub**: [sabeen864](https://github.com/sabeen864)  
- **Email**: syedasabeen583@gmail.com  
- **LinkedIn**: [Sabeen Zehra](http://www.linkedin.com/in/sabeen-zehra-6635aa355)
