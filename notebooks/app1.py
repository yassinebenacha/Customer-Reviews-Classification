import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from datetime import datetime
import altair as alt

# Configuration
st.set_page_config(layout="wide", page_title="Customer Review Classification", page_icon="🖊")

# Style personnalisé
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .big-button {
        font-size: 20px !important;
        padding: 20px 30px !important;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "SVM": "svm_model.pkl",
    "Random Forest": "random_forest.pkl"
}
VECTORIZER_PATH = BASE_DIR / "models/classical_m/tfidf_vectorizer.pkl"

def get_prediction_label(probabilities):
    max_prob_index = np.argmax(probabilities)
    max_prob = probabilities[max_prob_index]
    
    if max_prob_index == 0:
        return "🔴 Négatif", max_prob
    elif max_prob_index == 1:
        return "🟡 Neutre", max_prob
    else:
        return "🟢 Positif", max_prob

def plot_confidence_distribution(probabilities):
    fig = go.Figure()
    
    categories = ['Négatif', 'Neutre', 'Positif']
    colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
    
    for i, (cat, prob, color) in enumerate(zip(categories, probabilities, colors)):
        fig.add_trace(go.Bar(
            name=cat,
            y=[cat],
            x=[prob * 100],
            orientation='h',
            marker_color=color,
            text=f'{prob * 100:.1f}%',
            textposition='auto',
        ))

    fig.update_layout(
        title="Distribution des Probabilités",
        xaxis_title="Probabilité (%)",
        yaxis_title="Sentiment",
        barmode='group',
        height=250,
        showlegend=False
    )
    return fig

def analyze_review_length(review):
    words = len(review.split())
    chars = len(review)
    sentences = len([s for s in review.split('.') if s.strip()])
    
    return pd.DataFrame({
        'Métrique': ['Mots', 'Caractères', 'Phrases'],
        'Valeur': [words, chars, sentences]
    })

def clear_prediction():
    if 'example_review' in st.session_state:
        del st.session_state.example_review
    if 'prediction_made' in st.session_state:
        del st.session_state.prediction_made

def track_predictions(prediction, confidence):
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    st.session_state.prediction_history.append({
        'timestamp': datetime.now(),
        'prediction': prediction,
        'confidence': confidence
    })

try:
    # Load vectorizer
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Interface
    st.title("🖊 Classification des Avis Clients")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Prédiction", "Analyse en Masse", "Statistiques"])
    
    with tab1:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            selected_model = st.selectbox(
                "Choisir un modèle",
                list(MODELS.keys())
            )
            
        with col2:
            language = st.selectbox(
                "Langue",
                ["Français", "English", "Español"]
            )
        
        with col3:
            st.button("Effacer", on_click=clear_prediction, key="clear_btn")

        # Load selected model
        model_path = BASE_DIR / f"models/classical_m/{MODELS[selected_model]}"
        model = joblib.load(model_path)

        # Input area
        example_review = st.text_area(
            "Entrez un avis client",
            placeholder="Tapez votre avis ici...",
            height=150,
            key="review_input"
        )

        # Buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            predict_btn = st.button("📊 Analyser", use_container_width=True)
        with col2:
            save_btn = st.button("💾 Sauvegarder", use_container_width=True)
        with col3:
            export_btn = st.button("📤 Exporter", use_container_width=True)

        if predict_btn and example_review:
            st.session_state.prediction_made = True
            
            # Text analysis
            length_stats = analyze_review_length(example_review)
            
            # Prediction
            example_review_vectorized = vectorizer.transform([example_review])
            proba = model.predict_proba(example_review_vectorized)[0]
            sentiment_label, confidence = get_prediction_label(proba)
            
            # Track prediction
            track_predictions(sentiment_label, confidence)

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### Prédiction: {sentiment_label}")
                st.markdown(f"### Confiance: {confidence * 100:.1f}%")
                
                st.markdown("### Analyse du texte")
                st.dataframe(length_stats, hide_index=True)

            with col2:
                st.plotly_chart(plot_confidence_distribution(proba))

        if save_btn and example_review:
            if 'saved_reviews' not in st.session_state:
                st.session_state.saved_reviews = []
            st.session_state.saved_reviews.append(example_review)
            st.success("Avis sauvegardé!")

        if export_btn and 'saved_reviews' in st.session_state:
            df = pd.DataFrame(st.session_state.saved_reviews, columns=['review'])
            st.download_button(
                "📥 Télécharger les avis sauvegardés",
                df.to_csv(index=False),
                "avis_sauvegardes.csv",
                "text/csv"
            )

    with tab2:
        st.subheader("Prédiction en Masse")
        uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=['csv'])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            if 'review' in data.columns:
                progress_bar = st.progress(0)
                predictions = []
                probabilities = []
                
                for i, review in enumerate(data['review']):
                    prob = model.predict_proba(vectorizer.transform([review]))[0]
                    sentiment_label, confidence = get_prediction_label(prob)
                    predictions.append(sentiment_label)
                    probabilities.append(confidence)
                    progress_bar.progress((i + 1) / len(data))
                
                data['prediction'] = predictions
                data['confidence'] = [p * 100 for p in probabilities]
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(data, names='prediction', title='Distribution des Prédictions')
                    st.plotly_chart(fig)
                
                with col2:
                    fig = px.histogram(data, x='confidence', title='Distribution des Niveaux de Confiance')
                    st.plotly_chart(fig)
                
                st.dataframe(data)
                st.download_button("Télécharger les Résultats", data.to_csv(index=False), "resultats.csv", "text/csv")
            else:
                st.error("Le fichier CSV doit contenir une colonne 'review'.")

    with tab3:
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            st.subheader("Historique des Prédictions")
            
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            fig = px.line(history_df, x='timestamp', y='confidence', 
                         title='Évolution de la Confiance')
            st.plotly_chart(fig)
            
            pred_counts = history_df['prediction'].value_counts()
            fig = px.pie(values=pred_counts.values, names=pred_counts.index, 
                        title='Distribution des Prédictions')
            st.plotly_chart(fig)
            
            if st.button("🗑️ Effacer l'historique"):
                st.session_state.prediction_history = []
                st.experimental_rerun()
        else:
            st.info("Aucun historique disponible. Commencez à faire des prédictions!")

except FileNotFoundError as e:
    st.error(f"❌ Erreur: Fichier non trouvé\n{str(e)}")
except Exception as e:
    st.error(f"❌ Une erreur s'est produite: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Developped by Abdellah Ennajari*")
