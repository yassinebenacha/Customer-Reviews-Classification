import os
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from io import BytesIO
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')

# Set the page configuration at the very beginning
st.set_page_config(page_title="Advanced Sentiment Analyzer", layout="wide")

class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.history = []
        self.setup_models()

    def setup_models(self):
        """Load classification models with improved error handling."""
        try:
            project_root = self._get_project_root()

            # Update model directories with absolute paths
            possible_model_dirs = [
                os.path.abspath(os.path.join(project_root, "models", "classical_m")),
                os.path.abspath(os.path.join(project_root, "models", "classical_ml")),
                os.path.abspath(os.path.join(project_root, "models")),
                os.path.abspath(os.path.join(os.path.dirname(project_root), "models"))
            ]

            # Find the first existing model directory
            models_dir = next((path for path in possible_model_dirs if os.path.exists(path)), None)

            if not models_dir:
                st.error(f"Model directory not found. Searched in: {possible_model_dirs}")
                return

            # Load models with error handling for each file
            model_files = {
                'vectorizer': 'tfidf_vectorizer.pkl',
                'logistic_regression': 'logistic_regression.pkl',
                'svm': 'svm_model.pkl'
            }

            for model_name, filename in model_files.items():
                try:
                    model_path = os.path.join(models_dir, filename)
                    if os.path.exists(model_path):
                        if model_name == 'vectorizer':
                            self.vectorizer = joblib.load(model_path)
                        else:
                            self.models[model_name] = joblib.load(model_path)
                    else:
                        st.warning(f"Model file not found: {filename}")
                except Exception as e:
                    st.error(f"Error loading {filename}: {str(e)}")

        except Exception as e:
            st.error(f"Error in setup_models: {str(e)}")
            st.error(f"Current directory: {os.getcwd()}")
            st.error(f"Project root: {project_root}")

    def _get_project_root(self):
        """Find the root directory of the project."""
        try:
            current_file = os.path.abspath(__file__)
            return os.path.dirname(os.path.dirname(current_file))
        except NameError:
            # Handle case when running in Streamlit
            return os.path.abspath(os.path.join(os.getcwd()))

    def analyze_sentiment(self, text, model_name='TextBlob'):
        """Analyze sentiment using different methods."""
        try:
            if not isinstance(text, str):
                text = str(text)

            if model_name == 'TextBlob':
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity

                if polarity > 0.1:
                    return "positive", [0.1, 0.2, 0.7]
                elif polarity < -0.1:
                    return "negative", [0.7, 0.2, 0.1]
                else:
                    return "neutral", [0.2, 0.6, 0.2]

            elif model_name in ['Logistic Regression', 'SVM']:
                if not self.vectorizer or not self.models.get(model_name.lower().replace(' ', '_')):
                    raise ValueError(f"Required models for {model_name} not loaded")

                model = self.models[model_name.lower().replace(' ', '_')]
                text_vectorized = self.vectorizer.transform([text])
                prediction = model.predict(text_vectorized)
                probabilities = model.predict_proba(text_vectorized)[0]

                sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
                sentiment = sentiment_labels.get(prediction[0], "unknown")

                return sentiment, probabilities.tolist()

            else:
                st.warning(f"Unknown model: {model_name}")
                return "unknown", [0.33, 0.34, 0.33]

        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            return "error", [0.33, 0.34, 0.33]

    def analyze_dataframe(self, df, text_column, model_name='TextBlob'):
        """Analyze sentiment for entire DataFrame."""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        results = [self.analyze_sentiment(str(text), model_name) for text in df[text_column]]
        sentiments, probabilities = zip(*results)

        df = df.copy()
        df['sentiment'] = sentiments
        df['probabilities'] = probabilities
        df['negative_prob'] = [p[0] for p in probabilities]
        df['neutral_prob'] = [p[1] for p in probabilities]
        df['positive_prob'] = [p[2] for p in probabilities]

        return df

def create_3d_sentiment_visualization(sentiment, probabilities):
    """Create a 3D visualization of sentiment probabilities."""
    labels = ['Negative', 'Neutral', 'Positive']
    colors = ['red', 'gray', 'green']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=probabilities,
        hole=.4,
        marker_colors=colors
    )])

    fig.update_layout(
        title='Sentiment Probability Distribution',
        annotations=[{
            'text': sentiment.upper(),
            'x': 0.5,
            'y': 0.5,
            'font_size': 20,
            'showarrow': False
        }]
    )

    return fig

def create_3d_probability_scatter(probabilities):
    """Create a 3D scatter plot of sentiment probabilities."""
    fig = go.Figure(data=[go.Scatter3d(
        x=[0, 1, 2],
        y=[probabilities[0], probabilities[1], probabilities[2]],
        z=[0, 0, 0],
        mode='markers+text',
        marker=dict(
            size=10,
            color=['red', 'gray', 'green'],
            opacity=0.8
        ),
        text=['Negative', 'Neutral', 'Positive'],
        textposition="bottom center"
    )])

    fig.update_layout(
        title='3D Sentiment Probability Visualization',
        scene=dict(
            xaxis_title='Sentiment Categories',
            yaxis_title='Probability',
            zaxis_title='Baseline'
        ),
        width=600,
        height=500
    )

    return fig

def create_3d_scatter_visualization(df):
    """Create 3D scatter plot of sentiment probabilities."""
    fig = go.Figure(data=[go.Scatter3d(
        x=df['negative_prob'],
        y=df['neutral_prob'],
        z=df['positive_prob'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['sentiment'].map({'positive': 'green', 'neutral': 'gray', 'negative': 'red'}),
            opacity=0.8
        ),
        text=df['sentiment'],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='3D Sentiment Distribution',
        scene=dict(
            xaxis_title='Negative Probability',
            yaxis_title='Neutral Probability',
            zaxis_title='Positive Probability'
        ),
        width=800,
        height=600
    )
    return fig

def create_sentiment_heatmap(df):
    """Create sentiment correlation heatmap."""
    correlation_data = df[['negative_prob', 'neutral_prob', 'positive_prob']].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0)
    plt.title('Sentiment Probability Correlations')
    return plt.gcf()

def create_sentiment_distribution_plot(df):
    """Create detailed sentiment distribution plot."""
    plt.figure(figsize=(12, 6))

    colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    sentiment_counts = df['sentiment'].value_counts()

    bars = plt.bar(sentiment_counts.index, sentiment_counts.values)
    for bar, sentiment in zip(bars, sentiment_counts.index):
        bar.set_color(colors[sentiment])

    plt.title('Sentiment Distribution', pad=20)
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

    return plt.gcf()

def create_interactive_probability_sunburst(df):
    """Create interactive sunburst chart of sentiment probabilities."""
    avg_probs = {
        'Negative': df['negative_prob'].mean(),
        'Neutral': df['neutral_prob'].mean(),
        'Positive': df['positive_prob'].mean()
    }

    fig = go.Figure(go.Sunburst(
        labels=['Sentiment', 'Negative', 'Neutral', 'Positive'],
        parents=['', 'Sentiment', 'Sentiment', 'Sentiment'],
        values=[1, avg_probs['Negative'], avg_probs['Neutral'], avg_probs['Positive']],
        branchvalues='total',
        marker=dict(
            colors=['lightgray', 'red', 'gray', 'green']
        ),
    ))

    fig.update_layout(
        title='Sentiment Probability Distribution (Sunburst)',
        width=800,
        height=800
    )

    return fig

def create_time_series_sentiment(df, date_column):
    """Create time series visualization of sentiment trends."""
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])  # Drop rows with NaT values
        df_grouped = df.groupby([pd.Grouper(key=date_column, freq='D'), 'sentiment']).size().unstack(fill_value=0)

        fig = go.Figure()
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in df_grouped.columns:
                fig.add_trace(go.Scatter(
                    x=df_grouped.index,
                    y=df_grouped[sentiment],
                    name=sentiment.capitalize(),
                    mode='lines+markers'
                ))

        fig.update_layout(
            title='Sentiment Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Count',
            width=800,
            height=500
        )

        return fig
    return None

def generate_pdf_report(df, text_column):
    """Generate PDF report with analysis results."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont('Helvetica-Bold', 24)
    c.drawString(50, height - 50, "Sentiment Analysis Report")

    # Summary statistics
    c.setFont('Helvetica', 12)
    y = height - 100

    sentiment_counts = df['sentiment'].value_counts()
    c.drawString(50, y, f"Total Records Analyzed: {len(df)}")
    y -= 20

    for sentiment, count in sentiment_counts.items():
        c.drawString(50, y, f"{sentiment.capitalize()}: {count} ({count/len(df)*100:.1f}%)")
        y -= 20

    # Add timestamp
    c.setFont('Helvetica', 10)
    c.setFont('Helvetica-Oblique', 10)
    c.drawString(50, 50, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.save()
    buffer.seek(0)
    return buffer

def create_word_cloud(text):
    """Create a word cloud visualization."""
    if not text or not text.strip():
        st.warning("The text input is empty or contains only whitespace.")
        return None

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    return plt.gcf()

def create_3d_surface_plot(df):
    """Create a 3D surface plot of sentiment probabilities."""
    fig = go.Figure(data=[go.Surface(
        z=df[['negative_prob', 'neutral_prob', 'positive_prob']].values,
        colorscale='Viridis'
    )])

    fig.update_layout(
        title='3D Surface Plot of Sentiment Probabilities',
        scene=dict(
            xaxis_title='Negative Probability',
            yaxis_title='Neutral Probability',
            zaxis_title='Positive Probability'
        ),
        width=800,
        height=600
    )

    return fig

def create_confusion_matrix(df, true_sentiment_column, predicted_sentiment_column):
    """Create a confusion matrix for model performance."""
    confusion_matrix = pd.crosstab(df[true_sentiment_column], df[predicted_sentiment_column], rownames=['Actual'], colnames=['Predicted'])

    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Sentiment')
    plt.xlabel('Predicted Sentiment')
    return plt.gcf()

def create_roc_curve(df, true_sentiment_column, probabilities):
    """Create ROC curves for different models."""
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(10, 7))
    for sentiment in ['negative', 'neutral', 'positive']:
        fpr, tpr, _ = roc_curve(df[true_sentiment_column] == sentiment, df[f'{sentiment}_prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{sentiment.capitalize()} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    return plt.gcf()

def single_text_analysis(analyzer):
    col1, col2 = st.columns([2, 1])

    with col1:
        text_input = st.text_area(
            "Enter your text here",
            height=150,
            placeholder="Type or paste your text here..."
        )

    with col2:
        model_choice = st.selectbox(
            "Choose a model",
            ["TextBlob", "Logistic Regression", "SVM"]
        )

        if st.button("Analyze Text", type="primary"):
            if text_input and text_input.strip():  # Check if text_input is not empty or whitespace
                with st.spinner("Analyzing..."):
                    sentiment, probabilities = analyzer.analyze_sentiment(text_input, model_choice)

                    st.markdown(f"""
                    ### Analysis Result
                    - **Sentiment:** {sentiment.upper()}
                    - **Probabilities:**
                        - Negative: {probabilities[0]:.2%}
                        - Neutral: {probabilities[1]:.2%}
                        - Positive: {probabilities[2]:.2%}
                    """)

                    col3, col4 = st.columns(2)

                    with col3:
                        st.plotly_chart(create_3d_sentiment_visualization(sentiment, probabilities))

                    with col4:
                        st.plotly_chart(create_3d_probability_scatter(probabilities))

                    # Additional visualizations for single text analysis
                    st.subheader("Additional Visualizations")

                    col5, col6 = st.columns(2)

                    with col5:
                        word_cloud_fig = create_word_cloud(text_input)
                        if word_cloud_fig:
                            st.pyplot(word_cloud_fig)

                    with col6:
                        fig, ax = plt.subplots()
                        ax.bar(['Negative', 'Neutral', 'Positive'], probabilities, color=['red', 'gray', 'green'])
                        ax.set_title('Sentiment Probabilities')
                        ax.set_ylabel('Probability')
                        st.pyplot(fig)

            else:
                st.warning("Please enter some text to analyze")

def bulk_csv_analysis(analyzer):
    st.header("Bulk Customer Review Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    with col2:
        st.markdown("""
        ### File Requirements
        - CSV format
        - Text column for analysis
        - Optional date column for trends
        """)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            col1, col2, col3 = st.columns(3)

            with col1:
                text_column = st.selectbox("Select text column", df.columns)
            with col2:
                model_choice = st.selectbox("Choose Analysis Model",
                    ["TextBlob", "Logistic Regression", "SVM"])
            with col3:
                date_column = st.selectbox("Select date column (optional)",
                    ['None'] + list(df.columns))

            if st.button("Analyze CSV", type="primary"):
                with st.spinner("Analyzing..."):
                    analyzed_df = analyzer.analyze_dataframe(df, text_column, model_choice)

                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Data & Basic Visualizations",
                        "Advanced Visualizations",
                        "Time Series Analysis",
                        "Download Options"
                    ])

                    with tab1:
                        st.dataframe(analyzed_df)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(create_3d_scatter_visualization(analyzed_df),
                                use_container_width=True)
                        with col2:
                            st.pyplot(create_sentiment_heatmap(analyzed_df))

                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(create_interactive_probability_sunburst(analyzed_df),
                                use_container_width=True)
                        with col2:
                            st.pyplot(create_sentiment_distribution_plot(analyzed_df))

                    with tab3:
                        if date_column != 'None':
                            time_series_fig = create_time_series_sentiment(analyzed_df, date_column)
                            if time_series_fig:
                                st.plotly_chart(time_series_fig, use_container_width=True)
                        else:
                            st.info("Select a date column to view time series analysis")

                    with tab4:
                        col1, col2 = st.columns(2)
                        with col1:
                            # CSV download
                            csv = analyzed_df.to_csv(index=False)
                            st.download_button(
                                "Download CSV Results",
                                csv,
                                "analyzed_sentiment.csv",
                                "text/csv",
                                key='download-csv'
                            )

                        with col2:
                            # PDF download
                            pdf_buffer = generate_pdf_report(analyzed_df, text_column)
                            st.download_button(
                                "Download PDF Report",
                                pdf_buffer,
                                "sentiment_analysis_report.pdf",
                                "application/pdf",
                                key='download-pdf'
                            )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def advanced_visualization(analyzer):
    st.header("Advanced Sentiment Insights")

    uploaded_file = st.file_uploader("Upload CSV for Advanced Analysis", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            col1, col2 = st.columns(2)

            with col1:
                text_column = st.selectbox("Select text column", df.columns)

            with col2:
                viz_options = [
                    "Sentiment Distribution",
                    "Probability Boxplot",
                    "Sentiment Confidence",
                    "Interactive Sunburst Chart",
                    "Time Series Analysis",
                    "Word Cloud",
                    "3D Surface Plot",
                    "Confusion Matrix",
                    "ROC Curve"
                ]
                selected_viz = st.multiselect(
                    "Select Visualizations",
                    viz_options,
                    default=["Sentiment Distribution"]
                )

            if st.button("Generate Advanced Visualizations", type="primary"):
                with st.spinner("Processing..."):
                    analyzed_df = analyzer.analyze_dataframe(df, text_column)

                    for viz in selected_viz:
                        st.subheader(viz)

                        if viz == "Sentiment Distribution":
                            col1, col2 = st.columns(2)
                            with col1:
                                st.pyplot(create_sentiment_distribution_plot(analyzed_df))
                            with col2:
                                # Add a pie chart for comparison
                                fig = px.pie(analyzed_df, names='sentiment', title='Sentiment Distribution (Pie)')
                                st.plotly_chart(fig)

                        elif viz == "Probability Boxplot":
                            fig = go.Figure()
                            for prob in ['negative_prob', 'neutral_prob', 'positive_prob']:
                                fig.add_trace(go.Box(y=analyzed_df[prob], name=prob.replace('_prob', '').title()))
                            fig.update_layout(title='Probability Distribution by Sentiment', height=500)
                            st.plotly_chart(fig)

                        elif viz == "Sentiment Confidence":
                            # Create confidence visualization
                            fig = go.Figure()
                            sentiments = analyzed_df['sentiment'].unique()
                            for sentiment in sentiments:
                                mask = analyzed_df['sentiment'] == sentiment
                                fig.add_trace(go.Violin(
                                    y=analyzed_df[mask][f'{sentiment}_prob'],
                                    name=sentiment,
                                    box_visible=True,
                                    meanline_visible=True
                                ))
                            fig.update_layout(title='Sentiment Confidence Distribution', height=500)
                            st.plotly_chart(fig)

                        elif viz == "Interactive Sunburst Chart":
                            st.plotly_chart(create_interactive_probability_sunburst(analyzed_df))

                        elif viz == "Time Series Analysis":
                            date_columns = df.select_dtypes(include=['datetime64', 'object']).columns
                            if len(date_columns) > 0:
                                date_col = st.selectbox("Select date column", date_columns)
                                time_series_fig = create_time_series_sentiment(analyzed_df, date_col)
                                if time_series_fig:
                                    st.plotly_chart(time_series_fig)
                            else:
                                st.info("No date columns detected for time series analysis")

                        elif viz == "Word Cloud":
                            if not analyzed_df[text_column].empty and analyzed_df[text_column].str.strip().any():
                                st.pyplot(create_word_cloud(' '.join(analyzed_df[text_column].astype(str))))
                            else:
                                st.warning("The selected text column is empty or contains only whitespace.")

                        elif viz == "3D Surface Plot":
                            st.plotly_chart(create_3d_surface_plot(analyzed_df))

                        elif viz == "Confusion Matrix":
                            true_sentiment_column = st.selectbox("Select true sentiment column", df.columns)
                            st.pyplot(create_confusion_matrix(analyzed_df, true_sentiment_column, 'sentiment'))

                        elif viz == "ROC Curve":
                            true_sentiment_column = st.selectbox("Select true sentiment column", df.columns)
                            st.pyplot(create_roc_curve(analyzed_df, true_sentiment_column, ['negative_prob', 'neutral_prob', 'positive_prob']))

        except Exception as e:
            st.error(f"Error in advanced visualization: {str(e)}")

def main():
    st.title("🎭 Advanced Sentiment Analyzer")

    # Initialize session state if not exists
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedSentimentAnalyzer()

    # Add sidebar with additional information
    with st.sidebar:
        st.header("Navigation")
        menu = ["Single Text Analysis", "Bulk CSV Analysis", "Advanced Visualization"]
        choice = st.selectbox("Select Mode", menu)

        st.header("About")
        st.markdown("""
        This advanced sentiment analyzer provides:
        - Single text analysis
        - Bulk CSV processing
        - Multiple visualization options
        - PDF report generation
        - Time series analysis
        """)

        st.header("Model Information")
        st.markdown("""
        Available models:
        - TextBlob (Default)
        - Logistic Regression
        - Support Vector Machine (SVM)
        """)

        st.header("by: Ennajari Abdellah")
        st.markdown("""
        Please, if you encounter any error or any problem, feel free to send me your feedback or a screenshot of the error:
        - yassinebenacha1@gmail.com
        """)

    # Main content based on selection
    if choice == "Single Text Analysis":
        single_text_analysis(st.session_state.analyzer)
    elif choice == "Bulk CSV Analysis":
        bulk_csv_analysis(st.session_state.analyzer)
    elif choice == "Advanced Visualization":
        advanced_visualization(st.session_state.analyzer)

if __name__ == "__main__":
    main()
