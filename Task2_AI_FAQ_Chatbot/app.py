import streamlit as st
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #777;
    font-size: 18px;
    margin-bottom: 30px;
}

.answer-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f7fa;
    border-left: 5px solid #4CAF50;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI FAQ Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about Artificial Intelligence, Machine Learning, Python, Data Science and more.</div>',
    unsafe_allow_html=True
)

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    file_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "faq.csv"
    )

    if not os.path.exists(file_path):

        st.error(
            "❌ faq.csv was not found. "
            "Make sure it is inside the data folder."
        )

        st.stop()

    # Read tab-separated CSV
    data = pd.read_csv(
        file_path,
        sep="\t"
    )

    # Remove accidental spaces from column names
    data.columns = data.columns.str.strip()

    # Check required columns
    required_columns = ["Category", "Question", "Answer"]

    for column in required_columns:

        if column not in data.columns:

            st.error(
                f"❌ Missing column: {column}. "
                f"Available columns: {list(data.columns)}"
            )

            st.stop()

    # Remove empty rows
    data = data.dropna(
        subset=["Question", "Answer"]
    ).reset_index(drop=True)

    # Create processed question column
    data["processed_question"] = data["Question"].apply(
        preprocess_text
    )

    return data


# ============================================================
# BUILD AI MODEL
# ============================================================

@st.cache_resource
def build_model(data):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        data["processed_question"]
    )

    return vectorizer, tfidf_matrix


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

faq_data = load_dataset()

vectorizer, tfidf_matrix = build_model(
    faq_data
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Chatbot Information")

    st.write(
        f"📚 **FAQ Dataset:** {len(faq_data)} questions"
    )

    st.write(
        f"📂 **Categories:** "
        f"{faq_data['Category'].nunique()}"
    )

    st.divider()

    st.subheader("📌 Categories")

    categories = faq_data["Category"].unique()

    for category in categories:
        st.write(f"• {category}")

    st.divider()

    st.info(
        "This chatbot uses TF-IDF and cosine similarity "
        "to find the most relevant FAQ answer."
    )


# ============================================================
# USER INPUT
# ============================================================

question = st.text_input(
    "💬 Ask your question",
    placeholder="Example: What is Artificial Intelligence?"
)

# ============================================================
# SEARCH BUTTON
# ============================================================

if st.button(
    "🔍 Find Answer",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "⚠️ Please enter a question."
        )

    else:

        # Preprocess user question
        processed_question = preprocess_text(
            question
        )

        # Convert question to TF-IDF
        question_vector = vectorizer.transform(
            [processed_question]
        )

        # Calculate similarity
        similarity_scores = cosine_similarity(
            question_vector,
            tfidf_matrix
        )[0]

        # Find best match
        best_index = similarity_scores.argmax()

        best_score = similarity_scores[best_index]

        best_question = faq_data.iloc[
            best_index
        ]["Question"]

        best_answer = faq_data.iloc[
            best_index
        ]["Answer"]

        best_category = faq_data.iloc[
            best_index
        ]["Category"]

        # ====================================================
        # CONFIDENCE THRESHOLD
        # ====================================================

        if best_score < 0.15:

            st.warning(
                "🤔 Sorry, I couldn't find a highly relevant "
                "answer in the FAQ database."
            )

            st.write(
                "Try asking the question using different words."
            )

        else:

            st.success(
                f"✅ Answer found — Confidence: "
                f"{best_score * 100:.1f}%"
            )

            # Category
            st.markdown(
                f"### 📂 Category: {best_category}"
            )

            # Matching question
            st.markdown(
                f"**Matched FAQ:** {best_question}"
            )

            # Answer
            st.markdown(
                f"""
                <div class="answer-box">
                <strong>🤖 Answer:</strong><br><br>
                {best_answer}
                </div>
                """,
                unsafe_allow_html=True
            )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI FAQ Chatbot | CodeAlpha Artificial Intelligence Internship"
)
