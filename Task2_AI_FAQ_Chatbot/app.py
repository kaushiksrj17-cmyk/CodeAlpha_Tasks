import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import random
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI FAQ Chatbot Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #777;
    margin-bottom: 25px;
}

.answer-card {
    padding: 18px;
    border-radius: 14px;
    background-color: #f5f7fa;
    border-left: 5px solid #4CAF50;
    margin-top: 10px;
}

.info-card {
    padding: 15px;
    border-radius: 12px;
    background-color: #f5f7fa;
    margin-bottom: 10px;
}

.metric-card {
    padding: 12px;
    border-radius: 12px;
    background-color: #f5f7fa;
    text-align: center;
}

.small-text {
    color: #777;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI FAQ Chatbot Pro</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent FAQ assistant powered by NLP, TF-IDF and Cosine Similarity'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    file_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "faq.csv"
    )

    if not os.path.exists(file_path):

        st.error(
            "❌ FAQ dataset not found.\n\n"
            "Expected location: data/faq.csv"
        )

        st.stop()

    # The dataset is tab-separated
    data = pd.read_csv(
        file_path,
        sep="\t"
    )

    # Clean column names
    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
    )

    # Handle possible BOM in first column
    data.columns = [
        column.replace("\ufeff", "")
        for column in data.columns
    ]

    # Required columns
    required = [
        "Category",
        "Question",
        "Answer"
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        st.error(
            f"❌ Missing columns: {missing}\n\n"
            f"Available columns: {list(data.columns)}"
        )

        st.stop()

    # Remove empty rows
    data = data.dropna(
        subset=[
            "Question",
            "Answer"
        ]
    ).reset_index(drop=True)

    # IMPORTANT:
    # Create processed_question BEFORE build_model()
    data["processed_question"] = (
        data["Question"]
        .astype(str)
        .apply(preprocess_text)
    )

    return data


# ============================================================
# BUILD TF-IDF MODEL
# ============================================================

@st.cache_resource
def build_vectorizer(data):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        data["processed_question"]
    )

    return vectorizer, matrix


# ============================================================
# LOAD DATA + MODEL
# ============================================================

faq_data = load_data()

vectorizer, tfidf_matrix = build_vectorizer(
    faq_data
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "favorite_questions" not in st.session_state:
    st.session_state.favorite_questions = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Chatbot Controls")

    # --------------------------------------------------------
    # CATEGORY FILTER
    # --------------------------------------------------------

    st.subheader("📂 Category Filter")

    categories = [
        "All Categories"
    ] + sorted(
        faq_data["Category"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_category = st.selectbox(
        "Choose category",
        categories
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader("📊 Dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "FAQs",
            len(faq_data)
        )

    with col2:
        st.metric(
            "Categories",
            faq_data["Category"].nunique()
        )

    st.divider()

    # --------------------------------------------------------
    # CHAT CONTROLS
    # --------------------------------------------------------

    st.subheader("🧹 Chat Controls")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.search_history = []
        st.session_state.last_result = None

        st.rerun()

    if st.button(
        "🔄 Reset Session",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.search_history = []
        st.session_state.favorite_questions = []
        st.session_state.last_result = None

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # FAVORITES
    # --------------------------------------------------------

    st.subheader("⭐ Favorites")

    if st.session_state.favorite_questions:

        for favorite in st.session_state.favorite_questions:
            st.write(
                f"• {favorite}"
            )

    else:

        st.caption(
            "No favorite questions yet."
        )

    st.divider()

    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.subheader("ℹ️ About")

    st.write(
        "This chatbot uses Natural Language Processing "
        "to find the most relevant FAQ answer."
    )

    st.write(
        "**Technology:** TF-IDF + Cosine Similarity"
    )

    st.write(
        "**Framework:** Streamlit"
    )

    st.write(
        "**Dataset:** FAQ knowledge base"
    )


# ============================================================
# FILTER DATASET
# ============================================================

if selected_category == "All Categories":

    filtered_data = faq_data.copy()

else:

    filtered_data = faq_data[
        faq_data["Category"] == selected_category
    ].copy()


# ============================================================
# CHAT HISTORY DISPLAY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.subheader("💡 Suggested Questions")

suggestions = [
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "What is Deep Learning?",
    "What is Generative AI?",
    "What is Python?",
    "What is Natural Language Processing?",
]

suggestion_cols = st.columns(3)

for index, suggestion in enumerate(suggestions):

    with suggestion_cols[index % 3]:

        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True
        ):

            st.session_state["selected_question"] = suggestion


# ============================================================
# USER QUESTION
# ============================================================

selected_question = st.session_state.pop(
    "selected_question",
    None
)

question = st.chat_input(
    "💬 Ask your question..."
)

if selected_question:
    question = selected_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if question:

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.markdown(question)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # ----------------------------------------------------
        # SEARCH HISTORY
        # ----------------------------------------------------

        timestamp = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        st.session_state.search_history.append(
            {
                "question": question,
                "timestamp": timestamp
            }
        )

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        processed_question = preprocess_text(
            question
        )

        # ----------------------------------------------------
        # VECTORIZE
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Searching the FAQ knowledge base..."
        ):

            question_vector = (
                vectorizer.transform(
                    [processed_question]
                )
            )

            # ------------------------------------------------
            # SIMILARITY
            # ------------------------------------------------

            if selected_category == "All Categories":

                similarity_scores = (
                    cosine_similarity(
                        question_vector,
                        tfidf_matrix
                    )[0]
                )

                search_data = faq_data

            else:

                filtered_indices = (
                    filtered_data.index.tolist()
                )

                filtered_matrix = (
                    tfidf_matrix[
                        filtered_indices
                    ]
                )

                similarity_scores = (
                    cosine_similarity(
                        question_vector,
                        filtered_matrix
                    )[0]
                )

                search_data = filtered_data

            # ------------------------------------------------
            # BEST MATCH
            # ------------------------------------------------

            best_position = (
                similarity_scores.argmax()
            )

            best_score = float(
                similarity_scores[best_position]
            )

            best_row = search_data.iloc[
                best_position
            ]

            best_question = str(
                best_row["Question"]
            )

            best_answer = str(
                best_row["Answer"]
            )

            best_category = str(
                best_row["Category"]
            )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = best_score * 100

        # ----------------------------------------------------
        # BOT RESPONSE
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            if best_score < 0.15:

                st.warning(
                    "🤔 I couldn't find a highly relevant "
                    "answer in the FAQ database."
                )

                st.write(
                    "Try asking your question using different "
                    "words."
                )

                response_text = (
                    "I couldn't find a highly relevant "
                    "answer in the FAQ database."
                )

            else:

                if confidence >= 70:

                    confidence_label = "🟢 High Confidence"

                elif confidence >= 40:

                    confidence_label = "🟡 Medium Confidence"

                else:

                    confidence_label = "🔴 Low Confidence"

                st.success(
                    f"{confidence_label} — "
                    f"{confidence:.1f}%"
                )

                st.progress(
                    min(
                        confidence / 100,
                        1.0
                    )
                )

                st.markdown(
                    f"### 📂 {best_category}"
                )

                st.markdown(
                    f"**Matched Question:** "
                    f"{best_question}"
                )

                st.markdown(
                    f"""
                    <div class="answer-card">

                    <strong>🤖 Answer</strong>

                    <br><br>

                    {best_answer}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    f"🕒 {timestamp}"
                )

                response_text = best_answer

                # ------------------------------------------------
                # FAVORITE BUTTON
                # ------------------------------------------------

                if st.button(
                    "⭐ Add to Favorites",
                    key=f"favorite_{len(st.session_state.messages)}"
                ):

                    if (
                        best_question
                        not in st.session_state.favorite_questions
                    ):

                        st.session_state.favorite_questions.append(
                            best_question
                        )

                        st.success(
                            "Added to favorites!"
                        )

                # ------------------------------------------------
                # RELATED QUESTIONS
                # ------------------------------------------------

                st.markdown(
                    "### 🔗 Related Questions"
                )

                related_count = 0

                for i, score in enumerate(
                    similarity_scores
                ):

                    if i == best_position:
                        continue

                    if score > 0.05:

                        related_question = str(
                            search_data.iloc[i]["Question"]
                        )

                        st.write(
                            f"• {related_question}"
                        )

                        related_count += 1

                        if related_count >= 3:
                            break

        # ----------------------------------------------------
        # SAVE BOT MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text
            }
        )

        st.session_state.last_result = {
            "question": question,
            "answer": response_text,
            "category": best_category,
            "confidence": confidence,
            "timestamp": timestamp
        }


# ============================================================
# DOWNLOAD CHAT HISTORY
# ============================================================

st.divider()

st.subheader("📥 Downloads")

download_col1, download_col2 = st.columns(2)


# ------------------------------------------------------------
# CHAT DOWNLOAD
# ------------------------------------------------------------

with download_col1:

    if st.session_state.messages:

        chat_text = ""

        for message in st.session_state.messages:

            role = message["role"].upper()

            chat_text += (
                f"{role}:\n"
                f"{message['content']}\n\n"
            )

        st.download_button(
            "📥 Download Chat History",
            data=chat_text,
            file_name="faq_chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )

    else:

        st.caption(
            "No chat history available."
        )


# ------------------------------------------------------------
# FAQ DOWNLOAD
# ------------------------------------------------------------

with download_col2:

    faq_download = faq_data[
        [
            "Category",
            "Question",
            "Answer"
        ]
    ].to_csv(
        index=False
    )

    st.download_button(
        "📚 Download FAQ Dataset",
        data=faq_download,
        file_name="faq_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# SEARCH HISTORY
# ============================================================

if st.session_state.search_history:

    st.divider()

    st.subheader("🔎 Search History")

    history_df = pd.DataFrame(
        st.session_state.search_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ANALYTICS
# ============================================================

st.divider()

st.subheader("📊 FAQ Analytics")

analytics_col1, analytics_col2 = st.columns(2)


# ------------------------------------------------------------
# CATEGORY DISTRIBUTION
# ------------------------------------------------------------

with analytics_col1:

    category_counts = (
        faq_data["Category"]
        .value_counts()
    )

    st.write(
        "### 📂 Questions by Category"
    )

    st.bar_chart(
        category_counts
    )


# ------------------------------------------------------------
# FAQ TABLE
# ------------------------------------------------------------

with analytics_col2:

    st.write(
        "### 📋 FAQ Knowledge Base"
    )

    st.dataframe(
        faq_data[
            [
                "Category",
                "Question"
            ]
        ],
        use_container_width=True,
        hide_index=True,
        height=300
    )


# ============================================================
# RANDOM FAQ
# ============================================================

st.divider()

st.subheader("🎲 Random FAQ")

if st.button(
    "🎯 Show Random FAQ",
    use_container_width=True
):

    random_row = faq_data.iloc[
        random.randint(
            0,
            len(faq_data) - 1
        )
    ]

    st.info(
        f"**Question:** {random_row['Question']}"
    )

    st.success(
        f"**Answer:** {random_row['Answer']}"
    )

    st.caption(
        f"Category: {random_row['Category']}"
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.divider()

st.subheader("🟢 System Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.success("Dataset Loaded")

with status_col2:
    st.success("NLP Model Ready")

with status_col3:
    st.success("Chatbot Online")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#777;">

    🤖 <strong>AI FAQ Chatbot Pro</strong><br>

    Built for CodeAlpha Artificial Intelligence Internship<br>

    NLP • TF-IDF • Cosine Similarity • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
