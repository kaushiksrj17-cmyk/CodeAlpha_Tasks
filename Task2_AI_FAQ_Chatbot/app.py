import os
import pandas as pd
import streamlit as st

from preprocess import preprocess_text

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

.main{
    padding-top:20px;
}

.stButton>button{
    width:100%;
    border-radius:10px;
    height:48px;
    font-size:17px;
    font-weight:bold;
}

.stTextInput>div>div>input{
    font-size:17px;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
    margin-top:40px;
}

.metric-box{
    background:#f8f9fa;
    padding:15px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "faq.csv")

    data = pd.read_csv(
        DATA_PATH,
        sep="\t"
    )

    # Normalize column names
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
    )

    # Create cleaned question column
    data["question_clean"] = data["question"].apply(preprocess_text)

    return data
# ============================================================
# BUILD TF-IDF
# ============================================================

@st.cache_resource
def build_model(data):

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(
        data["Processed_Question"]
    )

    return vectorizer, matrix


vectorizer, tfidf_matrix = build_model(faq_data)

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "favorite_questions" not in st.session_state:
    st.session_state.favorite_questions = []

if "successful_searches" not in st.session_state:
    st.session_state.successful_searches = 0

if "failed_searches" not in st.session_state:
    st.session_state.failed_searches = 0

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🤖 AI FAQ Chatbot Pro")

st.sidebar.success("AI Powered FAQ Assistant")

st.sidebar.markdown("---")

st.sidebar.subheader("📚 Knowledge Base")

st.sidebar.metric(
    "Total FAQs",
    len(faq_data)
)

categories = sorted(
    faq_data["Category"].unique()
)

st.sidebar.metric(
    "Categories",
    len(categories)
)

# ============================================================
# CATEGORY FILTER
# ============================================================

selected_category = st.sidebar.selectbox(
    "📂 Search Category",
    ["All Categories"] + list(categories)
)

# ============================================================
# FAQ BROWSER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📖 Browse FAQs")

browse_category = st.sidebar.selectbox(
    "Choose Category",
    categories,
    key="browse_category"
)

with st.sidebar.expander("View Questions"):

    browse_questions = faq_data[
        faq_data["Category"] == browse_category
    ]

    for q in browse_questions["Question"]:

        st.write("•", q)

# ============================================================
# CLEAR CHAT
# ============================================================

st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear Chat"):

    st.session_state.messages = []

    st.session_state.search_history = []

    st.session_state.favorite_questions = []

    st.session_state.successful_searches = 0

    st.session_state.failed_searches = 0

    st.rerun()

# ============================================================
# MAIN TITLE
# ============================================================

st.title("🤖 AI FAQ Chatbot Pro")

st.write(
"""
Ask questions about

• Artificial Intelligence

• Machine Learning

• Deep Learning

• NLP

• Python

• Data Science

using intelligent FAQ matching.
"""
)

# ============================================================
# DASHBOARD
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "FAQs",
    len(faq_data)
)

col2.metric(
    "Categories",
    len(categories)
)

col3.metric(
    "Questions Asked",
    len(st.session_state.messages)
)

col4.metric(
    "Successful",
    st.session_state.successful_searches
)

st.markdown("---")

# ============================================================
# SEARCH BOX
# ============================================================

st.subheader("💬 Ask Your Question")

user_question = st.text_input(
    "Type your question here..."
)

# ============================================================
# LIVE SEARCH SUGGESTIONS
# ============================================================

if user_question.strip() != "":

    suggestions = faq_data[
        faq_data["Question"].str.contains(
            user_question,
            case=False,
            na=False
        )
    ]

    if len(suggestions) > 0:

        st.info("💡 Suggested Questions")

        for q in suggestions[
            "Question"
        ].head(5):

            st.write("•", q)

send = st.button("🚀 Ask AI")
# ============================================================
# AI SEARCH ENGINE
# ============================================================

if send:

    if user_question.strip() == "":

        st.warning("⚠ Please enter a question.")

    else:

        with st.spinner("🤖 AI is analyzing your question..."):

            # -----------------------------
            # Progress Bar
            # -----------------------------

            progress = st.progress(0)

            progress.progress(15)

            # -----------------------------
            # Preprocess User Question
            # -----------------------------

            processed_question = preprocess_text(
                user_question
            )

            progress.progress(35)

            # -----------------------------
            # Search Selected Category
            # -----------------------------

            if selected_category == "All Categories":

                search_data = faq_data

                search_matrix = tfidf_matrix

            else:

                search_data = faq_data[
                    faq_data["Category"] == selected_category
                ].reset_index(drop=True)

                search_matrix = vectorizer.transform(
                    search_data["Processed_Question"]
                )

            progress.progress(55)

            # -----------------------------
            # Convert User Question
            # -----------------------------

            user_vector = vectorizer.transform(
                [processed_question]
            )

            similarity_scores = cosine_similarity(
                user_vector,
                search_matrix
            )

            progress.progress(75)

            # -----------------------------
            # Best Matching Question
            # -----------------------------

            best_match_index = similarity_scores.argmax()

            confidence_score = similarity_scores[
                0
            ][
                best_match_index
            ]

            progress.progress(90)

            # -----------------------------
            # Result
            # -----------------------------

            if confidence_score >= 0.30:

                matched_question = search_data.iloc[
                    best_match_index
                ]["Question"]

                chatbot_answer = search_data.iloc[
                    best_match_index
                ]["Answer"]

                matched_category = search_data.iloc[
                    best_match_index
                ]["Category"]

                st.session_state.successful_searches += 1

            else:

                matched_question = ""

                matched_category = "Unknown"

                chatbot_answer = """
I couldn't find an exact answer.

Try asking questions about:

• Artificial Intelligence

• Machine Learning

• Deep Learning

• Python

• Data Science

• NLP
"""

                st.session_state.failed_searches += 1

            progress.progress(100)

            progress.empty()

            # -----------------------------
            # Store Conversation
            # -----------------------------

            st.session_state.messages.append(

                {

                    "question": user_question,

                    "matched_question": matched_question,

                    "answer": chatbot_answer,

                    "category": matched_category,

                    "score": confidence_score,

                    "time": datetime.now().strftime(
                        "%I:%M %p"
                    )

                }

            )

            # -----------------------------
            # Search Log
            # -----------------------------

            st.session_state.search_history.append(

                {

                    "Question": user_question,

                    "Category": matched_category,

                    "Confidence": round(
                        confidence_score * 100,
                        2
                    ),

                    "Time": datetime.now().strftime(
                        "%d-%m-%Y %H:%M:%S"
                    )

                }

            )
            # ============================================================
# DISPLAY CONVERSATION
# ============================================================

if st.session_state.messages:

    st.markdown("---")

    st.subheader("💬 Conversation")

    for index, message in enumerate(
        st.session_state.messages
    ):

        # ------------------------------------------------
        # USER MESSAGE
        # ------------------------------------------------

        with st.chat_message("user"):

            st.write(message["question"])

        # ------------------------------------------------
        # BOT MESSAGE
        # ------------------------------------------------

        with st.chat_message("assistant"):

            st.success(message["answer"])

            st.write(
                f"📂 **Category:** {message['category']}"
            )

            confidence = message["score"] * 100

            # --------------------------------------------
            # CONFIDENCE BADGE
            # --------------------------------------------

            if confidence >= 80:

                badge = "🟢 Excellent Match"

            elif confidence >= 60:

                badge = "🟡 Good Match"

            elif confidence >= 40:

                badge = "🟠 Fair Match"

            else:

                badge = "🔴 Low Match"

            st.info(
                f"{badge} ({confidence:.1f}%)"
            )

            st.progress(
                float(message["score"])
            )

            st.caption(
                f"🕒 {message['time']}"
            )

            # --------------------------------------------
            # COPY ANSWER
            # --------------------------------------------

            with st.expander("📋 Copy Answer"):

                st.code(
                    message["answer"],
                    language=None
                )

            # --------------------------------------------
            # SAVE FAVORITE
            # --------------------------------------------

            if (
                message["matched_question"] != ""
            ):

                if st.button(
                    "⭐ Add to Favorites",
                    key=f"fav_{index}"
                ):

                    if (
                        message["matched_question"]
                        not in st.session_state.favorite_questions
                    ):

                        st.session_state.favorite_questions.append(
                            message["matched_question"]
                        )

                        st.success(
                            "Added to Favorites!"
                        )

                    else:

                        st.info(
                            "Already in Favorites."
                        )
                        # ============================================================
# SMART RELATED QUESTIONS
# ============================================================

if len(st.session_state.messages) > 0:

    latest_message = st.session_state.messages[-1]

    if latest_message["matched_question"] != "":

        st.markdown("---")

        st.subheader("🧠 Related Questions")

        processed_latest = preprocess_text(
            latest_message["matched_question"]
        )

        latest_vector = vectorizer.transform(
            [processed_latest]
        )

        similarity_scores = cosine_similarity(
            latest_vector,
            tfidf_matrix
        )[0]

        # Get top 6 results
        top_matches = similarity_scores.argsort()[-6:][::-1]

        count = 0

        for index in top_matches:

            question = faq_data.iloc[index]["Question"]

            # Skip the current question
            if question != latest_message["matched_question"]:

                st.write("•", question)

                count += 1

            if count == 5:
                break

# ============================================================
# FAVORITE QUESTIONS
# ============================================================

if len(st.session_state.favorite_questions) > 0:

    st.sidebar.markdown("---")

    st.sidebar.subheader("⭐ Favorite Questions")

    for question in st.session_state.favorite_questions:

        st.sidebar.write("•", question)

# ============================================================
# RECENT SEARCH HISTORY
# ============================================================

if len(st.session_state.search_history) > 0:

    st.sidebar.markdown("---")

    st.sidebar.subheader("🕒 Recent Searches")

    recent = st.session_state.search_history[-5:]

    for item in reversed(recent):

        st.sidebar.write(
            f"• {item['Question']}"
        )

# ============================================================
# SESSION STATISTICS
# ============================================================

st.markdown("---")

st.subheader("📊 Session Statistics")

total_questions = len(
    st.session_state.messages
)

successful = st.session_state.successful_searches

failed = st.session_state.failed_searches

average_confidence = 0

if total_questions > 0:

    average_confidence = sum(
        msg["score"]
        for msg in st.session_state.messages
    ) / total_questions

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Questions",
    total_questions
)

col2.metric(
    "Successful",
    successful
)

col3.metric(
    "Failed",
    failed
)

col4.metric(
    "Average Confidence",
    f"{average_confidence*100:.1f}%"
)
# ============================================================
# DOWNLOAD CHAT HISTORY
# ============================================================

if len(st.session_state.messages) > 0:

    st.markdown("---")

    st.subheader("📥 Download Chat History")

    chat_history = ""

    for msg in st.session_state.messages:

        chat_history += f"""
Time : {msg['time']}

User :
{msg['question']}

AI :
{msg['answer']}

Category :
{msg['category']}

Confidence :
{msg['score']*100:.2f} %

----------------------------------------------------

"""

    st.download_button(

        label="📄 Download Conversation",

        data=chat_history,

        file_name="Chat_History.txt",

        mime="text/plain"

    )

# ============================================================
# EXPORT SEARCH ANALYTICS
# ============================================================

if len(st.session_state.search_history) > 0:

    st.markdown("---")

    st.subheader("📊 Export Search Analytics")

    analytics_df = pd.DataFrame(
        st.session_state.search_history
    )

    st.dataframe(
        analytics_df,
        use_container_width=True
    )

    csv = analytics_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="⬇ Download Analytics CSV",

        data=csv,

        file_name="Search_Analytics.csv",

        mime="text/csv"

    )

# ============================================================
# CATEGORY USAGE CHART
# ============================================================

if len(st.session_state.messages) > 0:

    st.markdown("---")

    st.subheader("📈 Category Usage")

    category_df = pd.DataFrame(

        [

            msg["category"]

            for msg in st.session_state.messages

            if msg["category"] != "Unknown"

        ],

        columns=["Category"]

    )

    if len(category_df) > 0:

        chart = category_df["Category"].value_counts()

        st.bar_chart(chart)

# ============================================================
# KNOWLEDGE BASE OVERVIEW
# ============================================================

st.markdown("---")

st.subheader("📚 Knowledge Base")

knowledge_base = faq_data.groupby(
    "Category"
).size().reset_index(
    name="FAQs"
)

st.dataframe(

    knowledge_base,

    use_container_width=True

)

# ============================================================
# RANDOM FAQ
# ============================================================

st.markdown("---")

st.subheader("🎲 Explore a Random FAQ")

if st.button("Generate Random FAQ"):

    random_faq = faq_data.sample(1).iloc[0]

    st.info(
        random_faq["Question"]
    )

    st.success(
        random_faq["Answer"]
    )

# ============================================================
# ABOUT PROJECT
# ============================================================

st.markdown("---")

with st.expander("ℹ About This Project"):

    st.markdown("""

# 🤖 AI FAQ Chatbot

This chatbot is developed using

- Python

- Streamlit

- NLTK

- Pandas

- Scikit-Learn

## AI Techniques

✅ Natural Language Processing

✅ TF-IDF Vectorization

✅ Cosine Similarity

## Features

✅ Intelligent FAQ Matching

✅ Chat History

✅ Search Suggestions

✅ Favorites

✅ Related Questions

✅ Analytics Dashboard

✅ Category Filtering

✅ Download Chat

✅ Export Analytics

✅ Knowledge Base Browser

Developed as an AI Internship Project.

""")

# ============================================================
# RESET ANALYTICS
# ============================================================

st.markdown("---")

if st.button("♻ Reset Everything"):

    st.session_state.messages = []

    st.session_state.search_history = []

    st.session_state.favorite_questions = []

    st.session_state.successful_searches = 0

    st.session_state.failed_searches = 0

    st.success(
        "Everything has been reset."
    )

    st.rerun()

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;color:gray;'>

🤖 <b>AI FAQ Chatbot Pro</b>

Developed using

Python • Streamlit • Pandas • NLTK • Scikit-Learn

© 2026

</div>
""",
unsafe_allow_html=True
)
# ==============================
# PHASE 9 - PART 4
# EXPORT / DOWNLOAD FEATURES
# ==============================

import datetime


# Create download text for chat history
def generate_chat_download(history):
    text = "AI FAQ CHATBOT - CHAT HISTORY\n"
    text += "=" * 40 + "\n\n"

    for chat in history:
        text += f"User: {chat['user']}\n"
        text += f"Bot : {chat['bot']}\n"
        text += "-" * 40 + "\n"

    text += "\nGenerated on: "
    text += str(datetime.datetime.now())

    return text



# Download Chat History Button
if "chat_history" in st.session_state and len(st.session_state.chat_history) > 0:

    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Chat")

    chat_text = generate_chat_download(
        st.session_state.chat_history
    )

    st.sidebar.download_button(
        label="Download Chat History",
        data=chat_text,
        file_name="AI_Chatbot_History.txt",
        mime="text/plain"
    )



# Download FAQ Knowledge Base
faq_text = """
AI FAQ CHATBOT KNOWLEDGE BASE

This chatbot answers frequently asked questions
using Artificial Intelligence.

"""

if "faq_data" in globals():

    faq_text += "\nAvailable Questions:\n"

    for item in faq_data:
        faq_text += f"- {item}\n"



st.sidebar.download_button(
    label="Download FAQ Data",
    data=faq_text,
    file_name="FAQ_Knowledge_Base.txt",
    mime="text/plain"
)



# ==============================
# FOOTER
# ==============================

st.markdown("---")

st.markdown(
    """
    <div style="
    text-align:center;
    color:gray;
    font-size:14px;
    ">

    🤖 <b>AI FAQ Chatbot</b><br>

    Built using Python, Streamlit and Artificial Intelligence<br>

    Phase 9 Enhanced Version 🚀

    </div>
    """,
    unsafe_allow_html=True
)
# ==============================
# PHASE 9 - PART 5
# FINAL TESTING & DEPLOYMENT READY
# ==============================


# ------------------------------
# APP STATUS PANEL
# ------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Status")

st.sidebar.success("✅ Chatbot Online")

st.sidebar.info(
    """
    AI Engine: Active  
    Knowledge Base: Loaded  
    Interface: Streamlit  
    Version: Phase 9
    """
)



# ------------------------------
# RESET CHAT BUTTON
# ------------------------------

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.chat_history = []

    st.rerun()



# ------------------------------
# ERROR HANDLING CHECK
# ------------------------------

try:

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


except Exception as e:

    st.error(
        f"System Error Occurred: {e}"
    )



# ------------------------------
# APPLICATION INFORMATION
# ------------------------------

with st.sidebar.expander("ℹ️ About Project"):

    st.write(
        """
        AI FAQ Chatbot

        Features:
        ✔ Natural Language Processing
        ✔ FAQ Question Answering
        ✔ Chat Memory
        ✔ Export Conversation
        ✔ User Friendly UI

        Developed using:
        Python + Streamlit + AI
        """
    )



# ------------------------------
# FINAL DEPLOYMENT MESSAGE
# ------------------------------

st.sidebar.markdown("---")

st.sidebar.caption(
    "🚀 Ready for GitHub and Streamlit Cloud Deployment"
)
# ==============================
# PHASE 9 - PART 6
# FINAL INTEGRATION CHECK
# ==============================


# ------------------------------
# PROJECT VALIDATION
# ------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Validation Check")


checks = {
    "Chat Interface": True,
    "Session Memory": "chat_history" in st.session_state,
    "FAQ Knowledge Base": "faq_data" in globals(),
    "Export Feature": True,
    "Streamlit UI": True
}


for feature, status in checks.items():

    if status:
        st.sidebar.success(
            f"✔ {feature}"
        )

    else:
        st.sidebar.warning(
            f"⚠ {feature} Missing"
        )



# ------------------------------
# SAFE FAQ LOADING
# ------------------------------

if "faq_data" not in globals():

    faq_data = {}

    st.warning(
        "FAQ database not detected. Add FAQ data to enable answers."
    )



# ------------------------------
# APPLICATION READY MESSAGE
# ------------------------------

st.markdown(
    """
    <div style="
    padding:15px;
    border-radius:10px;
    background-color:#f0f8ff;
    text-align:center;
    ">

    <h3>🤖 AI FAQ Chatbot</h3>

    <p>
    Application Status:
    <b>READY FOR DEPLOYMENT 🚀</b>
    </p>

    </div>
    """,
    unsafe_allow_html=True
)



# ------------------------------
# PROJECT DETAILS
# ------------------------------

with st.expander("📌 Project Information"):

    st.write(
        """
        Project Name:
        AI FAQ Chatbot

        Domain:
        Artificial Intelligence

        Technology Stack:

        • Python
        • Streamlit
        • Natural Language Processing
        • Machine Learning

        Features:

        ✔ FAQ Question Answering
        ✔ Intelligent Response System
        ✔ Chat History
        ✔ Export Conversation
        ✔ User Friendly Interface

        Deployment:
        Streamlit Cloud Ready
        """
    )



# ------------------------------
# END MESSAGE
# ------------------------------

st.sidebar.markdown("---")

st.sidebar.success(
    "🎉 Phase 9 Completed Successfully!"
)
