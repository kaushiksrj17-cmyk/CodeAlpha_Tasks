import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
import pyperclip
import os

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="AI Language Translator",
    page_icon="🌍",
    layout="wide"
)

# ======================================================
# SESSION STATE
# ======================================================

if "source_lang" not in st.session_state:
    st.session_state.source_lang = "Auto Detect"

if "target_lang" not in st.session_state:
    st.session_state.target_lang = "Tamil"

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""

if "history" not in st.session_state:
    st.session_state.history = []

if "favorites" not in st.session_state:
    st.session_state.favorites = []

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("⚙️ Settings")

dark_mode = st.sidebar.toggle("🌙 Dark Mode")

# ======================================================
# DARK MODE CSS
# ======================================================

if dark_mode:

    st.markdown(
        """
        <style>

        .stApp{
            background-color:#0E1117;
            color:white;
        }

        textarea{
            background:#262730 !important;
            color:white !important;
        }

        div[data-baseweb="select"]{
            color:black;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# ======================================================
# TITLE
# ======================================================

st.title("🌍 AI Language Translator")

st.markdown(
"""
Translate text instantly between **100+ languages**
using **Google Translate**.
"""
)

st.divider()

# ======================================================
# LANGUAGE DICTIONARY
# ======================================================

languages = {

    "Auto Detect":"auto",

    "English":"en",

    "Tamil":"ta",

    "Hindi":"hi",

    "French":"fr",

    "German":"de",

    "Spanish":"es",

    "Japanese":"ja",

    "Korean":"ko",

    "Chinese":"zh-CN",

    "Arabic":"ar",

    "Russian":"ru",

    "Italian":"it",

    "Portuguese":"pt",

    "Malayalam":"ml",

    "Kannada":"kn",

    "Telugu":"te",

    "Bengali":"bn",

    "Gujarati":"gu",

    "Punjabi":"pa",

    "Urdu":"ur"

}
# ======================================================
# USER INPUT
# ======================================================

st.subheader("✍️ Enter Text")

text = st.text_area(
    "Type text to translate",
    value=st.session_state.input_text,
    height=180,
    placeholder="Example: Hello, how are you?"
)

st.session_state.input_text = text

# ======================================================
# CHARACTER & WORD COUNT
# ======================================================

char_count = len(text)

word_count = len(text.split()) if text.strip() else 0

metric1, metric2 = st.columns(2)

with metric1:
    st.metric("Characters", char_count)

with metric2:
    st.metric("Words", word_count)

st.divider()

# ======================================================
# LANGUAGE SELECTION
# ======================================================

st.subheader("🌐 Select Languages")

col1, col2 = st.columns(2)

language_names = list(languages.keys())

with col1:

    source = st.selectbox(
        "Source Language",
        language_names,
        index=language_names.index(st.session_state.source_lang)
    )

with col2:

    target_languages = language_names[1:]  # Remove Auto Detect

    target = st.selectbox(
        "Target Language",
        target_languages,
        index=target_languages.index(st.session_state.target_lang)
    )

st.session_state.source_lang = source
st.session_state.target_lang = target

# ======================================================
# SWAP BUTTON
# ======================================================

swap_col1, swap_col2, swap_col3 = st.columns([1,1,1])

with swap_col2:

    if st.button("🔄 Swap Languages", use_container_width=True):

        if source == "Auto Detect":

            st.warning("Auto Detect cannot be swapped.")

        else:

            temp = st.session_state.source_lang

            st.session_state.source_lang = st.session_state.target_lang

            st.session_state.target_lang = temp

            st.rerun()

st.divider()
# ======================================================
# TRANSLATE SECTION
# ======================================================

st.subheader("🌍 Translate")

if st.button("🚀 Translate", use_container_width=True):

    # Validate Input
    if text.strip() == "":
        st.warning("Please enter some text to translate.")

    elif source == target:
        st.warning("Source and Target languages cannot be the same.")

    else:

        try:

            translated = GoogleTranslator(
                source=languages[source],
                target=languages[target]
            ).translate(text)

            # Save Translation
            st.session_state.translated_text = translated

            # Save to History
            st.session_state.history.append({

                "source": source,

                "target": target,

                "original": text,

                "translated": translated

            })

            st.success("✅ Translation Successful!")

        except Exception as e:

            st.error("Translation Failed!")

            st.error(str(e))

# ======================================================
# DISPLAY TRANSLATION
# ======================================================

if st.session_state.translated_text != "":

    st.subheader("📄 Translated Text")

    st.text_area(

        "Translation",

        value=st.session_state.translated_text,

        height=180,

        disabled=True

    )

st.divider()
# ======================================================
# EXTRA FEATURES
# ======================================================

if st.session_state.translated_text != "":

    st.subheader("🛠 Translation Tools")

    col1, col2, col3, col4, col5 = st.columns(5)

    # --------------------------------------------------
    # COPY
    # --------------------------------------------------

    with col1:

        if st.button("📋 Copy"):

            pyperclip.copy(st.session_state.translated_text)

            st.success("Copied to Clipboard!")

    # --------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------

    with col2:

        st.download_button(

            label="📥 Download",

            data=st.session_state.translated_text,

            file_name="translation.txt",

            mime="text/plain"

        )

    # --------------------------------------------------
    # TEXT TO SPEECH
    # --------------------------------------------------

    with col3:

        if st.button("🔊 Listen"):

            try:

                target_code = languages[st.session_state.target_lang]

                # gTTS doesn't support zh-CN directly
                if target_code == "zh-CN":
                    target_code = "zh"

                tts = gTTS(
                    text=st.session_state.translated_text,
                    lang=target_code
                )

                filename = "translation.mp3"

                tts.save(filename)

                with open(filename, "rb") as audio_file:

                    st.audio(audio_file.read())

                os.remove(filename)

            except Exception as e:

                st.error(f"Text-to-Speech Error: {e}")

    # --------------------------------------------------
    # FAVORITES
    # --------------------------------------------------

    with col4:

        if st.button("⭐ Favorite"):

            st.session_state.favorites.append({

                "original": st.session_state.input_text,

                "translated": st.session_state.translated_text,

                "source": st.session_state.source_lang,

                "target": st.session_state.target_lang

            })

            st.success("Added to Favorites!")

    # --------------------------------------------------
    # CLEAR
    # --------------------------------------------------

    with col5:

        if st.button("🗑 Clear"):

            st.session_state.input_text = ""

            st.session_state.translated_text = ""

            st.rerun()

st.divider()
# ======================================================
# SIDEBAR - TRANSLATION HISTORY
# ======================================================

st.sidebar.divider()
st.sidebar.header("📜 Translation History")

if len(st.session_state.history) == 0:

    st.sidebar.info("No translations yet.")

else:

    for i, item in enumerate(reversed(st.session_state.history), start=1):

        st.sidebar.markdown(f"""
### {i}.

**{item['source']} ➜ {item['target']}**

**Original:**

{item['original']}

**Translated:**

{item['translated']}

---
""")

# ======================================================
# CLEAR HISTORY
# ======================================================

if st.sidebar.button("🗑 Clear History"):

    st.session_state.history = []

    st.success("History Cleared!")

    st.rerun()

# ======================================================
# FAVORITES
# ======================================================

st.sidebar.divider()

st.sidebar.header("⭐ Favorite Translations")

if len(st.session_state.favorites) == 0:

    st.sidebar.info("No favorites added.")

else:

    for i, fav in enumerate(st.session_state.favorites, start=1):

        st.sidebar.markdown(f"""
### ⭐ {i}

**Original:**

{fav['original']}

**Translated:**

{fav['translated']}

---
""")

# ======================================================
# CLEAR FAVORITES
# ======================================================

if st.sidebar.button("❌ Clear Favorites"):

    st.session_state.favorites = []

    st.success("Favorites Cleared!")

    st.rerun()

# ======================================================
# ABOUT
# ======================================================

st.sidebar.divider()

st.sidebar.header("ℹ About")

st.sidebar.write("""
**AI Language Translator**

Built using:

- Streamlit
- Google Translator API
- gTTS
- Pyperclip

Features:

✅ Auto Detect Language

✅ 20+ Languages

✅ Translation History

✅ Favorites

✅ Copy Translation

✅ Download Translation

✅ Text-to-Speech

✅ Dark Mode

✅ Character & Word Count

Developed using Python.
""")

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.caption("🌍 AI Language Translator | Version 2.0")
st.caption("Built with ❤️ using Streamlit")