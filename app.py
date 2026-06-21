import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

st.set_page_config(
    page_title="Anime FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>

/* Animated Background */
.stApp{
    background: linear-gradient(-45deg,#0f0c29,#302b63,#24243e,#6a11cb);
    background-size:400% 400%;
    animation: gradient 15s ease infinite;
}

@keyframes gradient{
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}

/* Floating Icons */
.float{
    position:fixed;
    animation:float 10s linear infinite;
    z-index:0;
    opacity:0.7;
}

@keyframes float{
    from{
        transform:translateY(100vh);
    }
    to{
        transform:translateY(-100px);
    }
}

/* Title */
.title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:white;
    text-shadow:
    0 0 10px #ff00ff,
    0 0 20px #ff00ff,
    0 0 30px #8a2be2;
}

/* Subtitle */
.subtitle{
    text-align:center;
    color:#e0e0e0;
    margin-bottom:20px;
}

/* Glass Card */
.card{
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(12px);
    border-radius:20px;
    padding:20px;
    border:1px solid rgba(255,255,255,0.15);
}

/* Button */
.stButton>button{
    width:100%;
    height:50px;
    border-radius:15px;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
}

.stButton>button:hover{
    transform:scale(1.05);
}

footer{
    visibility:hidden;
}
</style>

<div class="float" style="left:10%;font-size:25px;">✨</div>
<div class="float" style="left:30%;font-size:22px;animation-delay:2s;">🤖</div>
<div class="float" style="left:50%;font-size:30px;animation-delay:4s;">⭐</div>
<div class="float" style="left:70%;font-size:24px;animation-delay:1s;">💜</div>
<div class="float" style="left:90%;font-size:28px;animation-delay:3s;">✨</div>

<div class="title">🤖 Anime FAQ Chatbot</div>
<div class="subtitle">
AI Powered FAQ Assistant ✨
</div>
""", unsafe_allow_html=True)

faq_data = {
    "What is CodeAlpha?":
        "CodeAlpha is a platform offering internships and project-based learning.",

    "How do I submit my task?":
        "Upload your source code to GitHub and submit the repository link.",

    "Will I get a certificate?":
        "Yes, eligible interns receive a completion certificate.",

    "How many tasks should I complete?":
        "You should complete any 2 or 3 tasks from your domain.",

    "Do I need to upload on LinkedIn?":
        "Yes, interns are encouraged to share their work on LinkedIn."
}

st.markdown('<div class="card">', unsafe_allow_html=True)

user_query = st.text_input(
    "💬 Ask your question"
)

if st.button("🚀 Send"):

    if user_query.strip():

        questions = list(faq_data.keys())

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            questions + [user_query]
        )

        similarity = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )

        best_match = similarity.argmax()

        st.chat_message("user").write(user_query)

        st.chat_message("assistant").write(
            faq_data[questions[best_match]]
        )

    else:
        st.warning("Please enter a question.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <center>
    <br>
    <h4 style='color:white;'>
    🌸 Made by Aarthi PS | CodeAlpha AI Internship
    </h4>
    </center>
    """,
    unsafe_allow_html=True
)