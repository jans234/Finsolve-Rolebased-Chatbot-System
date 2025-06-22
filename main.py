import streamlit as st
import requests

API = "http://127.0.0.1:8000"  # FastAPI base URL

# ── Session init ─────────────────────────────────────────────
if "session" not in st.session_state:
    st.session_state.session = requests.Session()
    st.session_state.logged = False
    st.session_state.role = ""
    st.session_state.user = ""

st.set_page_config(page_title="FinSolve Chatbot", page_icon="🤖")
st.title("🔐 FinSolve Role Based Chatbot")

# ── Login block ──────────────────────────────────────────────
if not st.session_state.logged:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            resp = st.session_state.session.post(
                f"{API}/login", data={"username": username, "password": password}
            )
            if resp.status_code == 200:
                st.success("✅ Login successful!")
                st.session_state.logged = True
                st.session_state.role = resp.json()["role"]
                st.session_state.user = username
                st.experimental_rerun()
            else:
                st.error("❌ Invalid credentials")

# ── Chat block ───────────────────────────────────────────────
if st.session_state.logged:
    st.info(f"Logged in as `{st.session_state.user}`  |  Role: **{st.session_state.role}**")

    query = st.text_area("Ask a question", height=120, key="query")
    if st.button("Get Answer", key="ask_btn") and query.strip():
        with st.spinner("Thinking…"):
            r = st.session_state.session.post(f"{API}/ask", data={"query": query})
        if r.status_code == 200:
            answer = r.json()["answer"]
            st.markdown("#### 💬 Answer")
            st.write(answer)
        elif r.status_code == 401:
            st.error("🔒 Session expired – please log in again.")
            st.session_state.logged = False
            st.experimental_rerun()
        else:
            st.error(f"⚠️ Error: {r.json().get('detail', r.status_code)}")

    if st.button("Logout 🔒", key="logout_btn"):
        st.session_state.session.post(f"{API}/logout")
        st.session_state.clear()
        st.experimental_rerun()

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "© 2025 Made by <b>Muhammad Afaq</b>"
    "</div>",
    unsafe_allow_html=True
)
