import streamlit as st

def render(go_to):
    st.title("Smartcast Page 🎬")

    st.write("Welcome to the Smartcast project!")

    # Example content

    # Back button (important for navigation)
    if st.button("⬅ Back to Home"):
        go_to("home")
        st.rerun()