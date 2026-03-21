import streamlit as st

def render(go_to):
    st.title("SmartCount")

    if st.button("← Back"):
        go_to("home")
        st.rerun()

    option = st.radio(
        "Select input type",
        ["Upload Image", "Upload Video", "Webcam Live"],
        horizontal=True
    )

    if option == "Upload Image":
        st.file_uploader("Upload image")

    elif option == "Upload Video":
        st.file_uploader("Upload video")

    elif option == "Webcam Live":
        st.write("Webcam mode")