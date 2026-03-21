import streamlit as st

st.set_page_config(page_title="SuperSmart", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page

if st.session_state.page == "home":
    st.title("SuperSmart")
    st.write("Choose a module")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SmartCount")
        if st.button("Open SmartCount", use_container_width=True):
            go_to("smartcount")
            st.rerun()

    with col2:
        st.subheader("SmartCast")
        if st.button("Open SmartCast", use_container_width=True):
            go_to("smartcast")
            st.rerun()

elif st.session_state.page == "smartcount":
    import pages.smartcount as smartcount
    smartcount.render(go_to)

elif st.session_state.page == "smartcast":
    import pages.smartcast as smartcast
    smartcast.render(go_to)