import os
import base64
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="SuperSmart",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to(page: str):
    st.session_state.page = page


LOGO_PATH = "logo.png"


def get_logo_base64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


logo_base64 = get_logo_base64(LOGO_PATH)

st.markdown(
    """
    <style>
    .main {
        background-color: #F4F8FB;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .section-title {
        color: #0B2A4A;
        font-size: 1.25rem;
        font-weight: 800;
        margin: 8px 0 14px 0;
    }

    .card {
        background: white;
        border: 1px solid #CBD5E1;
        border-radius: 22px;
        padding: 26px;
        min-height: 260px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    }

    .card-title {
        color: #0B2A4A;
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .card-desc {
        color: #475569;
        font-size: 1rem;
        line-height: 1.65;
        margin-bottom: 16px;
    }

    .pill {
        display: inline-block;
        background: #EEF5FB;
        color: #2F6FA3;
        border-radius: 999px;
        padding: 6px 12px;
        margin-right: 6px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        margin-top: 26px;
        color: #64748B;
        font-size: 0.95rem;
    }

    .stButton > button {
        background-color: #2F6FA3;
        color: white;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.75rem 1rem;
        border: none;
    }

    .stButton > button:hover {
        background-color: #0B2A4A;
        color: white;
    }

    .stButton > button:focus:not(:active) {
        color: white;
        border-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.page == "home":
    hero_logo = ""
    if logo_base64:
        hero_logo = f'<img src="data:image/png;base64,{logo_base64}" alt="SuperSmart logo">'

    hero_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body {{
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: transparent;
        }}

        .hero-card {{
          background: linear-gradient(135deg, #0B2A4A 0%, #2F6FA3 100%);
          border-radius: 26px;
          padding: 34px 38px;
          box-shadow: 0 12px 30px rgba(0,0,0,0.14);
          color: white;
        }}

        .hero-flex {{
          display: flex;
          align-items: center;
          gap: 28px;
        }}

        .hero-logo-wrap {{
          flex: 0 0 auto;
          padding: 14px;
          border-radius: 22px;
          background: rgba(255,255,255,0.18);
          border: 1px solid rgba(255,255,255,0.22);
          backdrop-filter: blur(8px);
          box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        }}

        .hero-logo-wrap img {{
          width: 120px;
          height: auto;
          display: block;
        }}

        .hero-text {{
          flex: 1;
        }}

        .hero-title {{
          font-size: 3rem;
          font-weight: 800;
          line-height: 1.05;
          margin: 0 0 10px 0;
        }}

        .hero-sub {{
          font-size: 1.1rem;
          line-height: 1.6;
          color: rgba(255,255,255,0.92);
          margin: 0;
        }}

        @media (max-width: 768px) {{
          .hero-flex {{
            flex-direction: column;
            align-items: flex-start;
          }}

          .hero-logo-wrap img {{
            width: 95px;
          }}

          .hero-title {{
            font-size: 2.2rem;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="hero-card">
        <div class="hero-flex">
          <div class="hero-logo-wrap">
            {hero_logo}
          </div>
          <div class="hero-text">
            <div class="hero-title">SuperSmart</div>
            <p class="hero-sub">
              A smart retail intelligence platform for automated counting, monitoring, and visual analysis using AI.
            </p>
          </div>
        </div>
      </div>
    </body>
    </html>
    """

    components.html(hero_html, height=210, scrolling=False)

    st.markdown('<div class="section-title">Choose a Module</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">📦 SmartCount</div>
                <div class="card-desc">
                    Detect and count products from images, videos, or live webcam.
                </div>
                <div>
                    <span class="pill">Image</span>
                    <span class="pill">Video</span>
                    <span class="pill">Webcam</span>
                    <span class="pill">Alerts</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open SmartCount", use_container_width=True):
            go_to("smartcount")
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">📺 SmartCast</div>
                <div class="card-desc">
                    Demand forecasting using AI and time-series insights.
                </div>
                <div>
                    <span class="pill">Live</span>
                    <span class="pill">AI</span>
                    <span class="pill">Insights</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open SmartCast", use_container_width=True):
            go_to("smartcast")
            st.rerun()

    st.markdown(
        '<div class="footer">SuperSmart • AI Visual Intelligence</div>',
        unsafe_allow_html=True,
    )

elif st.session_state.page == "smartcount":
    import pages.smartcount as smartcount
    smartcount.render(go_to)

elif st.session_state.page == "smartcast":
    import pages.smartcast as smartcast
    smartcast.render(go_to)