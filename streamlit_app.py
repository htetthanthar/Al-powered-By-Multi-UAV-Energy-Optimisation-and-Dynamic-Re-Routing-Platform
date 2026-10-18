# streamlit_app.py — Root entrypoint for Streamlit Community Cloud (share.streamlit.io)
import pathlib
import runpy

DASHBOARD_PATH = pathlib.Path(__file__).resolve().parent / "All sensor" / "dashboard.py"
runpy.run_path(str(DASHBOARD_PATH), run_name="__main__")
