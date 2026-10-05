# Entry point kept for the existing Streamlit Cloud deployment (main file = streamlit_app.py).
# The real app lives in app.py.
import runpy

runpy.run_path("app.py", run_name="__main__")
