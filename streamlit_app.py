# Entry point kept for the existing Streamlit Cloud deployment (main file = streamlit_app.py).
# The real app lives in app.py.
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "app.py"), run_name="__main__")
