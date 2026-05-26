from pathlib import Path
import runpy


if __name__ == "__main__":
    dashboard_path = Path(__file__).parent / "app" / "app.py"
    runpy.run_path(str(dashboard_path), run_name="__main__")
