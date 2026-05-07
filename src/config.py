import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "evaluation_input.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "evaluation_result.csv")

MAX_WORKERS = 10

ARK_API_KEY = os.getenv("ARK_API_KEY", "xxxxxxxxx")
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "xxxxxxxxx")
ARK_MODEL_ID = os.getenv("ARK_MODEL_ID", "xxxxxxxxx")
TEMPERATURE = 0
MAX_TOKENS = None
