import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

SAMPLES_DIR = BASE_DIR / "samples"
WORK_DIR = BASE_DIR / "work"
MASTER_STORE = WORK_DIR / "master_store"
LETTER_CROPS = WORK_DIR / "letter_crops"
OUTPUT_DIR = BASE_DIR / "output"
for _d in (MASTER_STORE, LETTER_CROPS, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RENDER_DPI = int(os.getenv("RENDER_DPI", "250"))
# Scores >= MATCH_THRESHOLD are MATCH; scores within REVIEW_BAND below it need manual review.
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.60"))
REVIEW_BAND = float(os.getenv("REVIEW_BAND", "0.05"))

OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:3b")
OLLAMA_AGENT_MODEL = os.getenv("OLLAMA_AGENT_MODEL", "qwen2.5vl:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
