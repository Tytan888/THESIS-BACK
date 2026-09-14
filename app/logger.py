import json
import sys
import os
from enum import Enum

class ProcessingStep(Enum):
    def __new__(cls, msg):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        obj.msg = msg
        return obj

    JUST_RECEIVED = "Just received your song..."
    SCRAPING_BASELINE = "Searching for the original lyrics..."
    SCRAPING_TRANSLATION = "Finding potential translations..."
    UNIFYING_LYRICS = "Syncing translations with the original lyrics..."
    CLEANING_TEXT = "Cleaning and formatting the text..."
    ALIGNING_TOKENS = "Finding vocabulary matches between Japanese and English..."
    FILTERING_POS = "Filtering and verifying vocabulary (1/3)..."
    FILTERING_LABSE = "Filtering and verifying vocabulary (2/3)..."
    FILTERING_LLM = "Filtering and verifying vocabulary (3/3)..."
    EXTRACTING_VOCAB = "Fetching dictionary definitions..."
    SAVING_DB = "Saving everything to the database..."

IS_CLOUD_RUN = "PORT" in os.environ

def log_gcp(severity: str, message: str, **kwargs):
    severity_upper = severity.upper()
    
    if IS_CLOUD_RUN:
        entry = {
            "severity": severity_upper,
            "message": message,
            **kwargs
        }
        print(json.dumps(entry), file=sys.stdout, flush=True)
    else:
        colors = {
            "DEFAULT": "\033[90m",          # Dark Grey (Routine, unclassified text)
            "DEBUG": "\033[36m",            # Cyan (Highly detailed developer tracing)
            "INFO": "\033[94m",             # Light Blue (Standard milestone milestones)
            "NOTICE": "\033[32m",           # Green (Normal but significant events)
            "WARNING": "\033[93m",          # Yellow (Issues that aren't errors yet)
            "ERROR": "\033[91m",            # Red (Functionality/pipeline crashes)
            "CRITICAL": "\033[41m\033[97m"  # Red Background + White Text (System-wide emergencies)
        }
        reset = "\033[0m"
        color = colors.get(severity_upper, "")
        
        extra_metadata = f" | Metadata: {kwargs}" if kwargs else ""
        print(f"{color}[{severity_upper}]{reset} {message}{extra_metadata}")