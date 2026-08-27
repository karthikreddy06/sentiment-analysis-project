import sys
import os

# Add parent directory to path so python can find server.py and SentimentAnalysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app
