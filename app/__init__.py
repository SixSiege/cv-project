from flask import Flask
from dotenv import load_dotenv
load_dotenv()
import os
secret_key = os.environ.get("SESSION_SECRET_KEY")

app = Flask(__name__)
app.secret_key = secret_key

from app import routes