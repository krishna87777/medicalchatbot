from dotenv import load_dotenv
import os

# Load .env file (place this at the top of your script)
load_dotenv()

# Verify the token is loaded (optional debug)
print("Token loaded:", os.getenv("HF_TOKEN") is not None)  # Should print "True"