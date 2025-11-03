import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(override=True)


print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("LANGSMITH_API_KEY:", os.getenv("LANGSMITH_API_KEY"))

