import openai
import json
from pathlib import Path
import pandas
import os
from dotenv import load_dotenv
import re

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT_PATH = Path("./speciality_system_prompt.txt")
DATA_PATH = Path("./agentclinic_medqa_extended.jsonl")
OUTPUT_Path = Path("./agentclinic_medqa_extended_specialitylabeled.jsonl")
MODEL = "gpt-4"
RATE_LIMIT_DELAY = 1.2

system_prompt = SYSTEM_PROMPT_PATH.read_text

with open(DATA_PATH) as data:
  ...

def main():
  print("Starting edits")
  
main()
