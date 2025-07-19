import openai
import json
from pathlib import Path
import pandas
import os
from dotenv import load_dotenv
import re
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT_PATH = Path("./specialty_system_prompt.txt")
DATA_PATH = Path("./agentclinic_medqa_extended.jsonl")
OUTPUT_PATH = Path("./agentclinic_medqa_extended_specialtylabeled.jsonl")
MODEL = "gpt-4"
RATE_LIMIT_DELAY = 1.2

system_prompt = SYSTEM_PROMPT_PATH.read_text()

def load_cases(jsonl_path):
    """Load the MedQA cases from a .jsonl file into a list of dictionaries."""
    with jsonl_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_user_message(case):
  """Extract relevant fields"""
  objective = case.get("OCSE_Objective", "").strip()
  actor = case.get("OSCE_Actor", "").strip()
  physical = case.get("OCSE_Physcial", "").strip()
  tests = case.get("OCSE_Tests", "").strip()

  return (
        f"Objective for Doctor: {objective}\n\n"
        f"Patient Actor: {actor}\n\n"
        f"Physical Examination Findings: {physical}\n\n"
        f"Test Results: {tests}"
    )


def label_case(case):
  """label one case"""
  prompt = build_user_message(case)

  try:
    response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=20,
        )
    
    specialty = response["choices"][0]["message"]["content"].strip()
    return specialty
  except Exception:
    print("Error with case")
    return "Error"

def main():
  """run the data change loop"""
  print("Starting...")
  data = load_cases(DATA_PATH)[:5]
  print(f"Loaded {len(data)} cases.")
  updated_data = []

  for i, case in enumerate(data):
    specialty = label_case(case)
    case["specialty"] = specialty
    updated_data.append(case)
    print(f"[{i+1}/{len(data)}] → {specialty}")
    time.sleep(RATE_LIMIT_DELAY)

  with OUTPUT_PATH.open("w") as f:
     for case in updated_data:
      f.write(json.dumps(case) + "\n")
  
  print(f"Saved labeled cases to {OUTPUT_PATH}")

main()
