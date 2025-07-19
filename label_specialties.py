from openai import OpenAI
import json
from pathlib import Path
import pandas
import os
from dotenv import load_dotenv
import re
import time
import traceback

load_dotenv()
client = OpenAI()
print(f"API Key loaded: {'OPENAI_API_KEY' in os.environ}")

ALLOWED_SPECIALTIES = {
"Primary Care", "Internal Medicine", "Surgical Specialties",
"Pediatrics", "Obstetrics & Gynecology", "Psychiatry",
"Neurology", "Radiology", "Pathology",
"Anesthesiology"
}

SYSTEM_PROMPT_PATH = Path("./specialty_system_prompt.txt")
DATA_PATH = Path("./data/agentclinic_medqa_extended.jsonl")
OUTPUT_PATH = Path("./data/agentclinic_medqa_extended_specialtylabeled.jsonl")
MODEL = "gpt-4"
RATE_LIMIT_DELAY = 1.2

system_prompt = SYSTEM_PROMPT_PATH.read_text()

def load_processed_case_ids(output_path):
    processed_ids = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    processed_ids.add(case.get("case_id"))
    return processed_ids

def load_cases_with_id(jsonl_path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    for idx, case in enumerate(data):
        case["case_id"] = idx  # Assign unique ID here
    return data

def build_user_message(case):
    osce = case.get("OSCE_Examination", {})

    objective = osce.get("Objective_for_Doctor", "").strip()
    actor = osce.get("Patient_Actor", {})
    physical = osce.get("Physical_Examination_Findings", {})
    tests = osce.get("Test_Results", {})

    return (
        f"Objective: {objective}\n\n"
        f"Patient History: {json.dumps(actor)}\n\n"
        f"Physical Findings: {json.dumps(physical)}\n\n"
        f"Test Results: {json.dumps(tests)}"
    )


def label_case(case):
  """label one case"""
  prompt = build_user_message(case)

  try:
    response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=20,
        )
    
    specialty = response.choices[0].message.content.strip()
    if specialty not in ALLOWED_SPECIALTIES:
       specialty = "Primary Care"
    return specialty
  except Exception:
    print("Error with case")
    #'''
    traceback.print_exc()
    #'''
    return "Error"

def main():
  """run the data change loop"""
  print("Starting...")
  data = load_cases_with_id(DATA_PATH)[:150]
  print(f"Loaded {len(data)} cases.")
  processed_ids = load_processed_case_ids(OUTPUT_PATH)

  with OUTPUT_PATH.open("w") as f:
     for case in data:
      if case["case_id"] in processed_ids:
        print(f"Skipping case: {case["case_id"]}")
        continue

      specialty = label_case(case)
      case["Specialty"] = specialty

      f.write(json.dumps(case) + "\n")  
      f.flush()
      print(f"Processed case {case['case_id']} → {specialty}")

  print(f"Saved labeled cases to {OUTPUT_PATH}")

main()
