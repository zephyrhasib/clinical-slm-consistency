"""
Generate synthetic demo data to exercise the metric + statistical pipeline
end-to-end BEFORE the real generation runs exist.

Produces:
  data/demo_eval/synthetic_generations.csv
  data/demo_eval/synthetic_references.csv

Schema (long format):
  input_id, model, quant_level, seed, generated_text

Design:
  - 5 patients (DEMO-001 .. DEMO-005) — short synthetic clinical notes
  - 1 model (placeholder: 'gemma_3_4b_it')
  - 2 quant levels: Q4_K_M (closer to reference), Q3_K_M (more drift across seeds)
  - 3 seeds per (patient, model, quant) -> C(3,2)=3 pairs for consistency tests
  - Total rows: 5 × 1 × 2 × 3 = 30 generations + 5 references

Deliberate signal: Q4 generations are closer to references AND more consistent
across seeds than Q3 generations. This means the statistical pipeline should
detect a significant difference in both reference-based and consistency metrics
between Q4 and Q3 — proving the stat pipeline actually discriminates conditions.
"""
import pandas as pd
from pathlib import Path

OUT = Path("data/demo_eval")
OUT.mkdir(parents=True, exist_ok=True)

# Five short clinical-note "references" (plausible style, not actual ACI-Bench)
references = {
    "DEMO-001": (
        "The patient is a 62-year-old male presenting with shortness of breath "
        "and chest tightness for two weeks. He has a history of hypertension "
        "and type 2 diabetes. Blood pressure 148/92, heart rate 84. Chest "
        "exam unremarkable. Plan: ECG, BNP, start lisinopril 10 mg daily, "
        "follow up in two weeks."
    ),
    "DEMO-002": (
        "A 45-year-old female with persistent cough and low-grade fever for "
        "five days. No known allergies. Lungs clear bilaterally on auscultation. "
        "Temperature 38.1 C. Plan: chest x-ray, sputum culture, empiric "
        "amoxicillin 500 mg three times daily, return if symptoms worsen."
    ),
    "DEMO-003": (
        "A 71-year-old male reports left knee pain after a fall last week. "
        "Pain worsens with weight-bearing. No swelling visible. Range of "
        "motion preserved. Plan: x-ray of the left knee, ibuprofen 400 mg "
        "as needed, refer to orthopedics if no improvement in two weeks."
    ),
    "DEMO-004": (
        "A 29-year-old female with intermittent abdominal pain for three "
        "months, worse postprandial. No nausea or weight loss. Abdomen soft, "
        "mild epigastric tenderness. Plan: H. pylori testing, trial of "
        "omeprazole 20 mg daily, dietary modification, follow up in four weeks."
    ),
    "DEMO-005": (
        "A 56-year-old male with fatigue and unintentional weight loss of "
        "six kilograms over three months. Labs show mild anemia. Plan: "
        "complete blood count with differential, iron studies, colonoscopy "
        "referral, follow up after results."
    ),
}

# Q4_K_M generations: paraphrased reference, mostly preserves clinical content
# Three seeds with mild variation (good consistency)
q4_generations = {
    "DEMO-001": {
        0: "62-year-old man with two weeks of dyspnea and chest tightness. History of hypertension and type 2 diabetes. Vitals: BP 148/92, HR 84. Chest examination normal. Plan: obtain ECG and BNP, initiate lisinopril 10 mg daily, return in two weeks.",
        1: "The patient is a 62-year-old male with shortness of breath and chest tightness for two weeks. Past medical history significant for hypertension and diabetes mellitus type 2. BP 148/92, HR 84. Chest exam unremarkable. Plan: ECG, BNP labs, start lisinopril 10 mg once daily, two-week follow-up.",
        2: "A 62-year-old male with two-week history of dyspnea and chest tightness, with hypertension and type 2 diabetes. Blood pressure 148/92, pulse 84. Chest auscultation clear. Will order ECG and BNP, start lisinopril 10 mg daily, follow up in 14 days.",
    },
    "DEMO-002": {
        0: "45-year-old woman presenting with persistent cough and low-grade fever for 5 days. NKDA. Lungs clear. Temp 38.1 C. Plan: chest x-ray, sputum culture, empiric amoxicillin 500 mg TID, return if worsening.",
        1: "The patient is a 45-year-old female with a 5-day history of cough and low-grade fever. No drug allergies. Lung exam clear bilaterally. Temperature 38.1 C. Plan: chest radiograph, sputum culture, start amoxicillin 500 mg three times daily, follow up if symptoms worsen.",
        2: "A 45-year-old woman with 5 days of cough and fever (38.1 C). No allergies. Clear lungs on auscultation. Will obtain chest x-ray and sputum culture, prescribe amoxicillin 500 mg TID, return if not improving.",
    },
    "DEMO-003": {
        0: "71-year-old man with left knee pain following a fall last week. Pain worse with weight bearing. No swelling. ROM preserved. Plan: left knee x-ray, ibuprofen 400 mg PRN, orthopedics referral if no improvement.",
        1: "The patient is a 71-year-old male with left knee pain after a fall one week ago. Pain worse on weight bearing. No visible swelling. Range of motion intact. Plan: x-ray of left knee, ibuprofen 400 mg as needed, refer to ortho if not improved in two weeks.",
        2: "A 71-year-old male with left knee pain since a fall last week. Pain increases with weight bearing. No swelling observed. ROM preserved. Will order knee x-ray, ibuprofen 400 mg PRN, refer ortho if not better.",
    },
    "DEMO-004": {
        0: "29-year-old woman with 3 months of intermittent abdominal pain, worse after meals. No nausea or weight loss. Mild epigastric tenderness. Plan: H. pylori testing, trial of omeprazole 20 mg daily, dietary changes, 4-week follow-up.",
        1: "The patient is a 29-year-old female with intermittent postprandial abdominal pain for 3 months. No associated nausea or weight loss. Abdomen soft with mild epigastric tenderness. Plan: H. pylori testing, omeprazole 20 mg daily, dietary modification, follow up in 4 weeks.",
        2: "A 29-year-old female with 3-month history of intermittent abdominal pain worsening after meals. Denies nausea, weight loss. Soft abdomen, mild epigastric tenderness. H. pylori testing planned, start omeprazole 20 mg daily, dietary advice, return in 1 month.",
    },
    "DEMO-005": {
        0: "56-year-old man with fatigue and 6 kg unintentional weight loss over 3 months. Labs show mild anemia. Plan: CBC with differential, iron studies, colonoscopy referral, follow up after results return.",
        1: "The patient is a 56-year-old male with fatigue and unintentional 6 kg weight loss over 3 months. Mild anemia on initial labs. Plan: complete blood count with differential, iron studies, refer for colonoscopy, follow up post-results.",
        2: "A 56-year-old male with 3-month history of fatigue and 6 kg unintentional weight loss. Mild anemia identified. Will obtain CBC with diff, iron panel, colonoscopy referral, return after results.",
    },
}

# Q3_K_M generations: more drift from reference, less consistent across seeds
# - paraphrased more freely
# - some seeds omit details (e.g., specific BP, specific dose)
# - one seed per patient introduces a mild substantive deviation
q3_generations = {
    "DEMO-001": {
        0: "An older man complains of breathing difficulty and chest pressure for several weeks. He has high blood pressure and diabetes. Vitals slightly elevated. Plan to do an ECG and start a blood pressure medication, see him again soon.",
        1: "Patient with shortness of breath and chest discomfort for two weeks. Past history of hypertension and diabetes. Will check ECG, start an ACE inhibitor, follow up.",
        2: "62-year-old man with respiratory symptoms and chest tightness. Has cardiac risk factors. Plan involves cardiac workup and antihypertensive therapy. Follow up in a few weeks.",
    },
    "DEMO-002": {
        0: "Middle-aged woman with cough and mild fever for almost a week. No allergies known. Lungs sound clear. Plan: imaging, antibiotic such as amoxicillin, follow up if not better.",
        1: "A 45-year-old patient with cough and fever for 5 days. Normal lung exam. Will get chest imaging and begin oral antibiotics. Patient to return if symptoms persist.",
        2: "Patient with a productive cough and a low fever recently. No history of drug reactions. On exam lungs are clear. Plan: get a chest x-ray, prescribe an antibiotic, advise return precautions.",
    },
    "DEMO-003": {
        0: "Elderly man with knee pain after a recent fall. Painful with movement and weight bearing. No swelling. Will obtain x-ray and prescribe pain medication, possible orthopedics referral.",
        1: "71-year-old presenting with left knee pain, occurred after a fall. Plan: imaging and NSAIDs, refer to orthopedics if needed.",
        2: "Senior patient with knee discomfort post fall last week. Pain on weight bearing, no swelling. Plan: knee x-ray, ibuprofen, ortho if no improvement.",
    },
    "DEMO-004": {
        0: "Young woman with stomach pain for months, worse after eating. No other major symptoms. Plan: GI workup, acid suppression, dietary advice, follow up.",
        1: "29-year-old female with abdominal pain worse with meals over three months. Mild epigastric tenderness. Plan: H. pylori test, start PPI, follow up.",
        2: "Patient with intermittent postprandial abdominal pain for 3 months. Will test for H. pylori, prescribe omeprazole, give dietary recommendations.",
    },
    "DEMO-005": {
        0: "Middle-aged man with fatigue and weight loss over several months. Mild anemia. Plan: further blood tests, GI workup, follow up.",
        1: "56-year-old with unintentional weight loss and tiredness. Anemia on labs. Plan to repeat labs and arrange colonoscopy.",
        2: "Patient with three-month history of fatigue and weight loss, anemic. Plan: CBC, iron studies, colonoscopy referral.",
    },
}

# Assemble long-format dataframe
rows = []
for input_id in references:
    for seed in [0, 1, 2]:
        rows.append({
            "input_id":       input_id,
            "model":          "gemma_3_4b_it",
            "quant_level":    "Q4_K_M",
            "seed":           seed,
            "generated_text": q4_generations[input_id][seed],
        })
        rows.append({
            "input_id":       input_id,
            "model":          "gemma_3_4b_it",
            "quant_level":    "Q3_K_M",
            "seed":           seed,
            "generated_text": q3_generations[input_id][seed],
        })

df_gen = pd.DataFrame(rows)
df_ref = pd.DataFrame([{"input_id": k, "reference_note": v} for k, v in references.items()])

df_gen.to_csv(OUT / "synthetic_generations.csv", index=False)
df_ref.to_csv(OUT / "synthetic_references.csv",  index=False)

print(f"Wrote: {OUT/'synthetic_generations.csv'} ({len(df_gen)} rows)")
print(f"Wrote: {OUT/'synthetic_references.csv'}  ({len(df_ref)} rows)")
print(f"\nShape summary:")
print(f"  Patients         : {df_gen['input_id'].nunique()}")
print(f"  Models           : {df_gen['model'].nunique()}")
print(f"  Quant levels     : {sorted(df_gen['quant_level'].unique())}")
print(f"  Seeds per cond.  : {df_gen.groupby(['input_id', 'model', 'quant_level'])['seed'].count().unique()}")
print(f"  Total generations: {len(df_gen)}")
