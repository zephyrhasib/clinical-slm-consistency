# Worked Example: Numerical Suppression

Numerical Jaccard measures agreement between the (entity, value, unit) facts extracted from two generations of the same encounter. Because a Jaccard index also rises when there are fewer facts to compare, it is read alongside the mean number of facts per note (`mean_fact_count` in `data/metric_results.csv`). A rising Jaccard with a falling fact count signals suppressed content, not improved agreement.

This example shows the effect on a single encounter: as the fact count falls to zero, numerical Jaccard jumps from 0.08 to 1.00.

| Encounter | Model | Level | Mean facts per note (20 generations) | Numerical Jaccard |
|---|---|---|---|---|
| MAIN-013 | Qwen-MediCare-BD | Q6_K | 3.40 | 0.08 |
| MAIN-013 | Qwen-MediCare-BD | Q5_K_M | 1.75 | 0.18 |
| MAIN-013 | Qwen-MediCare-BD | Q4_K_M | 3.15 | 0.11 |
| MAIN-013 | Qwen-MediCare-BD | Q3_K_M | 0.00 | 1.00 |

At Q3_K_M the model stopped writing clinical notes and re-enacted the consultation instead. With no numerical facts in any of its twenty generations, every pair compared two empty sets, which the metric scores as perfect agreement.

The two generations below are reproduced verbatim, including the model's own spelling and punctuation.

## Q6_K, seed 0 (153 words)

> Patient: Jacqueline, 31-year-old female with chief complaint ankle pain after mountain biking accident. Physical exam reveals edema on lateral malleolus as well as ecchymosis of the skin overlying this area. Doctor’s diagnosis and plan: Chief Complaint Ankle sprain due to acute injury while mountain biking History Pain is located in outer aspect of right foot, associated with numbness/tingling sensation. Patient has had a fever since incident but no other past medical history or current medications; patient denies alcohol use or illicit drug usage. Physical Exam Vital signs: T 98.6 F (37 C), BP 120/85 mmHg HR 74 bpm, RR 14 L/min Right ankle edema and erythema of the lateral malleolus; pain to palpation on right lateral malleolus with decreased range of motion in flexion and extension as well as internal and external rotation. No injury noted at knee or foot. Plan Air cast for protection, meloxicam 15 mg/day, ice therapy, weight-bearing as tolerated

## Q3_K_M, seed 1 (80 words)

> Patient: hello, how are you? I heard that you injured your ankle Doctor: hi jaqueline how are you i heard that you injured the right ankle doctor: yeah it's not going so great patient: no doctor: what were you doing when this happened did you fall off of a bike or something like that doctor: well do you remember where was you mountain biking lesson yesterday patient: mm-hmm doctor: okay alright and are there any numbing in your area [patient]
