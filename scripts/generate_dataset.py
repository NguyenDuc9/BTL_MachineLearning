from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)
rows = 800
study_hours = np.round(rng.uniform(0, 10, rows), 1)
attendance = np.round(rng.uniform(45, 100, rows), 1)
assignment_score = np.round(rng.uniform(2, 10, rows), 1)
midterm_score = np.round(rng.uniform(2, 10, rows), 1)
practice_score = np.round(rng.uniform(2, 10, rows), 1)
latent_score = (
    0.16 * study_hours + 0.025 * attendance + 0.22 * assignment_score
    + 0.32 * midterm_score + 0.25 * practice_score + rng.normal(0, 0.65, rows)
)
result = (latent_score >= 6.5).astype(int)
frame = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "assignment_score": assignment_score,
    "midterm_score": midterm_score,
    "practice_score": practice_score,
    "result": result,
})
output = ROOT_DIR / "data" / "students.csv"
output.parent.mkdir(exist_ok=True)
frame.to_csv(output, index=False)
print(f"Generated {len(frame)} rows at {output}")
