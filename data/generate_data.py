"""
generate_data.py
Gera dados artificiais realistas de consultas numa clínica médica.
"""

import numpy as np
import pandas as pd

def generate_appointments(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    # --- Atributos do paciente ---
    age = rng.integers(5, 90, size=n)
    gender = rng.choice(["M", "F"], size=n)
    distance_km = rng.exponential(scale=8, size=n).clip(0.5, 60).round(1)
    chronic_disease = rng.choice([0, 1], size=n, p=[0.65, 0.35])
    previous_noshow_rate = rng.beta(a=2, b=5, size=n).round(2)   # histórico 0–1
    sms_received = rng.choice([0, 1], size=n, p=[0.3, 0.7])

    # --- Atributos da consulta ---
    day_of_week = rng.integers(0, 5, size=n)         # 0=Segunda … 4=Sexta
    lead_days = rng.integers(0, 60, size=n)          # dias entre marcação e consulta
    appointment_hour = rng.integers(8, 18, size=n)
    specialty = rng.choice(
        ["Clínica Geral", "Pediatria", "Cardiologia", "Dermatologia", "Ortopedia"],
        size=n,
        p=[0.40, 0.20, 0.15, 0.15, 0.10],
    )
    is_first_visit = rng.choice([0, 1], size=n, p=[0.55, 0.45])

    # --- Probabilidade de no-show (regra determinística + ruído) ---
    logit = (
        -1.5
        + 0.03 * np.clip(age - 40, -35, 40)  # jovens faltam mais
        + 0.5 * (gender == "M").astype(float)
        + 0.04 * distance_km
        + 2.5 * previous_noshow_rate
        - 0.6 * chronic_disease
        - 0.7 * sms_received
        + 0.02 * lead_days
        + 0.3 * is_first_visit
        + rng.normal(0, 0.4, size=n)
    )
    prob_noshow = 1 / (1 + np.exp(-logit))
    no_show = (rng.random(size=n) < prob_noshow).astype(int)

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "distance_km": distance_km,
        "chronic_disease": chronic_disease,
        "previous_noshow_rate": previous_noshow_rate,
        "sms_received": sms_received,
        "day_of_week": day_of_week,
        "lead_days": lead_days,
        "appointment_hour": appointment_hour,
        "specialty": specialty,
        "is_first_visit": is_first_visit,
        "no_show": no_show,
    })

    return df


if __name__ == "__main__":
    df = generate_appointments(n=2000)
    df.to_csv("appointments.csv", index=False)
    print(f"Dataset gerado: {len(df)} consultas")
    print(f"Taxa de no-show: {df['no_show'].mean():.1%}")
    print(df.head())
