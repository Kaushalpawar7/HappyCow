import pandas as pd
import numpy as np

# Parameters
num_cows = 200
samples_per_cow = 100
error_rate = 0.02
fever_base_rate = 0.35
mastitis_rate = 0.25
blackquarter_rate = 0.09
pneumonia_rate = 0.15
anaplasmosis_rate = 0.10
bluetongue_rate = 0.08
lameness_rate = 0.08

# Environmental factors
temp_env = 32
humidity = 0.7
farm_quality = np.random.choice([0.9, 1.0, 1.1], p=[0.2, 0.6, 0.2])

# Initialize random seed
np.random.seed(42)

# Generate data
data = []
for cow_id in range(num_cows):
    age = np.random.choice([2, 4, 6], p=[0.3, 0.5, 0.2])
    lactating = np.random.choice([0, 1], p=[0.4, 0.6])
    base_temp = np.random.normal(38.5, 0.2) if age < 5 else np.random.normal(38.7, 0.2)
    base_bpm = np.random.normal(70, 5) if age < 5 else np.random.normal(75, 5)
    base_activity = np.random.normal(75, 10) if age < 5 else np.random.normal(65, 10)
    base_activity *= (0.9 if lactating else 1.0) * farm_quality

    fever_prob = fever_base_rate * (1.2 if age > 5 else 1.0)  # Per-cow rate
    disease_risk = 1.1 if age > 5 else 1.0

    prev_temp = base_temp
    prev_bpm = base_bpm
    prev_activity = base_activity
    fever_duration = 0

    for t in range(samples_per_cow):
        time = t
        is_daytime = (t % 24) < 16
        is_feeding = (t % 24) in [6, 18]

        temp = base_temp + np.random.normal(0, 0.1)
        bpm = base_bpm + np.random.normal(0, 2)
        activity = base_activity + np.random.normal(0, 5)

        if is_feeding:
            activity *= 1.3
        elif not is_daytime:
            activity *= 0.8
        if np.random.rand() < 0.05:
            activity *= 1.2

        if temp_env > 30 and humidity > 0.6:
            temp += np.random.uniform(0.1, 0.3)
            bpm += np.random.uniform(2, 5)
            activity *= 0.95

        if fever_duration > 0:
            has_fever = True
            fever_duration -= 1
        else:
            has_fever = np.random.rand() < (fever_prob / 10)  # 3.5% per sample
            if has_fever:
                fever_duration = np.random.randint(4, 12)
        if has_fever:
            temp = np.random.uniform(39.5, 40.5)
            bpm += np.random.uniform(5, 15)
            activity *= np.random.uniform(0.8, 0.9)

        is_lame = np.random.rand() < lameness_rate
        if is_lame:
            activity *= np.random.uniform(0.6, 0.8)

        temp = min(max(temp, 37.5), 41.0)
        bpm = min(max(bpm, 50), 110)
        activity = min(max(activity, 20), 100)

        fever = 1 if has_fever else 0
        mastitis = 0
        blackquarter = 0
        pneumonia = 0
        anaplasmosis = 0
        bluetongue = 0

        if fever:
            if np.random.rand() < mastitis_rate * disease_risk:
                mastitis = 1
                activity *= 0.6  # Stronger activity drop (was 0.8)
                bpm += np.random.uniform(0, 5)  # Slight BPM increase
            if np.random.rand() < blackquarter_rate * disease_risk:
                blackquarter = 1
                activity *= 0.9
                bpm += np.random.uniform(10, 20)  # Larger BPM spike (was 5–10)
            if np.random.rand() < pneumonia_rate * disease_risk:
                pneumonia = 1
                activity *= 0.5
                bpm += np.random.uniform(15, 25)  # Stronger BPM signal (new)
            if np.random.rand() < anaplasmosis_rate * disease_risk:
                anaplasmosis = 1
                activity *= 0.7
                temp += np.random.uniform(0.2, 0.4)  # Sustained fever (was 0.1–0.3)
            if np.random.rand() < bluetongue_rate * disease_risk:
                bluetongue = 1
                activity *= 0.3  # Severe activity drop (was 0.4)
                bpm += np.random.uniform(5, 10)  # Moderate BPM increase (new)

        temp_trend = temp - prev_temp
        bpm_trend = bpm - prev_bpm
        activity_trend = activity - prev_activity

        data.append({
            'Cow_ID': cow_id,
            'Time': time,
            'Temperature (°C)': temp,
            'BPM': bpm,
            'Activity': activity,
            'Temp_Trend': temp_trend,
            'BPM_Trend': bpm_trend,
            'Activity_Trend': activity_trend,
            'Fever': fever,
            'Mastitis': mastitis,
            'Black Quarter': blackquarter,
            'Bovine Pneumonia': pneumonia,
            'Anaplasmosis': anaplasmosis,
            'Blue Tongue': bluetongue
        })

        prev_temp = temp
        prev_bpm = bpm
        prev_activity = activity

# Create DataFrame
df = pd.DataFrame(data)

# Introduce sensor noise
num_errors = int(len(df) * error_rate)
error_indices = np.random.choice(len(df), num_errors, replace=False)
df.loc[error_indices, 'Temperature (°C)'] += np.random.normal(0, 0.2, num_errors)
df.loc[error_indices, 'BPM'] += np.random.normal(0, 5, num_errors)
df.loc[error_indices, 'Activity'] += np.random.normal(0, 5, num_errors)

# Ensure realistic bounds
df['Temperature (°C)'] = df['Temperature (°C)'].clip(37.5, 41.0)
df['BPM'] = df['BPM'].clip(50, 110)
df['Activity'] = df['Activity'].clip(20, 100)

# Verify positive cases
print("Positive cases per disease before saving:")
print(df[['Fever', 'Mastitis', 'Black Quarter', 'Bovine Pneumonia', 'Anaplasmosis', 'Blue Tongue']].sum())

# Save to Excel
df.to_excel('cow_health_dataset_realistic_v5.xlsx', index=False)
print("Realistic dataset with balanced diseases created and saved successfully!")