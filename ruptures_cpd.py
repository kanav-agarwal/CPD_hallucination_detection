from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt


RESPONSE_INDEX = 14
# Load data

data = pd.read_csv("hallu_token_probabilities.csv")
answer_data = data.iloc[RESPONSE_INDEX - 1 :].reset_index(drop=True)
index_mapping = answer_data["token_index"] 
index = answer_data["token_index"]
signal = answer_data["log_probability"].values.reshape(-1, 1)


# Dynp parameters
L = len(signal)
n_bkps = 1
min_size = 3
jump = 1

# Fit ruptures model and make preditions
algo = rpt.Dynp(model="l2", min_size=min_size, jump=jump).fit(signal)
cp_indices = algo.predict(n_bkps=n_bkps)

# Create dataframe of detected change points
cp_records = []

for cp in cp_indices:
    if cp < L:
        true_token_id = index_mapping.iloc[cp]
        actual_token = data.loc[data["token_index"] == true_token_id, "token"].values[0]
        
        cp_records.append({
            "variable": "log_probability",
            "slice_index": cp,
            "true_token_index": true_token_id,
            "token_text": actual_token
        })

cp_df = pd.DataFrame(cp_records)
print(cp_df)

# output_path = "ruptures_change_points.csv"
# cp_df.to_csv(output_path, index=False)

# print(f"Change points saved to {output_path}")

# Save directory
# save_dir = project_root / "figures" / "ruptures_change_points"
# save_dir.mkdir(parents=True, exist_ok=True)

# # Plot
# plt.figure(figsize=(12,6))

# plt.plot(time, signal, label="Log Probabilities")

# plt.scatter(
#     time.iloc[cp_indices],
#     signal[cp_indices, 0],
#     color="red",
#     label="Change Points"
# )

# plt.xlabel("Time (s)")
# plt.ylabel("Z-Score")
# plt.title("Ruptures Change Points - AHU Mixed Air Temperature")
# plt.legend()

# plt.tight_layout()
# plt.savefig(save_dir / "ruptures_mixed_air_temp.png")
# plt.close()

# print("Done. Change point plot saved.")