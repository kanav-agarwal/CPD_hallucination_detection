import pandas as pd
import numpy as np
from scipy import stats

data = pd.read_csv("Generated_hallu_prompt/hallu_token_probabilities1.csv")
answer_index = (data["token"] == "Answer").idxmax()

response = data.iloc[answer_index + 1:].copy()

response['z_score'] = np.abs(stats.zscore(response['log_probability']))

outliers = response[response['z_score'] > 3]

print(outliers)
