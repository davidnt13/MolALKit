"""Print '<peak_train_size> <n_rows>' for a no-forget al_traj.csv (peak = iteration with max mcc_selector)."""
import sys, ast, pandas as pd
df = pd.read_csv(sys.argv[1])
ts = df['id_prior_al'].apply(lambda x: len(ast.literal_eval(x))) + df['id_add'].apply(lambda x: len(ast.literal_eval(x)))
print(int(ts.iloc[df['mcc_selector'].idxmax()]), len(df))
