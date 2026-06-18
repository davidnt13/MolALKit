import json
from itertools import product

# Define your 5 experimental axes
batch_sizes = [(20, 100), (50, 200)]  # Pairs of (addition_batch, forget_batch)
forget_starts = [0, 10]               # When to start forgetting (iteration index)
forget_intervals = [1, 5]             # How many iterations between forget steps
uncertainty_types = ["min_oob_uncertainty", "max_oob_uncertainty"]    # Forgetter selection criterion
seeds = [0, 1, 2]                     # Triplicate runs for statistical confidence

# Use itertools to generate every single permutation automatically
experiments = list(product(
    batch_sizes, 
    forget_starts, 
    forget_intervals, 
    uncertainty_types, 
    seeds
))

# Write out the mapping file
# Format per line: Index \t [parameters JSON string]
with open("job_array_map.txt", "w") as f:
    for idx, item in enumerate(experiments):
        f.write(f"{idx}\t{json.dumps(item)}\n")

print(f"Success! Generated job_array_map.txt with {len(experiments)} unique combinations.")