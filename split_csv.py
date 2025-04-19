import pandas as pd

# Load your CSV
df = pd.read_csv("../data/original_data.csv", encoding='latin-1')

# Split (e.g., 70% for Part 1, 30% for Part 2)
split_ratio = 0.7
part1 = df.sample(frac=split_ratio, random_state=42)
part2 = df.drop(part1.index)

# Save to new files
part1.to_csv("../data/data1.csv", index=False)
part2.to_csv("../data/data2.csv", index=False)
