import re
import pandas as pd
from collections import defaultdict

for i in [1,2]:
    # Step 1: Read the content of the input text file
    with open(f"results-Scenario#{i}.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Step 2: Use regular expressions to extract (pretrain_model, N, avg_accuracy)
    pattern = re.compile(r"(D1-[^-]+-Ori)-D\d+-[^ ]+ - (\d+)\s*\n avg -> ([\d.]+)")
    matches = pattern.findall(text)

    # Step 3: Organize the data: {N: {pretrain_model: [list of avg accuracies]}}
    data = defaultdict(lambda: defaultdict(list))
    for pretrain_model, n_str, avg_str in matches:
        N = int(n_str)
        avg = float(avg_str)
        data[N][pretrain_model].append(avg)

    # Step 4: Compute the mean accuracy for each (pretrain_model, N) and round to two decimals
    rows = []
    all_pretrain_keys = set()

    for N in sorted(data.keys()):
        row = {"N": N}
        for pretrain_model in data[N]:
            all_pretrain_keys.add(pretrain_model)
            values = data[N][pretrain_model]
            average = round(sum(values) / len(values), 2)  # Round to two decimal places
            row[pretrain_model] = average
        rows.append(row)

    # Step 5: Create DataFrame and reorder columns (N first, then sorted pretrain models)
    df = pd.DataFrame(rows)
    df = df[["N"] + sorted(all_pretrain_keys)]

    # Step 6: Save the result to a CSV file
    df.to_csv(f"results-Scenario#{i}-table.csv", index=False, encoding="utf-8-sig")

    print("✅ Finished: Output saved to pretrain_finetune_avg_comparison.csv with rounded averages")


