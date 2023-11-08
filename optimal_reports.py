import os
import ujson as json
import numpy as np

report_dir = "./reports"
report_files = os.listdir(report_dir)
datasets = []
configs = []
reports = {}
for report_file in report_files:
    with open(os.path.join(report_dir, report_file), "r") as f:
        name = report_file.replace(".json", "")
        parts = name.split("_")
        dataset = parts[1]
        config = "_".join(parts[2:])
        datasets.append(dataset)
        configs.append(config)
        if config not in reports:
            reports[config] = {}
        reports[config][dataset] = json.load(f)
datasets = sorted(list(set(datasets)))
configs = sorted(list(set(configs)))

condensed_reports = {}
for config in reports:
    ndcgs = []
    for dataset in reports[config]:
        ndcgs.append(
            reports[config][dataset]["collated_report"][
                "normalized_discounted_cumulative_gain_10"
            ]["mean"]
        )
    condensed_reports[config] = float(np.mean(ndcgs))
condensed_reports = {
    k: v for k, v in sorted(condensed_reports.items(), key=lambda x: x[1], reverse=True)
}
top_rank = {}
for dataset in datasets:
    scores = [
        reports[config][dataset]["collated_report"][
            "normalized_discounted_cumulative_gain_10"
        ]["mean"]
        for config in configs
    ]
    top_rank[dataset] = (configs[np.argmax(scores)], np.max(scores))
print(json.dumps(condensed_reports, indent=2))
print(json.dumps(top_rank, indent=2))
