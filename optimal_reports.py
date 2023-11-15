import os
import ujson as json
import numpy as np
import matplotlib.pyplot as plt

report_dir = "./reports"
report_files = os.listdir(report_dir)
datasets = []
configs = []
reports_by_dataset = {}
reports_by_config = {}
for report_file in report_files:
    with open(os.path.join(report_dir, report_file), "r") as f:
        name = report_file.replace(".json", "")
        parts = name.split("_")
        dataset = parts[1]
        config = "_".join(parts[2:])
        datasets.append(dataset)
        configs.append(config)
        dat = json.load(f)
        if dataset not in reports_by_dataset:
            reports_by_dataset[dataset] = {}
        reports_by_dataset[dataset][config] = dat["collated_report"][
            "normalized_discounted_cumulative_gain_10"
        ]["mean"]
        if config not in reports_by_config:
            reports_by_config[config] = {}
        reports_by_config[config][dataset] = dat["collated_report"][
            "normalized_discounted_cumulative_gain_10"
        ]["mean"]
datasets = sorted(list(set(datasets)))
configs = sorted(list(set(configs)))

print("Configs")
sorted_configs = [
    k
    for k, v in sorted(
        reports_by_config.items(),
        key=lambda item: np.mean(list(item[1].values())),
        reverse=True,
    )
]
for config in sorted_configs:
    print(
        "{:<64}: {}".format(config, np.mean(list(reports_by_config[config].values())))
    )
    for dataset in reports_by_config[config]:
        print("\t{:<32}: {}".format(dataset, reports_by_config[config][dataset]))

print("Datasets")
for dataset in datasets:
    print(dataset)
    sorted_data = {
        k: v
        for k, v in sorted(
            reports_by_dataset[dataset].items(), key=lambda x: x[1], reverse=True
        )
    }
    for config in sorted_data:
        print("\t{:<64}: {}".format(config, reports_by_dataset[dataset][config]))

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_title("RAGMAN Performance by Dataset")
ax.set_xlabel("BM25+ Weight")
ax.set_ylabel("NDCG@10")
plotting_data = {
    dataset: {
        float(config.split("_")[-1].split("-")[3]) / 100: value
        for config, value in configs.items()
        if "bge" in config and "pool" in config and "-3" in config and "mean" in config
    }
    for dataset, configs in reports_by_dataset.items()
}
for dataset in plotting_data.keys():
    domain = np.array(list(plotting_data[dataset].keys()))
    rng = np.array(list(plotting_data[dataset].values()))
    order = np.argsort(domain)
    ax.plot(
        domain[order],
        rng[order],
        label=dataset,
    )
ax.legend()
fig.savefig("test.png")
