from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkDefinition:
    max_wallclock_time: float
    n_workers: int
    elapsed_time_attr: str
    metric: str
    mode: str
    blackbox_name: str
    dataset_name: str
    time_this_resource_attr: Optional[str] = None
    max_num_evaluations: Optional[int] = None
    surrogate: Optional[str] = None
    search_options: Optional[dict] = None


def fcnet_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=1200,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_loss",
        mode="min",
        blackbox_name="fcnet",
        dataset_name=dataset_name,
        search_options={'threshold_failure': 14},
    )


def nas201_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=3600 * 6,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        time_this_resource_attr='metric_runtime',
        metric="metric_valid_error",
        mode="min",
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
        search_options={'searcher_sidelength_init': 1.5,
                        'searcher_sidelength_max': 2.0,
                        'searcher_sidelength_min': 0.25,
                        'threshold_failure': 15},
    )


def lcbench_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=7200,
        n_workers=4,
        elapsed_time_attr="time",
        metric="val_accuracy",
        mode="max",
        blackbox_name="lcbench",
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",
        max_num_evaluations=4000,
    )


benchmark_definitions = {
    "fcnet-protein": fcnet_benchmark("protein_structure"),
    "fcnet-naval": fcnet_benchmark("naval_propulsion"),
    "fcnet-parkinsons": fcnet_benchmark("parkinsons_telemonitoring"),
    "fcnet-slice": fcnet_benchmark("slice_localization"),
    "nas201-cifar10": nas201_benchmark("cifar10"),
    "nas201-cifar100": nas201_benchmark("cifar100"),
    "nas201-ImageNet16-120": nas201_benchmark("ImageNet16-120"),
}

# 5 most expensive lcbench datasets
lc_bench_datasets = [
    "Fashion-MNIST",
    "airlines",
    "albert",
    "covertype",
    "christine",
]
for task in lc_bench_datasets:
    benchmark_definitions["lcbench-" + task.replace("_", "-").replace(".", "")] = lcbench_benchmark(task)
