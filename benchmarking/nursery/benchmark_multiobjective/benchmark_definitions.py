# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from benchmarking.commons.benchmark_definitions import (
    SurrogateBenchmarkDefinition,
)



def fcnet_mo_benchmark(dataset_name):
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=int(1e26),
        n_workers=1,
        max_num_evaluations=1000,
        elapsed_time_attr="metric_elapsed_time",
        metric=["metric_valid_loss", "metric_n_params"],
        mode=["min", "min"],
        blackbox_name="fcnet",
        dataset_name=dataset_name,
    )


fcnet_mo_benchmark_definitions = {
    "fcnet-protein": fcnet_mo_benchmark("protein_structure"),
    "fcnet-naval": fcnet_mo_benchmark("naval_propulsion"),
    "fcnet-parkinsons": fcnet_mo_benchmark("parkinsons_telemonitoring"),
    "fcnet-slice": fcnet_mo_benchmark("slice_localization"),
}



def nas201_mo_benchmark(dataset_name):
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=int(1e26),
        n_workers=1,
        max_num_evaluations=400 * 200,
        elapsed_time_attr="metric_elapsed_time",
        metric=["metric_valid_error", "metric_latency"],
        mode=["min", "min"],
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
        max_resource_attr="epochs",
    )


nas201_mo_benchmark_definitions = {
    "nas201-mo-cifar10": nas201_mo_benchmark("cifar10"),
    "nas201-mo-cifar100": nas201_mo_benchmark("cifar100"),
    "nas201-mo-ImageNet16-120": nas201_mo_benchmark("ImageNet16-120"),
}


benchmark_definitions = {
    **nas201_mo_benchmark_definitions,
    **fcnet_mo_benchmark_definitions,
}
