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
from syne_tune.try_import import try_import_moo_message
from syne_tune.optimizer.schedulers.multiobjective.moasha import MOASHA
from syne_tune.optimizer.schedulers.multiobjective.multi_objective_regularized_evolution import (
    MultiObjectiveRegularizedEvolution,
)
from syne_tune.optimizer.schedulers.multiobjective.nsga2_searcher import (
    NSGA2Searcher,
)
from syne_tune.optimizer.schedulers.multiobjective.linear_scalarizer import (
    LinearScalarizedScheduler,
)
from syne_tune.optimizer.schedulers.multiobjective.expected_hyper_volume_improvement import (
    ExpectedHyperVolumeImprovement,
)

__all__ = [
    "MOASHA",
    "MultiObjectiveRegularizedEvolution",
    "NSGA2Searcher",
    "LinearScalarizedScheduler",
    "ExpectedHyperVolumeImprovement"
]

try:
    from syne_tune.optimizer.schedulers.multiobjective.multi_surrogate_multi_objective_searcher import (  # noqa: F401
        MultiObjectiveMultiSurrogateSearcher,
    )
    from syne_tune.optimizer.schedulers.multiobjective.random_scalarization import (  # noqa: F401
        MultiObjectiveLCBRandomLinearScalarization,
    )

    __all__.extend(
        [
            "MultiObjectiveMultiSurrogateSearcher",
            "MultiObjectiveLCBRandomLinearScalarization",
        ]
    )
except ImportError:
    print(try_import_moo_message())
