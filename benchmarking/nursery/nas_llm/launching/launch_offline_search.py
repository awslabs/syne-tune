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
import transformers
import syne_tune
import sagemaker

from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from syne_tune.util import random_string

from pathlib import Path


model_type = "bert-base-cased"
dataset = "rte"
seed = 0
num_epochs = 20
search_strategy = "local_search"
sampling = "one_shot"
search_space = "small"
run = 0

use_accelerate = False

entry_point = "run_offline_search.py"
instance_type = "ml.g4dn.xlarge"
num_samples = 500
volume_size = 125
bucket = sagemaker.Session().default_bucket()
tensorboard_log_dir = "/opt/ml/output/tensorboard"

if model_type in ["gpt2-xl"]:
    instance_type = "ml.g5.8xlarge"
    use_accelerate = True
    volume_size = 900

sm_args = dict(
    entry_point=entry_point,
    source_dir=str(Path(__file__).parent.parent),
    instance_type=instance_type,
    instance_count=1,
    py_version="py38",
    framework_version="1.10.0",
    pytorch_version="1.9",
    transformers_version="4.12",
    max_run=3600 * 72,
    role=get_execution_role(),
    dependencies=transformers.__path__ + syne_tune.__path__,
    volume_size=volume_size,
    checkpoint_local_path="/opt/ml/checkpoints",
)

hyperparameters = {
    "model_name_or_path": model_type,
    "output_dir": sm_args["checkpoint_local_path"],
    "task_name": dataset,
    "search_strategy": search_strategy,
    "seed": seed + 10 * run,
    "search_space": search_space,
    "num_samples": num_samples,
}
if use_accelerate:
    hyperparameters["use_accelerate"] = True

sm_args["hyperparameters"] = hyperparameters

model_uri = (
    f"s3://{bucket}/checkpoints_nas/"
    f"{search_space}/"
    f"{model_type}"
    f"/epochs_{num_epochs}/"
    f"{dataset}/"
    f"{sampling}/"
    f"seed_{seed}/"
)

sm_args["model_uri"] = model_uri

checkpoint_s3_uri = (
    f"s3://{bucket}/weight_sharing_nas/{search_space}/{model_type}/epochs_{num_epochs}/"
    f"{dataset}/{sampling}/seed_{seed}/{search_strategy}/run_{run}/"
)
sm_args["checkpoint_s3_uri"] = checkpoint_s3_uri

output_path = (
    f"s3://{bucket}/tensorboard_search/"
    f"{search_space}/"
    f"{model_type}"
    f"/epochs_{num_epochs}/"
    f"{dataset}/"
    f"{sampling}/"
    f"seed_{seed}/"
)

tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=output_path, container_local_output_path=tensorboard_log_dir
)
sm_args["tensorboard_output_config"] = tensorboard_output_config
est = PyTorch(**sm_args)
hash = random_string(4)

sampling_name = sampling.replace("_", "-")
search_strategy_name = search_strategy.replace("_", "-")
jn = f"search-{dataset}-{sampling_name}-{search_strategy_name}-{seed}-{run}-{hash}"
print(f"Start job {jn}")
est.fit(
    job_name=jn,
    wait=False,
)
