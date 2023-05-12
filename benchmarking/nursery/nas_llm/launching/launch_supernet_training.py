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
import sagemaker

from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

import transformers

from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from syne_tune.util import random_string

from pathlib import Path


model_type = "bert-base-cased"
dataset = "rte"
seed = 0
num_epochs = 20
search_space = "small"
sampling = "one_shot"
instance_type = "ml.g4dn.xlarge"
accelerate = False
entry_point = "train_supernet.py"

bucket = sagemaker.Session().default_bucket()
log_dir_tensorboard = "/opt/ml/output/tensorboard"

if model_type in ["gpt2-xl"]:
    instance_type = "ml.g5.12xlarge"
    accelerate = True
    num_epochs = 10
    entry_point = "accelerate_launcher.py"


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
    dependencies=transformers.__path__,
    disable_profiler=True,
    checkpoint_local_path="/opt/ml/checkpoints",
)

hyperparameters = {
    "model_name_or_path": model_type,
    "output_dir": sm_args["checkpoint_local_path"],
    "task_name": dataset,
    "num_train_epochs": num_epochs,
    "save_strategy": "epoch",
    "sampling_strategy": sampling,
    "learning_rate": 2e-05,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "seed": seed,
    "fp16": True,
    "search_space": search_space,
}

if accelerate:
    hyperparameters["use_accelerate"] = True
    hyperparameters["training_script"] = "train_supernet.py"
    hyperparameters["config_file"] = "default_config.yaml"

sm_args["hyperparameters"] = hyperparameters
sm_args["checkpoint_s3_uri"] = (
    f"s3://{bucket}/checkpoints_nas/"
    f"{search_space}/"
    f"{model_type}"
    f"/epochs_{num_epochs}/"
    f"{dataset}/"
    f"{sampling}/"
    f"seed_{seed}/"
)

output_path = (
    f"s3://{bucket}/tensorboard/"
    f"{search_space}/"
    f"{model_type}"
    f"/epochs_{num_epochs}/"
    f"{dataset}/"
    f"{sampling}/"
    f"seed_{seed}/"
)
tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=output_path, container_local_output_path=log_dir_tensorboard
)
sm_args["tensorboard_output_config"] = tensorboard_output_config

est = PyTorch(**sm_args)
hash = random_string(4)

job_name = f"{model_type}-{search_space}-{dataset}-{sampling.replace('_', '-')}-{seed}-{hash}"
print(f"Start job {job_name}")
est.fit(
    job_name=job_name,
    wait=False,
)
