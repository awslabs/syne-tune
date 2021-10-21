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
"""
Reproduces NASBench201 benchmark from AutoGluonExperiments repo
"""
import os
import argparse
import logging
import time

from sagemaker_tune.report import Reporter
from sagemaker_tune.search_space import choice, add_to_argparse
from benchmarks.checkpoint import resume_from_checkpointed_model, \
    checkpoint_model_at_rung_level, add_checkpointing_to_argparse
from benchmarks.utils import parse_bool


# First is default value
x_range = ['skip_connect',
           'none',
           'nor_conv_1x1',
           'nor_conv_3x3',
           'avg_pool_3x3']


CONFIG_KEYS = ('x0', 'x1', 'x2', 'x3', 'x4', 'x5')


_config_space = {k: choice(x_range) for k in CONFIG_KEYS}


def nasbench201_default_params(params=None):
    dont_sleep = str(
        params is not None and params.get('backend') == 'simulated')
    return {
        'max_resource_level': 200,
        'grace_period': 1,
        'reduction_factor': 3,
        'max_resource_attr': 'epochs',
        'instance_type': 'ml.m5.large',
        'num_workers': 4,
        'framework': 'PyTorch',
        'framework_version': '1.6',
        'dataset_path': './',
        'dataset_name': 'cifar10-valid',
        'dataset_s3_bucket': 'TODO',
        'dont_sleep': dont_sleep,
        'cost_model_type': 'linear',
    }

def nasbench201_benchmark(params):
    config_space = dict(
        _config_space,
        epochs=params['max_resource_level'],
        dataset_path=params['dataset_path'],
        dataset_name=params['dataset_name'],
        dataset_s3_bucket=params.get('dataset_s3_bucket'),
        dont_sleep=params['dont_sleep'])
    return {
        'script': __file__,
        'metric': 'objective',
        'mode': 'max',
        'resource_attr': 'epoch',
        'elapsed_time_attr': 'elapsed_time',
        'map_reward': '1_minus_x',
        'config_space': config_space,
        'supports_simulated': True,
        'cost_model': get_cost_model(params),
    }


def get_cost_model(params):
    try:
        cost_model_type = params.get('cost_model_type')
        if cost_model_type is None:
            cost_model_type = 'linear'
        if cost_model_type.startswith('linear'):
            from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.models.cost.linear_cost_model \
                import NASBench201LinearCostModel

            map_config_values = {
                'skip_connect': NASBench201LinearCostModel.Op.SKIP_CONNECT,
                'none': NASBench201LinearCostModel.Op.NONE,
                'nor_conv_1x1': NASBench201LinearCostModel.Op.NOR_CONV_1x1,
                'nor_conv_3x3': NASBench201LinearCostModel.Op.NOR_CONV_3x3,
                'avg_pool_3x3': NASBench201LinearCostModel.Op.AVG_POOL_3x3,
            }
            conv_separate_features = ('cnvsep' in cost_model_type)
            count_sum = ('sum' in cost_model_type)
            cost_model = NASBench201LinearCostModel(
                config_keys=CONFIG_KEYS,
                map_config_values=map_config_values,
                conv_separate_features=conv_separate_features,
                count_sum=count_sum)
        else:
            from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.models.cost.sklearn_cost_model \
                import ScikitLearnCostModel

            cost_model = ScikitLearnCostModel(cost_model_type)
        return cost_model
    except Exception:
        return None


def download_datafile(dataset_path, dataset_name, s3_bucket):
    assert dataset_name in ["ImageNet16-120", "cifar10-valid", "cifar100"]
    assert s3_bucket is not None and s3_bucket != 'TODO', \
        "TODO: Need s3_bucket to point to bucket where nasbench201 data can be downloaded"
    dataset_fname = f"nasbench201_reduced_{dataset_name}.csv"
    s3_path = 'dataset'
    fname_local = os.path.join(dataset_path, dataset_fname)
    if not os.path.isfile(fname_local):
        os.makedirs(dataset_path, exist_ok=True)
        import boto3
        s3 = boto3.resource('s3')
        s3.meta.client.download_file(
            s3_bucket, os.path.join(s3_path, dataset_fname), fname_local)
    return fname_local


def objective(config):
    dont_sleep = parse_bool(config['dont_sleep'])

    # Make sure the datafile is on the local filesystem
    path = config['dataset_path']
    os.makedirs(path, exist_ok=True)
    # Lock protection is needed for backends which run multiple worker
    # processes on the same instance
    lock_path = os.path.join(path, 'lock')
    lock = SoftFileLock(lock_path)
    dataset_name = config['dataset_name']
    s3_bucket = config['dataset_s3_bucket']
    try:
        with lock.acquire(timeout=120, poll_intervall=1):
            fname_local = download_datafile(path, dataset_name, s3_bucket)
    except Timeout:
        print(
            "WARNING: Could not obtain lock for dataset files. Trying anyway...",
            flush=True)
        fname_local = download_datafile(path, dataset_name, s3_bucket)

    data = pandas.read_csv(fname_local)
    row = data.loc[(data['x0'] == config['x0']) &
                   (data['x1'] == config['x1']) &
                   (data['x2'] == config['x2']) &
                   (data['x3'] == config['x3']) &
                   (data['x4'] == config['x4']) &
                   (data['x5'] == config['x5'])]

    ts_start = time.time()
    report = Reporter()

    # Checkpointing
    # Since this is a tabular benchmark, checkpointing is not really needed.
    # Still, we use a "checkpoint" file in order to store the epoch at which
    # the evaluation was paused, since this information is not passed

    def load_model_fn(local_path: str) -> int:
        local_filename = os.path.join(local_path, 'checkpoint.json')
        try:
            with open(local_filename, 'r') as f:
                data = json.load(f)
                resume_from = int(data['epoch'])
        except Exception:
            resume_from = 0
        return resume_from

    def save_model_fn(local_path: str, epoch: int):
        os.makedirs(local_path, exist_ok=True)
        local_filename = os.path.join(local_path, 'checkpoint.json')
        with open(local_filename, 'w') as f:
            json.dump({'epoch': str(epoch)}, f)

    resume_from = resume_from_checkpointed_model(config, load_model_fn)

    # Loop over epochs
    elapsed_time_raw = 0
    for epoch in range(resume_from + 1, config['epochs'] + 1):
        y = float(row['lc_valid_epoch_{}'.format(epoch - 1)])
        t = float(row['eval_time_epoch'])
        accuracy = y / 100

        if dont_sleep:
            elapsed_time_raw += t
        else:
            time.sleep(t)
        elapsed_time = time.time() - ts_start + elapsed_time_raw

        report(
            epoch=epoch,
            objective=accuracy,
            elapsed_time=elapsed_time)

        # Write checkpoint (optional)
        if (not dont_sleep) or epoch == config['epochs']:
            checkpoint_model_at_rung_level(config, save_model_fn, epoch)


if __name__ == '__main__':
    # Benchmark-specific imports are done here, in order to avoid import
    # errors if the dependencies are not installed (such errors should happen
    # only when the code is really called)
    from filelock import SoftFileLock, Timeout
    import pandas
    import json

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dont_sleep', type=str, required=True)
    parser.add_argument('--dataset_s3_bucket', type=str)
    add_to_argparse(parser, _config_space)
    add_checkpointing_to_argparse(parser)

    args, _ = parser.parse_known_args()

    objective(config=vars(args))
