# Neural Architecture Search for Large Language Models


This example shows how to use Syne Tune for multi-objective Neural Architecture Search (NAS)
to prune pre-trained Large Language Models. 
Our approach consists of two stages:
1. We fine-tune the pre-trained network (dubbed super-network) via weight-sharing based NAS strategies. 
   In a nutshell, in each update steps, we only update parts of the network to train different sub-networks.
2. In the second stage, we run multi-objective search to find the Pareto set of sub-networks 
   of the super-network. To evaluate each sub-network we use the shared weights of the super-networks, without
   any further training. This is relatively cheap compared to standard NAS, since we only do a single pass
   over the validation data without computing gradients. To account for the overhead of loading the model and 
   the dataset, we use an ask/tell interface of Syne Tune, such that the full optimization process run in a single
   python process.

## Install

To get started, install the dependencies via:

```bash
cd benchmarking/nursery/nas_llm
pip install -r requirements.txt
```

## Super-Network Training

To run the training of the super-network, execute the following script:

```python train_supernet.py --fp16 True --learning_rate 2e-05 --model_name_or_path bert-base-cased --num_train_epochs 20 --output_dir ./supernet_model_checkpoint --per_device_eval_batch_size 8 --per_device_train_batch_size 4 --sampling_strategy one_shot --save_strategy epoch --search_space small --seed 0 --task_name rte``` 

This runs the super-network training ('one_shot') on the RTE dataset for 20 epochs. Checkpoints are saved in the
`output_dir`, such that we can load it later for the multi-objective search.

### Distributed Training via Accelerate

For larger models, for example GPT-2-XL, we have to distribute the training across multiple GPUs on the same instance via HuggingFace Accelearte and DeepSpeed.
To run the supernet-training, we have to run it via the `accelerate_launcher.py` which invokes our actual training script. 
Make sure to add `--use_accelerate True` to your training arguments. This will make sure that checkpoints are saved in the correct format.

```python accelerate_launcher.py --config_file default_config.yaml  --training_script train_supernet.py --fp16 True --learning_rate 2e-05 --model_name_or_path gpt2-xl --num_train_epochs 10 --output_dir ./supernet_model_checkpoint --per_device_eval_batch_size 8 --per_device_train_batch_size 4 --sampling_strategy one_shot --save_strategy epoch --search_space small --seed 0 --task_name rte --use_accelerate True```

You can customize deepspeed by updated `default_config.yaml`. For more information, see `https://huggingface.co/docs/accelerate/v0.11.0/en/deepspeed#how-it-works`

### Running it on SageMaker

If you want to run this on SageMaker, you can run the following launcher script:

```python launch_supernet_training.py``` 

This will run the training in a SageMaker Training Job. The model checkpoints are automatically uploaded to S3, such that we can load them later for the multi-objective
search.

### Tensorboard Visualization

To visualize the super-network training, run:

```bash
tensorboard --logdir  tensorboard_log_dir
```

If you run experiments on SageMaker, logs are automatically uploaded to visualize them with SageMaker's Tensorboard
visualization, which you can access via the SageMaker console. 
See here for more information: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-tensorboard.html

## Multi-Objective Search

Next, we use the model checkpoint from the previous step to perform the multi-objective search:

```python run_offline_search.py --model_name_or_path bert-base-cased --num_samples 500 --output_dir ./results_nas  --checkpoint_dir_model ./supernet_model_checkpoint --search_space small --search_strategy local_search --seed 0 --task_name rte``` 

Make sure that `checkpoint_dir_model` points to the directory with the model checkpoint from the previous step. 
Results will be saved as a json file in `output_dir`.

### Running it on SageMaker

We can also run this on SageMaker:

```python launch_offline_search.py``` 


This script will automatically download the model checkpoint created during the super-network training from S3.
After the search, results are uploaded to S3.

## Select the Optimal Model

TBD

## Run all Experiments

TBD