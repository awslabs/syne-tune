# Neural Architecture Search for Large Language Models


This example shows how to use Syne Tune for multi-objective Neural Architecture Search (NAS)
to prune pre-trained Large Language Models. 
Our approach consists of two stages:
1. We fine-tune the pre-trained network (dubbed super-network) via weight-sharing based NAS strategies. 
   In a nutshell, in each update steps, we only update parts of the network to train different sub-networks.
2. In the second stage, we run multi-objective search to find the Parteo set of sub-networks 
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

```python train_supernet.py``` 

If you want to run this on SageMaker, you can run the following launcher script:

This will automatically upload the model checkpoints to S3, such that we can load them later for the multi-objective
search.

## Multi-Objective Search

Next, we use the model checkpoint from the previous step to perform the multi-objective search:

```python run_offline_search.py``` 

We can also run this on SageMaker:

This script will automatically download the model checkpoint and upload the results to S3.

## Select the Optimal Model

At the end we can visualize the Pareto set of sub-networks: TBD

To select a network we instantiate a new model and copy the checkpoint. TBD
