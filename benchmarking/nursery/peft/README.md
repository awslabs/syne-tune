
# Parameter Efficient Fine-Tuning

NOTE: Training is **automatically** distributed across all GPUs on the instance.


Run LORA with BERT model on RTE dataset:

```
python run_peft.py --task_name rte --peft_method lora --model_name_or_path bert-base-cased --output_dir peft_output_dir
```

Run vanilla fine-tuning
```
python run_peft.py --task_name rte --peft_method fine_tuning --model_name_or_path bert-base-cased --output_dir peft_output_dir
```

Save checkpoints after each epoch:
```
python run_peft.py --task_name rte --peft_method fine_tuning --model_name_or_path bert-base-cased --output_dir peft_output_dir --save_strategy epoch --save_total_limit 1
```


## Supported Datasets:

### Multi-choice Q/A

- SWAG

### Sequence Classification (GLUE)

- RTE
- MRPC
- COLA
- STSB
- MNLI
- QNLI
- QQP
- SST2
