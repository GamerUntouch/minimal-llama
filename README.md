# Minimal LLaMA

This repo contains a random assortment of code for running and fine-tuning LLaMA. Many parts are still work in progress. There ought to be more efficient methods of tuning (DeepSpeed / ZeRO, NeoX) than the ones presented here, but folks may find this useful already.

- [Tokenize datasets](#tokenize-datasets)
- [PEFT Fine-tuning with 8-bit](#peft-fine-tuning-with-8-bit)
- [Fine-tuning with Naive Pipeline Parallel](#fine-tuning-with-naive-pipeline-parallel)
- (New) [PEFT Fine-tuning with 8-bit and Pipeline Parallel](#peft-fine-tuning-with-8-bit-and-pipeline-parallel)
- [Misc notes](#misc-notes)

This code was fairly quickly thrown together and may contains many, many bugs. Feedback is welcome!

## Tokenize datasets

First, we tokenize the data so we never have to worry about the tokenizer again. The tokenization script takes in a JSONL (each row containing the key `"text"` for the document text), and effectively concatenates, tokenizes, and slices into `max_seq_length` chunks.

(This is a quick and dirty script that loads the whole dataset into memory.)

```bash
python tokenize_dataset.py \
    --tokenizer_path /path/to/tokenizer \
    --text_path /path/to/data.txt \
    --save_path /path/to/tokenized_dataset \
    --max_seq_length 512
```

## PEFT Fine-tuning with 8-bit

*Requires using the **Transformers** PR [here](https://github.com/huggingface/transformers/pull/21955/), based on the fork [here](https://github.com/zphang/transformers/tree/llama_push). Model weights need to be converted to HF format using the weight conversion script in the PR.*

*Requires using the **PEFT** PR [here](https://github.com/huggingface/peft/pull/160), based on the fork [here](https://github.com/zphang/peft/tree/llama).*


We can fine-tune using the [PEFT](https://github.com/huggingface/peft) library, with the model converted to 8-bit. This is based on the guide [here](https://github.com/huggingface/peft#int8-training-of-large-models-in-colab-using-peft-lora-and-bits_and_bytes).

```bash
python finetune_peft.py \
    --model_path /path/to/llama-7b/ \
    --dataset_path /path/to/tokenized_dataset \
    --peft_mode lora \
    --lora_rank 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_steps 2500 \
    --learning_rate 2e-4 \
    --fp16 \
    --logging_steps 10 \
    --output_dir /path/to/save
```

The above configuration (with `max_seq_length=512`) uses about 20GB of RAM on a single GPU. (With bs=1 and `max_seq_length=256`, this gets down to about 12 GB.)

You can generate using the trained PEFT params using something like the following:

```bash
python test_peft.py \
    --model_path /path/to/llama-7b/ \
    --peft_path /path/to/peft.p \
    --tokenizer_path /path/to/tozenizer \
    --test_input "input text" \
````

## Fine-tuning with Naive Pipeline Parallel

*Requires using the **Transformers** PR [here](https://github.com/huggingface/transformers/pull/21955/), based on the fork [here](https://github.com/zphang/transformers/tree/llama_push). Model weights need to be converted to HF format using the weight conversion script in the PR.*


For *fully* fine-tuning (larger) models, we can use (a very naively implemented version of) pipeline parallelism. This is preferable for larger models that won't fit on a single GPU.

```bash
python finetune_pp.py \
    --model_path /path/to/llama-7b/ \
    --dataset_path /path/to/tokenized_dataset \
    --save_dir /path/to/save \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_interval 2000 \
    --num_train_steps 20000
```

The above configuration uses about 30-35GB of RAM per GPU across 8 GPUs.

## PEFT Fine-tuning with 8-bit and Pipeline Parallel

**Seems buggy, don't use this yet.**

*Requires using the **Transformers** PR [here](https://github.com/huggingface/transformers/pull/21955/), based on the fork [here](https://github.com/zphang/transformers/tree/llama_push). Model weights need to be converted to HF format using the weight conversion script in the PR.*

*Requires using the **PEFT** PR [here](https://github.com/huggingface/peft/pull/160), based on the fork [here](https://github.com/zphang/peft/tree/llama).*

Here, we combine PEFT training with pipeline parallel to train with large models. See [PEFT Fine-tuning with 8-bit](#peft-fine-tuning-with-8-bit) for more details.

```bash
python finetune_pp.py \
    --model_path /path/to/llama-7b/ \
    --dataset_path /path/to/tokenized_dataset \
    --save_dir /path/to/save \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_interval 2000 \
    --num_train_steps 20000
```


For instance, you can fine-tune LoRA on 65B LLaMA with about 120GB of memory in total (e.g. 15GB each on 8 GPUs, or 60GB on 2 GPUs) with batch size=1 and sequence length = 512.

## Merging LoRAs and Base Models

Merges HF converted 8bit weights with LoRA configs. Takes an exceptional amount of RAM to load (30B, for example, requires 145GB+) so be warned.

```bash
python lora_merge.py \
    --model_path /path/to/llama-7b/ \
    --peft_path /path/to/peft \
    --output_path /path/to/output \
    --tokenizer_path /path/to/tozenizer \
```



## Misc Notes

- I have no idea what hyperparameters are best for fine-tuning.
- Aside from model parameters + gradients + optimizer states, the hidden activations also take up a big chunk of memory. Shortening the `max_sequence_length` is a good way of reducing memory consumption. I don't really know how much that affects fine-tuning performance either.
