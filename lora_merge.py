from dataclasses import dataclass, field
from typing import Optional

import peft
import torch
import argparse
from peft import PeftConfig, PeftModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--peft_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    
    print("Merging...")
    
    base_model_name_or_path = args.model_path

    peft_config = PeftConfig.from_pretrained(args.peft_path)
    
    model = transformers.LlamaForCausalLM.from_pretrained(args.model_path)
    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.tokenizer_path)

    # Load the LoRA model
    model = PeftModel.from_pretrained(model, args.peft_path)
    model.eval()

    key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
    for key in key_list:
        parent, target, target_name = model.base_model._get_submodules(key)
        if isinstance(target, peft.tuners.lora.Linear):
            bias = target.bias is not None
            new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
            model.base_model._replace_module(parent, target_name, new_module, target)

    model = model.base_model.model

    model.save_pretrained(args.output_path)

    print("Merged and saved to" + args.output_path + ".")
    
if __name__ == "__main__":
    main()
