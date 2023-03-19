import argparse
import torch
import transformers
from finetune_peft import get_peft_config, PEFTArguments
from peft import get_peft_model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--peft_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--test_input", type=str)
    args = parser.parse_args()
    
 
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = transformers.LlamaForCausalLM.from_pretrained(args.model_path)
    peft_config = get_peft_config(peft_args=PEFTArguments(peft_mode="lora"))
    model = get_peft_model(model, peft_config)
    model.load_state_dict(torch.load(args.peft_path), strict=False)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    tokenizer = transformers.LLaMATokenizer.from_pretrained(args.tokenizer_path)
    batch = tokenizer(args.test_input, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=torch.ones_like(batch["input_ids"]),
            max_length=200,
        )
    
    print(tokenizer.decode(out[0]))

if __name__ == "__main__":
    main()
