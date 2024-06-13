import argparse
import numpy as np
import sys
import torch
import zlib
import csv
import pandas as pd
# import transformers.generation_logits_process
import transformers.generation.logits_process
from model_utils import DecayingTemperatureWarper, calculate_perplexity_sliding, calculate_perplexity, parse_pilecorpus, parse_lang, print_best, device
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
# from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def main(args):
    print(f"Using device: {device}")
    print("Loading dataset...")
    # path="swa_sample.txt"
    ds= parse_pilecorpus(args.corpus_path)
    print("Length:", len(ds))
    # print("The sample of dataset is:", ds[:1000])
   
    seq_len = 256
    logits_warper = LogitsProcessorList(
            [
                DecayingTemperatureWarper(10.0)
            ]
        )
    
    # top_k = 1000

    print("Loading models...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model1)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
      
    model1 = GPTNeoXForCausalLM.from_pretrained(args.model1, return_dict=True).to(device)
    model1.config.pad_token_id = model1.config.eos_token_id
    model2 = GPTNeoXForCausalLM.from_pretrained(args.model2, return_dict=True).to(device)
    model2.eval()
    
    samples = []
    prompts_list = []
    scores = {"XL": [], "S": [], "Lower": [], "zlib": [], "window": []}

    num_batches = int(np.ceil(args.N / args.batch_size))
    
    with tqdm(total=args.N) as pbar:
        for _ in range(num_batches):
            
            input_len = 10
            input_ids = []
            attention_mask = []

            while len(input_ids) < args.batch_size:
                # Sample random text from the Pile corpus
                r = np.random.randint(0, len(ds))
                
                # print("*"*100)
                # print("The index Selected is:", r)
                
                
                prompt = " ".join(ds[r:r+100].split(" ")[1:-1])
                
                
                # print("The untruncated prompt is:",prompt)

                # Tokenize the prompt ensuring consistent input lengths
                inputs = tokenizer(prompt, return_tensors="pt", max_length=input_len, truncation=True, padding="max_length")
                if len(inputs['input_ids'][0]) == input_len:
                    input_ids.append(inputs['input_ids'][0])
                    attention_mask.append(inputs['attention_mask'][0])

            inputs = {'input_ids': torch.stack(input_ids), 
                      'attention_mask': torch.stack(attention_mask)}

            # The actual truncated prompts
            prompts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            # print("Sample truncated prompt to check:", prompts)
            # print("*"*100)
            # print("Length of prompt tensor:", len(inputs))    
            # print(inputs)
            # print("Input IDs shape:", inputs['input_ids'].shape)
            print("Attention Mask shape:", inputs['attention_mask'].shape)

            output_sequences = model1.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                do_sample=True, 
                # top_k=top_k, 
                logits_processor = logits_warper,
                renormalize_logits = True
                # top_p=1.0
            )

            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            prompts_list.append(prompts)
            for text in texts:
                p1 = calculate_perplexity(text, model1, tokenizer)
                p2 = calculate_perplexity(text, model2, tokenizer)
                p_lower = calculate_perplexity(text.lower(), model1, tokenizer)
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
                perplexity_window = calculate_perplexity_sliding(text.lower(), model1, tokenizer, device)
                samples.append(text)
                
                
                scores["XL"].append(p1)
                scores["S"].append(p2)
                scores["Lower"].append(p_lower)
                scores["zlib"].append(zlib_entropy)
                scores["window"].append(perplexity_window.cpu())
            pbar.update(args.batch_size)
    # print("*"*100)
    # print("Prompt List has the following prompts:",prompts_list[0])
    scores["XL"] = np.asarray(scores["XL"])
    scores["S"] = np.asarray(scores["S"])
    scores["Lower"] = np.asarray(scores["Lower"])
    scores["zlib"] = np.asarray(scores["zlib"])
    scores["window"] = np.asarray(scores["window"])
    model1_name = args.model1.replace("/", "_")
    model2_name = args.model2.replace("/", "_")
    
     # Remove duplicate samples
    idxs = pd.Index(samples)
    idxs_mask = ~(idxs.duplicated())
    print(idxs_mask)
    generated_samples_clean = np.asarray(samples)[idxs_mask]
    generated_samples_clean = samples.tolist()
    scores["XL"] = scores["XL"][idxs_mask]
    scores["S"] = scores["S"][idxs_mask]
    # scores["MEDIUM"] = scores["MEDIUM"][idxs_mask]
    scores["Lower"] = scores["Lower"][idxs_mask]
    scores["zlib"] = scores["zlib"][idxs_mask]
    scores["window"] = scores["window"][idxs_mask]

    assert len(generated_samples_clean) == len(scores["XL"])
    assert len(scores["S"]) == len(scores["XL"])
    print("Num duplicates:", len(samples) - len(generated_samples_clean))

    output_csv = f'output_scores_{model1_name}_{model2_name}.csv'
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['sample', 'prompt', 'PPL_XL', 'PPL_S', 'PPL_Lower', 'Zlib']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample,prompt, xl, s, lower, zlib_,window in zip(generated_samples_clean, prompts_list[0], scores["XL"], scores["S"], scores["Lower"], scores["zlib"], scores["window"]):
            writer.writerow({'sample': sample, 'prompt': prompt,'PPL_XL': xl, 'PPL_S': s, 'PPL_Lower': lower, 'Zlib': zlib_, 'window': window})

    print("Results saved to ", output_csv)

    output_txt = f'output_results_{model1_name}_{model2_name}.txt'
    with open(output_txt, 'w') as f:
        metric = -np.log(scores["XL"])
        f.write(f"======== top sample by XL perplexity: ========\n")
        f.write(print_best(metric, generated_samples_clean, "PPL", scores["XL"]))
        f.write("\n")

        metric = np.log(scores["S"]) / np.log(scores["XL"])
        f.write(f"======== top sample by ratio of S and XL perplexities: ========\n")
        f.write(print_best(metric, generated_samples_clean, "PPL-XL", scores["XL"], "PPL-S", scores["S"]))
        f.write("\n")

        metric = np.log(scores["Lower"]) / np.log(scores["XL"])
        f.write(f"======== top sample by ratio of lower-case and normal-case perplexities: ========\n")
        f.write(print_best(metric, generated_samples_clean, "PPL-XL", scores["XL"], "PPL-XL-Lower", scores["Lower"]))
        f.write("\n")

        metric = scores["zlib"] / np.log(scores["XL"])
        f.write(f"======== top sample by ratio of Zlib entropy and XL perplexity: ========\n")
        f.write(print_best(metric, generated_samples_clean, "PPL-XL", scores["XL"], "Zlib", scores["zlib"]))

        metric = scores["window"]
        f.write(f"========  top samples by minimum XL perplexity across a sliding window of size 50: ========\n")
        f.write(print_best(metric, generated_samples_clean, "PPL-Window", scores["window"]))
    print("Top results written to ", output_txt)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=10, help="Batch size for generation")
    parser.add_argument('--model1', type=str, required=True, help="Hugging Face model name for the first model")
    parser.add_argument('--model2', type=str, required=True, help="Hugging Face model name for the second model")
    parser.add_argument('--corpus-path', type=str, required=True, help="Path to the corpus dataset")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
