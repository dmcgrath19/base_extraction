import argparse
import numpy as np
import sys
import math
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation_logits_process import LogitsProcessor, LogitsProcessorList
from collections import defaultdict
from model_utils import parse_pilecorpus, parse_lang
from tqdm import tqdm
from pprint import pprint
import pandas as pd

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

LOW_MEMORY = True


# Custom LogitProcessor to decay Temperature from 10.0 to 1.0 over the first 20 tokens
# and 1.0 for subsequent tokens 
class DecayingTemperatureWarper(LogitsProcessor):
    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature
        self.mapping = {1: 10.0, 2: 9.53, 3: 9.06, 4: 8.59, 5: 8.12, 6: 7.65, 7: 7.18, 8: 6.71, 9: 6.24, 10: 5.77, 11: 5.30, 
                        12: 4.83, 13: 4.36, 14: 3.89, 15: 3.42, 16: 2.95, 17: 2.49, 18: 2.01, 19: 1.54, 20: 1.0}

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        self.temperature = self.mapping.get(cur_len, 1.0)
        
        return scores

def load_tokenizer_for_causal_lm(model_name):
    """
    Load tokenizer with required config changes
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # For Autoregressive models, padding on the right would mean the model 
    # will receive padded tokens as context, which is not useful during generation
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def load_model_for_causal_lm(model_name, device):
    """
    Load model with required config changes
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=LOW_MEMORY).to(device)

    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    return model


def calculate_perplexity(input_sentence, model, tokenizer, device):
    """
    Calculate exp(loss), where loss is obtained py passing tokenized input sentence to the model
    with the labels set as the same tokenized input (the shifting of the labels is done internally)
    https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/gpt2#transformers.GPT2LMHeadModel.forward.labels
    """
    tokenized = tokenizer(input_sentence)
    input = torch.tensor(tokenized.input_ids).to(device)
    with torch.no_grad():
        output = model(input, labels=input)
    
    return torch.exp(output.loss)

def calculate_perplexity_sliding(input_sentence, model, tokenizer, device, window_size=50):
    """
    Calculate min(exp(loss)) over a sliding window
    """
    tokenized = tokenizer(input_sentence)
    input = torch.tensor(tokenized.input_ids).to(device)
    min_perplexity = 100000
    with torch.no_grad():
        for start_idx in range(input.shape[0]-window_size):
            input_window = input[start_idx: start_idx+window_size]
            output = model(input_window, labels=input_window)
            min_perplexity = min(min_perplexity, torch.exp(output.loss))
    return min_perplexity

def print_best(metric, samples, metric_name, name1, scores1, name2=None, scores2=None, lower_better=True, n=1000):
    """
    Print the top-n best samples according to the given metric
    """
    if lower_better:
        idxs = np.argsort(metric)[:n]
    else:
        idxs = np.argsort(metric)[::-1][:n]

    print("Metric Name:", metric_name)
    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

        print()
        pprint(samples[idx])
        print()
        print()

def print_best_to_file(outfile, metric, samples, metric_name, name1, scores1, name2=None, scores2=None, lower_better=True, n=1000):
    """
    Print the top-n best samples according to the given metric to a file
    """
    original_stdout = sys.stdout # Save a reference to the original standard output

    with open(outfile, 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print("Metric Name:", metric_name)

        if lower_better:
            idxs = np.argsort(metric)[:n]
        else:
            idxs = np.argsort(metric)[::-1][:n]

        for i, idx in enumerate(idxs):
            if scores2 is not None:
                print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
            else:
                print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

            print()
            print(samples[idx])
            print()
            print()
        
        print()
        print()
        sys.stdout = original_stdout # Reset the standard output to its original value

def main(args):
    # Load models
    print("Loading models...")
    tokeniser = load_tokenizer_for_causal_lm(args.model2)
    small_model = load_model_for_causal_lm(args.model2, device)
    # MODEL_GPT2_MEDIUM = load_model_for_causal_lm(gpt2-medium", device)
    xl_model = load_model_for_causal_lm(args.model1, device)
    print("small and XL models loaded!")
    ds= parse_pilecorpus(path)
    print("Length:", len(ds))
    # number of tokens to generate (from paper)
    seq_len = 256

    # Initialize the custom warper
    logits_warper = LogitsProcessorList(
            [
                DecayingTemperatureWarper(10.0)
            ]
        )

    num_batches = int(math.ceil(args.N / args.batch_size))
    new_tot = num_batches * args.batch_size

    generated_samples = []
    scores = defaultdict(list)
    prompts_list = []
    with tqdm(total=new_tot) as pbar:
        for batch in range(num_batches):
            # Create empty prompts
            input_len = 10
            input_ids = []
            attention_mask = []
            while len(input_ids) < args.batch_size:
                # Sample random text from the Pile corpus
                r = np.random.randint(0, len(ds))
                # prompt = " ".join(ds[r].split()[:100])
                prompt = " ".join(ds[r:r+100].split(" ")[1:-1])
                # Tokenize the prompt ensuring consistent input lengths
                inputs = tokeniser(prompt, return_tensors="pt", max_length=input_len, truncation=True, padding="max_length")
                if len(inputs['input_ids'][0]) == input_len:
                    input_ids.append(inputs['input_ids'][0])
                    attention_mask.append(inputs['attention_mask'][0])

            inputs = {'input_ids': torch.stack(input_ids), 
                      'attention_mask': torch.stack(attention_mask)}
            
            prompts = tokeniser.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            # prompts = [tokeniser.eos_token] * args.batch_size
            # inputs = tokeniser(prompts, return_tensors="pt", padding=True).to(device)
            
            # Batched sequence generation
            generated_sequences = xl_model.generate(
                input_ids = inputs.input_ids,
                attention_mask = inputs.attention_mask,
                max_length = seq_len,
                do_sample = True, 
                logits_processor = logits_warper,
                renormalize_logits = True
            )

            generated_texts = tokeniser.batch_decode(generated_sequences, skip_special_tokens=True)

            for text in generated_texts:
                # Calculate perplexity of XL, Small and Medium on each generated text
                perplexity_xl = calculate_perplexity(text, xl_model, tokeniser, device) 
                perplexity_small = calculate_perplexity(text, small_model, tokeniser, device) 
                # perplexity_medium = calculate_perplexity(text, MODEL_MEDIUM, tokeniser, device) 

                # Calculate perplexity of XL on each lower-cased text
                perplexity_xl_lower = calculate_perplexity(text.lower(), xl_model, tokeniser, device) 

                # Calculate Z-lib entropy of sample
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                # Calculate minimum perplexity of GPT2-XL across any sliding window of 50 tokens
                perplexity_xl_window = calculate_perplexity(text.lower(), xl_model, tokeniser, device)

                generated_samples.append(text)
                scores["XL"].append(perplexity_xl.cpu())
                scores["SMALL"].append(perplexity_small.cpu())
                # scores["MEDIUM"].append(perplexity_gpt2_medium.cpu())
                scores["ZLIB"].append(zlib_entropy)
                scores["LOWER"].append(perplexity_xl_lower.cpu())
                scores["WINDOW"].append(perplexity_xl_window.cpu())
                prompts_list.append(prompts)
            pbar.update(args.batch_size)

    print(len(scores["XL"]))
    scores["XL"] = np.asarray(scores["XL"])
    scores["SMALL"] = np.asarray(scores["SMALL"])
    # scores["MEDIUM"] = np.asarray(scores["MEDIUM"])
    scores["ZLIB"] = np.asarray(scores["ZLIB"])
    scores["LOWER"] = np.asarray(scores["LOWER"])
    scores["WINDOW"] = np.asarray(scores["WINDOW"])

    # Remove duplicate samples
    idxs = pd.Index(generated_samples)
    idxs_mask = ~(idxs.duplicated())
    print(idxs_mask)
    generated_samples_clean = np.asarray(generated_samples)[idxs_mask]
    generated_samples_clean = generated_samples_clean.tolist()
    scores["XL"] = scores["XL"][idxs_mask]
    scores["SMALL"] = scores["SMALL"][idxs_mask]
    # scores["MEDIUM"] = scores["MEDIUM"][idxs_mask]
    scores["ZLIB"] = scores["ZLIB"][idxs_mask]
    scores["LOWER"] = scores["LOWER"][idxs_mask]
    scores["WINDOW"] = scores["WINDOW"][idxs_mask]

    assert len(generated_samples_clean) == len(scores["XL"])
    assert len(scores["SMALL"]) == len(scores["XL"])
    print("Num duplicates:", len(generated_samples) - len(generated_samples_clean))
    
    # Show best samples based on Metrics
    # Sort by perplexity of GPT2-XL
    metric = np.log(scores["XL"])
    print(f"======== top samples by XL perplexity: ========")
    print_best(metric, generated_samples_clean, "Sort by perplexity of XL", "PPL-XL", scores["XL"], lower_better=True)
    print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by perplexity of GPT2-XL", "PPL-XL", scores["XL"], lower_better=True)
    print()
    print()

    # Sort by ratio of perplexity of GPT2-XL and GPT2-Small
    metric = np.log(scores["XL"]) / np.log(scores["SMALL"])
    print(f"======== top samples by ratio of XL and SMALL perplexity: ========")
    print_best(metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL and GPT2-Small", "PPL-XL", scores["XL"], "PPL-SMALL", scores["SMALL"], lower_better=True)
    print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL and GPT2-Small", "PPL-XL", scores["XL"], "PPL-SMALL", scores["SMALL"], lower_better=True)
    print()
    print()

    # Sort by ratio of perplexity of GPT2-XL and GPT2-Medium
    # metric = np.log(scores["XL"]) / np.log(scores["MEDIUM"])
    # print(f"======== top samples by ratio of XL and SMALL perplexity: ========")
    # print_best(metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL and GPT2-Medium", "PPL-XL", scores["XL"], "PPL-MEDIUM", scores["MEDIUM"], lower_better=True)
    # print_best_to_file(metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL and GPT2-Medium", "PPL-XL", scores["XL"], "PPL-MEDIUM", scores["MEDIUM"], lower_better=True)
    # print()
    # print()

    # Sort by ratio of XL perplexity and ZLIB entropy
    metric = np.log(scores["XL"]) / np.log(scores["ZLIB"])
    print(f"======== top samples by ratio of XL perplexity and ZLIB entropy: ========")
    print_best(metric, generated_samples_clean, "Sort by ratio of XL perplexity and ZLIB entropy", "PPL-XL", scores["XL"], "Entropy-Zlib", scores["ZLIB"], lower_better=True)
    print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by ratio of XL perplexity and ZLIB entropy", "PPL-XL", scores["XL"], "Entropy-Zlib", scores["ZLIB"], lower_better=True)
    print()
    print()

    # Sort by ratio of perplexity of GPT2-XL on normal and lower-cased sample
    metric = np.log(scores["XL"]) / np.log(scores["LOWER"])
    print(f"======== top samples by ratio of perplexity of GPT2-XL on normal and lower-cased sample: ========")
    print_best(metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL on normal and lower-cased sample", "PPL-XL", scores["XL"], "PPL-XL-Lower", scores["LOWER"], lower_better=True)
    print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL on normal and lower-cased sample", "PPL-XL", scores["XL"], "PPL-XL-Lower", scores["LOWER"], lower_better=True)
    print()
    print()

    # Sort by minimum perplexity of GPT2-XL on window of size 50
    metric = np.log(scores["WINDOW"])
    print(f"======== top samples by minimum XL perplexity across a sliding window of size 50: ========")
    print_best(metric, generated_samples_clean, "Sort by minimum perplexity of GPT2-XL on window of size 50", "PPL-WINDOW", scores["WINDOW"], lower_better=True)
    print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by minimum perplexity of GPT2-XL on window of size 50", "PPL-WINDOW", scores["WINDOW"], lower_better=True)
    print()
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=20, type=int, help='Number of samples to generate')
    parser.add_argument('--batch_size', default=6, type=int, help='Batch size')
    parser.add_argument('--model1', type=str, required=True, help="Hugging Face model name for the large, first model")
    parser.add_argument('--model2', type=str, required=True, help="Hugging Face model name for the small, second model")
    parser.add_argument('--corpus-path', type=str, required=True, help="Path to the corpus dataset")
    parser.add_argument('--outfile', type=str, help='Output file to log top samples based on each metric')

    args = parser.parse_args()

    main(args)