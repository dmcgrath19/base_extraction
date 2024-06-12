import torch
import numpy as np
import logging
from datasets import load_dataset
logging.basicConfig(level='ERROR')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def parse_lang(path):
    file_content=""
    chunk_size = 10 * 1024 * 1024  # 10 MB

    try:
        # Open the file in read mode
        with open(path, 'r', encoding='utf-8') as file:
            while True:
                # Read the next chunk from the file
                chunk = file.read(chunk_size)
                if not chunk:
                    break  # End of file reached
                # Append the chunk to the file content string
                file_content += chunk
        print("File read successfully.")
    except FileNotFoundError:
        print(f"The file at {path} was not found.")
    except IOError as e:
        print(f"An error occurred while reading the file at {path}: {e}")
    
    return file_content


def parse_pilecorpus(path):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    
    all_texts = ""
    dataset = load_dataset(path, split="train", streaming=True)
    shuffled_dataset = dataset.shuffle(seed=42)
    #len(dataset['train'])
    dataset_head= shuffled_dataset.skip(0)
    dataset_head = shuffled_dataset.take(1000000)
    for text in dataset_head:
        all_texts+= text['text']

    return all_texts

def calculate_perplexity(sentence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=10):
    """
    Print the `n` best samples according to the given `metric`.
    Returns a string containing the information for each sample.
    """
    idxs = np.argsort(metric)[::-1][:n]
    output_string = ""

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            sample_info = f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}"
        else:
            sample_info = f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}"

        sample_text = samples[idx]
        output_string += sample_info + "\n" + sample_text + "\n\n"

    return output_string