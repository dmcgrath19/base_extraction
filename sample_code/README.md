# Extracting Training Data from Large Langauge Models

A re-implementation of the "Extracting Training Data from Large Language Models" paper by Carlini et al. The [paper](https://arxiv.org/abs/2012.07805) already has an official implementation - [https://github.com/ftramer/LM_Memorization](https://github.com/ftramer/LM_Memorization), from which I have borrowed parts of the code, at the same time improving the readability of a few functions.

However, the official repository does not cover - 
- Sampling Method - Sampling With A Decaying Temperature (Section 5.1.1 of the paper)
- Metric - Minimum perplexity on a Sliding Window (Section 5.2 of the paper)

I was really fascinated with the paper and wanted to implement it myself. Like the official implementation, I have also included a [Samples.md](Samples.md) file, which has some of the memorized content that I could extract from GPT-2. Although I am able to find some interesting memorized content, the results still have a few limitations -

- Due to compute time constraints, I could only generate 25,000 samples for each sampling method (as compared to 600,000 generated by the paper authors)
- Due to memory constraints, I was not able to incorporate the metric - ratio of log-perplexities of GPT2-XL and GPT2-Medium. I have included the code for that in script and if one has sufficient compute, they can uncomment the relevant lines and incorporate that metric as well.

## Requirements

* PyTorch 
* Transformers
* Numpy
* Tqdm

Or, directly 

`pip install -r requirements.txt`

## Extracting Data

### Metrics for Ranking

The generated samples are ranked according to six membership inference metrics introduced in the paper:

- The log-perplexity of the GPT2-XL model
- The ratio of the log-perplexities of the GPT2-XL model and the GPT2-Small model
- The ratio of the log-perplexities of the GPT2-XL model and the GPT2-Medium model (implemented but not couldn't be run due to compute constraints)
- The ratio of the log perplexity of GPT2-XL and the sample's entropy estimated by Zlib
- The ratio of the log-perplexities of the GPT2-XL for the generated sample and the same sample in lower-case letters
- The minimum log-perplexity of GPT2-XL on window of size 50

The top 10 samples according to each metric are printed out, and the top 100 samples according to each metric ae logged in the *outfile*. These samples are likely to contain verbatim text from the GPT-2 training data.


### Top-k sampling

```
python extraction_top_n.py --N 5000 --batch_size 20 --outfile top_n_samples.txt
```

This generates 5000 samples with GPT2-XL. The samples are generated with top-k sampling (k=40) and an empty prompt.

### Temperature Decay

```
python extraction_temperature_decay.py --N 5000 --batch_size 20 --outfile temperature_decay_samples.txt
```

This generates 5000 samples with GPT2-XL. The samples are generated with sampling with temperature decay (decay the softmax temperature from 10 to 1 or the first 20 tokens and 1 for all subsequent tokens) and an empty prompt.

### Conditioning on Internet text

In the paper, the authors also tried prompting the GT2-XL model with snippets of text from the web (commoncrawl) which increased the chance of the model generating memorized content.

I used the same sample of the Crawl from May 2021 (~350 MB) used by the authors.

```
./download_cc.sh
```

Then,

```
python extraction_commoncrawl.py --N 5000 --batch_size 20 --outfile commoncrawl_samples.txt
```

All the generated sequences have a final length of atmost 256 tokens.

## Sample outputs

Some interesting outputs that were extracted from GPT-2 can be found [here](Samples.md).