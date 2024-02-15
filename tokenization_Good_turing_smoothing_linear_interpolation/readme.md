## TOKENIZER

The tokenizer.py script provides a comprehensive text preprocessing and tokenization toolkit. It includes the following features:

- Abbreviation and contraction handling: Expands common contractions and abbreviations (e.g., "can't" becomes "cannot").
- Special token replacement: Recognizes and replaces URLs, email addresses, hashtags, mentions, dates, times, numbers, phone numbers, currency amounts, emoticons, and emojis with corresponding tokens.
- Punctuation and special character handling: Identifies sentence-ending punctuation and isolates punctuation from words.
- Whitespace normalization: Replaces newlines, carriage returns, and tabs with single spaces.
- Tokenization functions: Provides methods for tokenizing text into sentences and words, including n-gram tokenization.

To run the Tokenizer, use the following command:
```
python3 tokenizer.py
```
## LANGUAGE MODEL

N-gram Language Modeling
The language_model.py script allows you to build N-gram language models using different smoothing techniques. It supports two types of models:

Good-Turing Smoothing Model (LM 1 and LM 3): This model uses 3-gram tokenization and employs Good-Turing smoothing for probability estimation.
Linear Interpolation Model (LM 2 and LM 4): This model also uses 3-gram tokenization but uses linear interpolation for probability estimation.
To run the Language Model, use the following command:
```
python3 language_model.py <lm_type> <corpus_path>

```



# GENERATION
## How to Run?

N-gram Language Modeling and Text Generation
The generator.py script builds on the N-gram language models to perform text generation and analysis. It includes:

Text generation: Given an input prompt, it generates text using the selected language model.
Evaluation: It allows you to evaluate the generated text in terms of probability
To run the N-gram Language Model and Text Generation, use the following command:
```
python3 generator.py <lm_type> <corpus_path> <k>
```


1. Using the generated N-gram models (without the smoothing techniques),
try generating sequences. Experiment with different values of N and
report which models perform better in terms of fluency.

when trying to generate next word on unsmoothed data using 3Gram and 2 gram
```
3gram
input sentence: is for the use
of   1.0
the   1.0

2gram
input sentence: is for the use

the   0.5160479725870931
to   0.48395202741290694
```
3gram gives better predictions and fluency

2. Attempt to generate a sentence using an Out-of-Data (OOD) scenario
with your N-gram models. Analyze and discuss the behavior of N-gram
models in OOD contexts.

tested OOD data are

test_model_ood("mhgnbf dgnfb mgnbf ngdbfv hfngbd yjdsfs")

test_model_ood("gdbf dgnrjtedffb sfbzvsd asfasdv hfkjgmhngbd esdfh")

for LM2 the next words possible, incase of OOD are 
```
the     0.5160479725870931
to      0.48395202741290694
perplexity: 1000
the     .5160479725870931
to      0.48395202741290694
perplexity: 1000
```
for LM2 the next words possible, incase of OOD are 
```
the      0.6468452304669144
of      0.35315476953308556
perplexity: 1000
the 0.6468452304669144
of  0.35315476953308556
perplexity: 1000
```

The model is designed to output the most occuring unigrams in case of OOD data.
different LMs behave similarly in generationg the data, but the probabilty of occurances for next word is different.

3. Now try to generate text using the models with the smoothing tech-
niques (LM1, LM2, LM3, LM4).

when trying to generate next word on unsmoothed data using 3Gram
```


test_model_ood("It is a truth universally ")

test_model_ood("You are over scrupulous ")


LM1
```

output is
```
acknowledged 1
the 1
Perplexity: 9.172078918680583

surely 1
the 1
Perplexity : 12.811944308807373


```
LM2
```

output is
```
acknowledged
the
Perplexity: 31.774286462361793

surely
the
Perplexity : 44.345715405560824


```


test_model_ood("Stephen Dedalus ")

test_model_ood("He peered sideways ")


LM3
```

output is
```
and   0.5
<initial>   0.5
Perplexity : 49.48803894935499

up   1.0
the   1.0
Perplexity : 35.35133915321014

```

LM4


output is
```
and   0.5626554026470085
<initial>   0.43734459735299136
Perplexity : 274.9207715424033

up   1.0
the   1.0
Perplexity : 132.7228076526619

```


all models gives good first word predictions.
LM3 and LM4 gives relativeley high perplexities



# ASSUMPTIONS
 - the unseen probability of good turing smoothing is taken as 1/1000 for better estimation of unseen items
