import random
import re
import numpy as np
import math
from scipy.stats import linregress
import sys
import os



class Tokenizer:
    def __init__(self):
        self.stop_words = set(['i', 'a', 's', 't', 'd', 'm', 'o', 'y', 'me', 'my',
            'we', 'he', 'it', 'am', 'is', 'be', 'do', 'an', 'if', 'or',
            'as', 'of', 'at', 'by', 'to', 'up', 'in', 'on', 'no', 'so',
            'll', 're', 've', 'ma', 'our', 'you', 'him', 'his', 'she',
            'her', 'its', 'who', 'are', 'was', 'has', 'had', 'did', 'the',
            'and', 'but', 'for', 'out', 'off', 'why', 'how', 'all', 'any',
            'few', 'nor', 'not', 'own', 'too', 'can', 'don', 'now', 'ain',
            'isn', 'won', 'ours', 'your', 'hers', 'it', 'is', 'they', 'them',
            'what', 'whom', 'this', 'that', 'were', 'been', 'have', 'does',
            'with', 'into', 'from', 'down', 'over', 'then', 'once', 'here',
            'when', 'both', 'each', 'more', 'most', 'some', 'such', 'only',
            'same', 'than', 'very', 'will', 'just', 'aren', 'didn', 'hadn',
            'hasn', 'shan', 'wasn', 'you', 'would', 'yours', 'she', 'is', 'their',
            'which', 'these', 'those', 'being', 'doing', 'until', 'while',
            'about', 'after', 'above', 'below', 'under', 'again', 'there',
            'where', 'other', 'do', 'not', 'doesn', 'haven', 'is', 'not', 'mustn',
            'needn', 'weren', 'will', 'not', 'myself', 'you', 'are', 'you', 'have',
            'you', 'will', 'itself', 'theirs', 'having', 'during', 'before',
            'should', 'are', 'not', 'couldn', 'did', 'not', 'had', 'not', 'has', 'not',
            'mightn', 'shall', 'not', 'was', 'not', 'wouldn', 'himself', 'herself',
            'that', 'will', 'because', 'against', 'between', 'through',
            'further', 'does', 'not', 'have', 'not', 'must', 'not', 'need', 'not',
            'shouldn', 'were', 'not', 'yourself', 'could', 'not', 'might', 'not',
            'would', 'not', 'ourselves', 'should', 'have', 'should', 'not', 'yourselves',
            'themselves'])

        self.regex_patterns = {
            '<url>': re.compile(r'https?://\S+|www\.\S+'),
            '<hashtag>': re.compile(r'\B#\w*[a-zA-Z]+\w*'),
            '<mention>': re.compile(r'\B@\w*[a-zA-Z]+\w*'),
            '<subscript>': re.compile(r'[A-Za-z0-9]+\^\{[A-Za-z0-9]+\}'),
            '<number>': re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b'),
            '<mailid>': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            '<date>': re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b'),
            '<time>': re.compile(r'\b(?:[01]?\d|2[0-3]):[0-5]\d\b'),
            '<initial>': re.compile(r'\b[A-Z]\.', re.IGNORECASE),
            '<title>': re.compile(r'(Mr|Mrs|Ms|Dr|Prof)\.', re.IGNORECASE),
            '<phoneno>': re.compile(r'\b(?:\+\d{1,4}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'),
            ' ': re.compile(r'[\,\!\:\(\)\[\]\'\"\_\-\*]'),
        }

        self.general_token_pattern = re.compile(r'<\w+>|\b\w+\b|[^\w\s<>]+')
        self.contractions = {
            "can't": 'can not', "don't": 'do not', "won't": 'will not', "you're": 'you are', "you've": 'you have',
            "you'll": 'you will', "you'd": 'you would', "he's": 'he is', "he'd": 'he would', "he'll": 'he will',
            "she's": 'she is', "she'd": 'she would', "she'll": 'she will', "it's": 'it is', "it'd": 'it would',
            "it'll": 'it will', "we're": 'we are', "we've": 'we have', "we'll": 'we will', "we'd": 'we would',
            "they're": 'they are', "they've": 'they have', "they'll": 'they will', "they'd": 'they would',
            "that's": 'that is', "that'd": 'that would', "that'll": 'that will', "who's": 'who is',
            "who'll": 'who will', "who'd": 'who would', "what's": 'what is', "what'll": 'what will',
            "what'd": 'what would', "where's": 'where is', "where'd": 'where would', "where'll": 'where will',
            "when's": 'when is', "when'd": 'when would', "when'll": 'when will', "why's": 'why is',
            "why'd": 'why would', "why'll": 'why will',
            "I'm": 'I am', "I've": 'I have', "I'll": 'I will', "I'd": 'I would',
            "you'd've": 'you would have', "he'd've": 'he would have', "she'd've": 'she would have',
            "it'd've": 'it would have', "we'd've": 'we would have', "they'd've": 'they would have',
            "how's": 'how is', "how'd": 'how did', "how'll": 'how will'
        }

    def add_spaces_to_tags(self,text):
        pattern = re.compile(r'<.*?>')
        def add_space(match):
            return ' ' + match.group(0) + ' '
        return pattern.sub(add_space, text)
    
    def preprocess(self, text):
        text = text.lower()
        for pattern, repl in self.contractions.items():
            text = re.sub(pattern, repl, text)

        for key, pattern in self.regex_patterns.items():
            text = pattern.sub(key, text)

        text = self.add_spaces_to_tags(text)
        text = re.sub(r'\s+', ' ', text)

        return text
    
    def sentencize(self, processedText):
        pattern = r'([.!?;])'
        processedSentences = re.split(pattern, processedText)
        sentencesTemp =[]
        i = 0
        while(i<len(processedSentences)-1):
          sentencesTemp.append (processedSentences[i]+" "+processedSentences[i+1])
          i += 2
        processedSentences = sentencesTemp
        return processedSentences

    def tokenize(self, text, remove_stop_words=True):
        tokens = self.general_token_pattern.findall(text)
        if remove_stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

    def replace_last_word_with_punctuation(self,sentences):
        processed_sentences = []
        for sentence in sentences:
            modified_sentence = re.sub(r'[^ ]+$', '', sentence)
            processed_sentences.append(modified_sentence)
        return processed_sentences

##########################################################################################################
    
def generateNgrams(tokens, n):
    tokens = ['<START>'] * (n - 1) + tokens
    ngrams = []

    for i in range(n - 1, len(tokens)):
        context = []
        for p in reversed(range(n - 1)):
            context.append(tokens[i - p - 1])
        context = tuple(context)
        ngram = (context, tokens[i])
        ngrams.append(ngram)

    return ngrams



##########################################################################################################

def getNgramToCount(ngrams):
    ngramToCount = {}

    for ngram in ngrams:
        if ngram not in ngramToCount:
            ngramToCount[ngram] = 1
        else:
            ngramToCount[ngram] += 1

    return ngramToCount



##########################################################################################################


""" r to Nr"""
def getCountToCountOfCounts(wordCount):
    CountToCountOfCounts = {}
    for count in wordCount.values():
        if count not in CountToCountOfCounts:
            CountToCountOfCounts[count] = 1
        else:
            CountToCountOfCounts[count] += 1
    return CountToCountOfCounts


##########################################################################################################
def getBigramToTotalCount(threeGramToCount, countToNewCounts):
    bigramToTotalCount = {}

    # Iterate over the threeGramToCount dictionary
    for three_gram, count in threeGramToCount.items():
        # Extract the bigram (word1, word2)
        bigram = three_gram[0]

        # Get the new count value from countToNewCounts if it exists, else use the original count
        new_count = countToNewCounts.get(count, count)

        # Add the new count to the bigram's total count
        if bigram in bigramToTotalCount:
            bigramToTotalCount[bigram] += new_count
        else:
            bigramToTotalCount[bigram] = new_count

    return bigramToTotalCount


def getNgramToProb(threeGramToCount, countToNewCounts,bigramToTotalCount):
    threeGramToProb = {}
    for threeGram, Count in threeGramToCount.items():
        threeGramToProb[threeGram] = countToNewCounts[Count]/bigramToTotalCount[threeGram[0]]
    return threeGramToProb



##########################################################################################################

def getPerplexity(sentence, n, ngramToCount , ngramToProbs,unseenProb,showlog):

    tokenizer = Tokenizer()
    user_input = sentence

    user_input = tokenizer.preprocess(user_input)

    if user_input[-1] not in ".!?;":
        user_input += "." 
    
    
    processed_user_input_sentences = tokenizer.sentencize( user_input)


    processed_user_input_sentences = tokenizer.replace_last_word_with_punctuation(processed_user_input_sentences)

    tokenized_user_input_sentences = [tokenizer.tokenize(sentence,remove_stop_words=False) for sentence in processed_user_input_sentences]

    ngrams = []
    for i in range(0,len(tokenized_user_input_sentences)):
        ngrams += generateNgrams(tokenized_user_input_sentences[i], n)



    # print(ngrams)


    log_sum = 0
    for ngram in ngrams:
        if ngram in ngramToProbs:
            
            if showlog:
                print("Match :\t\t", end='')
                print(ngram, end='')
                print("\t", end='')
                print(ngramToCount[ngram], end='')
                print("\t", end='')
                print(ngramToProbs[ngram])


            probability = ngramToProbs[ngram]
        else:
            if  showlog:
                print("MISMATCH :\t\t", end='')
                print(ngram, end='')
                print("\t", end='')
                print(unseenProb)

            probability = unseenProb

        log_sum += math.log2(probability)

    if(len(ngrams) != 0):
      average_log_probability = -log_sum / len(ngrams)
    else:
      average_log_probability = 0

    perplexity = math.pow(2, average_log_probability)
    return perplexity

def getSentenceProbability(sentence, n, ngramToCount, ngramToProbs, unseenProb, showlog):
    tokenizer = Tokenizer()
    user_input = sentence

    user_input = tokenizer.preprocess(user_input)

    if user_input[-1] not in ".!?;":
        user_input += "."
    
    processed_user_input_sentences = tokenizer.sentencize(user_input)
    processed_user_input_sentences = tokenizer.replace_last_word_with_punctuation(processed_user_input_sentences)
    tokenized_user_input_sentences = [tokenizer.tokenize(sentence, remove_stop_words=False) for sentence in processed_user_input_sentences]
    ngrams = []
    for i in range(0, len(tokenized_user_input_sentences)):
        ngrams += generateNgrams(tokenized_user_input_sentences[i], n)

    log_prob_sum = 0  # Use log probabilities to avoid underflow
    for ngram in ngrams:
        if ngram in ngramToProbs:
            probability = ngramToProbs[ngram]
            if showlog:
                print(f"Match:\t\t{ngram}\t{ngramToCount[ngram]}\t{probability}")
        else:
            probability = unseenProb
            if showlog:
                print(f"MISMATCH:\t\t{ngram}\t{unseenProb}")
        # print("math.log2(probability)  :",math.log2(probability))
        log_prob_sum += math.log2(probability)

    # Convert log probability sum back to normal probability
    # print("log_prob_sum  :",log_prob_sum)
    probability_product = math.pow(2, log_prob_sum) if log_prob_sum != 0 else 0
    return probability_product


##########################################################################################################

class NgramInterpolation:
    def __init__(self):
        self.lambda1 = 0
        self.lambda2 = 0
        self.lambda3 = 0



    def estimate_lambdas(self, ngramToCount, nMinusOneGramToCount, nMinusTwoGramToCount):
        lambda1_count = 0
        lambda2_count = 0
        lambda3_count = 0

        N = sum(nMinusTwoGramToCount.values())

        print("N: ", N)
        for ((word1, word2), word3), count in ngramToCount.items():

            c123 = ngramToCount.get(((word1, word2), word3), 0)
            if(c123 == 0):
                return
            c12 = nMinusOneGramToCount.get(((word1,), word2), 0)
            c23 = nMinusOneGramToCount.get(((word2,), word3), 0)
            c2 = nMinusTwoGramToCount.get(((), word2), 0)
            c3 = nMinusTwoGramToCount.get(((), word3), 0)

            max_c = 0
            if c12 != 1:
                max_c = (c123 - 1) / (c12 - 1)

            max_c2 = 0
            if c2 != 1:
                max_c2 = (c23 - 1) / (c2 - 1)


            max_c3 = 0
            if N != 1:
                max_c3 = (c3 - 1) / (N - 1)


            maxim = max(max_c, max_c2, max_c3)

            if maxim == max_c:
                lambda3_count += c123

            elif maxim == max_c2:
                lambda2_count += c123

            elif maxim == max_c3:
                lambda1_count += c123


        lambda_sum = lambda1_count + lambda2_count + lambda3_count
        self.lambda1 = lambda1_count / lambda_sum if lambda_sum > 0 else 0
        self.lambda2 = lambda2_count / lambda_sum if lambda_sum > 0 else 0
        self.lambda3 = lambda3_count / lambda_sum if lambda_sum > 0 else 0


    def getNgramToProbs(self, ngramToCount, nMinusOneGramToCount, nMinusTwoGramToCount):
        ngramToProbs = {}
        N = sum(nMinusTwoGramToCount.values())

        print(len(ngramToCount))
        print(len(nMinusOneGramToCount))
        print(len(nMinusTwoGramToCount))
        for ((word1, word2), word3), count in ngramToCount.items():

            c123 = ngramToCount.get(((word1, word2), word3), 0)
            c12 = nMinusOneGramToCount.get(((word1,), word2), 0)
            c23 = nMinusOneGramToCount.get(((word2,), word3), 0)
            c2 = nMinusTwoGramToCount.get(((), word2), 0)
            c3 = nMinusTwoGramToCount.get(((), word3), 0)

            max_c = 0

            if c12 != 0:
                max_c = (c123) / (c12 )

            max_c2 = 0
            if c2 != 0:
                max_c2 = (c23 ) / (c2 )


            max_c3 = 0
            if N != 0:
                max_c3 = (c3) / (N)



            ngramToProbs[((word1, word2), word3)] = self.lambda1 * max_c3 + self.lambda2 * max_c2 + self.lambda3 * max_c

        return ngramToProbs




##########################################################################################################
    
def calculate_turing_estimates(rToNrMap):
    newRToNrMap = {}
    varianceOfNewRMap = {}

    for r, Nr in rToNrMap.items():
        NrPlusOne = rToNrMap.get(r + 1, 0)
        
        newRToNrMap[r] = r
        varianceOfNewRMap[r] = 0 

        if NrPlusOne > 0:
            numerator = (r + 1) * NrPlusOne
            denominator = Nr
            newRToNrMap[r] = numerator / denominator

            numeratorPart1 = ((r + 1) ** 2) * NrPlusOne
            denominatorPart1 = Nr ** 2 
            numeratorPart2 = Nr + NrPlusOne
            denominatorPart2 = Nr
            varianceOfNewRMap[r] = (numeratorPart1/ denominatorPart1 ) * (numeratorPart2 / denominatorPart2)
        
            

    return newRToNrMap, varianceOfNewRMap






def linear_good_turing_estimator(frequency_to_count_map):
    log_frequency_values = []
    log_smoothed_counts = []

    sorted_frequencies = sorted(frequency_to_count_map.keys())
    total_frequencies = len(sorted_frequencies)

    for i in range(total_frequencies):
        current_frequency = sorted_frequencies[i]
        
        previous_frequency = 0
        if i > 0:
            previous_frequency = sorted_frequencies[i - 1]  
        next_frequency = current_frequency
        if i < total_frequencies - 1:
            next_frequency = sorted_frequencies[i + 1] 
        
        if current_frequency != sorted_frequencies[-1]:
            gap = 0.5 * (next_frequency - previous_frequency)
        else:
            gap = current_frequency - previous_frequency
        
        if gap != 0:
            smoothed_count = frequency_to_count_map[current_frequency] / gap
        else:
            smoothed_count = 0
        
        log_frequency = np.log(current_frequency) if current_frequency > 0 else 0
        log_smoothed_count = np.log(smoothed_count) if smoothed_count > 0 else 0
        log_frequency_values.append(log_frequency)
        log_smoothed_counts.append(log_smoothed_count)

    slope, intercept, _, _, _ = linregress(log_frequency_values, log_smoothed_counts)
    
    frequency_star_LGT = {}

    for r in frequency_to_count_map:
        log_r_plus_1 = np.log(r + 1)
        exponent_with_r_plus_1 = intercept + slope * log_r_plus_1
        exponent_with_r = intercept + slope * np.log(r)
        exp_term_with_r_plus_1 = np.exp(exponent_with_r_plus_1)
        exp_term_with_r = np.exp(exponent_with_r)
        numerator = (r + 1) * exp_term_with_r_plus_1
        value_for_r = numerator / exp_term_with_r
        frequency_star_LGT[r] = value_for_r


    return frequency_star_LGT



def compute_frequency_distribution(event_frequencies):
    return {frequency: event_frequencies.count(frequency) for frequency in set(event_frequencies)}

def compute_ratio_of_singletons(frequency_distribution, total_observations):
    return frequency_distribution.get(1, 0) / total_observations if total_observations > 0 else 0

def compute_estimates(frequency_distribution):
    turing_estimates, var_turing_estimates = calculate_turing_estimates(frequency_distribution)
    lgt_estimates = linear_good_turing_estimator(frequency_distribution)
    return turing_estimates, var_turing_estimates, lgt_estimates

def choose_estimates(frequency_distribution, turing_estimates, var_turing_estimates, lgt_estimates):
    estimates = {}
    use_lgt = False
    for frequency in sorted(frequency_distribution):
        if use_lgt or frequency not in turing_estimates:
            estimates[frequency] = lgt_estimates[frequency]
        else:
            difference = abs(turing_estimates[frequency] - lgt_estimates[frequency])
            if difference <= 1.65 * np.sqrt(var_turing_estimates[frequency]):
                estimates[frequency] = turing_estimates[frequency]
            else:
                estimates[frequency] = lgt_estimates[frequency]
                use_lgt = True
    return estimates

def normalize_probabilities(estimates, total_events, singleton_ratio):
    unnormalized_probs = {freq: est / total_events for freq, est in estimates.items()}
    normalization_factor = sum(unnormalized_probs.values())
    normalized_probs = {freq: (1 - singleton_ratio) * (prob / normalization_factor) for freq, prob in unnormalized_probs.items()}
    normalized_probs[0] = singleton_ratio
    return normalized_probs

def hybrid_estimator(event_frequencies):
    frequency_distribution = compute_frequency_distribution(event_frequencies)
    total_observations = sum(frequency * count for frequency, count in frequency_distribution.items())
    singleton_ratio = compute_ratio_of_singletons(frequency_distribution, total_observations)

    turing_estimates, var_turing_estimates, lgt_estimates = compute_estimates(frequency_distribution)
    chosen_estimates = choose_estimates(frequency_distribution, turing_estimates, var_turing_estimates, lgt_estimates)

    # normalized_probabilities = normalize_probabilities(chosen_estimates, sum(event_frequencies), singleton_ratio)

    # print("Ratio of singletons to total observed items (N1/N):", singleton_ratio)
    return chosen_estimates,singleton_ratio





##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################


n=3

os.system('clear')

if len(sys.argv) != 3:
    print("Error : incorrect number of arguments")
    print("Usage: language_model.py <lm_type> <corpus_path>")
    print("***Exiting***")
    sys.exit(1)

choice = sys.argv[1]

input_file_path = sys.argv[2]





with open(input_file_path, "r") as file:
    text = file.read()
text = text.replace('\n', ' ')


tokenizer = Tokenizer()
user_input =text
user_input = tokenizer.preprocess(user_input)
if user_input[-1] not in ".!?;":
    user_input += "."  



def filter_short_sentences(sentences):
    return [sentence for sentence in sentences if len(sentence.split()) >1]

processed_user_input_sentences = tokenizer.sentencize( user_input)
processed_user_input_sentences = tokenizer.replace_last_word_with_punctuation(processed_user_input_sentences)
processed_user_input_sentences = filter_short_sentences(processed_user_input_sentences)









tokenized_user_input_sentences = [tokenizer.tokenize(sentence,remove_stop_words=False) for sentence in processed_user_input_sentences]


ngrams = []
nMinusOneGrams = []
nMinusTwoGrams = []
for i in range(0,len(tokenized_user_input_sentences)-1):
    ngrams += generateNgrams(tokenized_user_input_sentences[i], n)
    nMinusOneGrams += generateNgrams(tokenized_user_input_sentences[i], n-1)
    nMinusTwoGrams += generateNgrams(tokenized_user_input_sentences[i], n-2)

ngramToCount = getNgramToCount(ngrams)
nMinusOneGramToCount = getNgramToCount(nMinusOneGrams)
nMinusTwoGramToCount = getNgramToCount(nMinusTwoGrams)








 
ngramToProbs = {}

unseenProb = 0

if (choice ==  "g"):

    counts = list(ngramToCount.values())
    countToNewCounts,singletonRatio =  hybrid_estimator(counts)
    bigramToTotalCount = getBigramToTotalCount(ngramToCount, countToNewCounts)
    ngramToProbs = getNgramToProb(ngramToCount, countToNewCounts,bigramToTotalCount)
    # unseenProb = singletonRatio
    unseenProb = 10 ** -3
elif (choice ==  "i"):

    ngram_interpolation = NgramInterpolation()
    ngram_interpolation.estimate_lambdas(ngramToCount,nMinusOneGramToCount,nMinusTwoGramToCount)
    ngramToProbs = ngram_interpolation.getNgramToProbs(ngramToCount, nMinusOneGramToCount, nMinusTwoGramToCount)
    unseenProb = 10 ** -3





sentence = input("input sentence: ")


probability = getSentenceProbability(sentence, n, ngramToCount , ngramToProbs,unseenProb,0)
print("probability:\t\t", probability)
Perplexity = getPerplexity(sentence, n, ngramToCount , ngramToProbs,unseenProb,0)
print("Perplexity:\t\t", Perplexity)

