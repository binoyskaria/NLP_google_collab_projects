import re
import numpy as np
import os

os.system('clear')


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



tokenizer = Tokenizer()
user_input =input("your text:")

if user_input[-1] not in ".!?;":
    user_input += "."                                       
user_input = tokenizer.preprocess(user_input)
processed_user_input_sentences = tokenizer.sentencize( user_input)
processed_user_input_sentences = tokenizer.replace_last_word_with_punctuation(processed_user_input_sentences)
tokenized_user_input_sentences = [tokenizer.tokenize(sentence,remove_stop_words=False) for sentence in processed_user_input_sentences]

print("Tokenized Text: ")
for i in range(0,len(tokenized_user_input_sentences)):
    print(tokenized_user_input_sentences[i])



