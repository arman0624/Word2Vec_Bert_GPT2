import copy
import re
from typing import Optional
import csv
from bs4 import BeautifulSoup
from transformers import BertTokenizer, GPT2Tokenizer
import torch
import pandas as pd
from typing import List
from collections import Counter
from torch.utils.data import Dataset, DataLoader


class TrainDataDataset(Dataset):
    def __init__(self, data, window_size=2, num_neg_samples=5, min_freq=3, neg_exponent=0.75):
        self.data = []
        self.tokens = []
        for sentence in data:
            tokens = sentence.split(" ")
            self.tokens.extend(tokens)
            self.data.append(tokens)
        print("Finished tokenizing...")
        self.word_count = Counter(self.tokens)
        self.window_size = window_size
        self.vocab = [k for k in self.word_count.keys() if self.word_count[k] > min_freq]
        self.vocab_size = len(self.vocab) + 1
        self.word_to_value = {word: ind for ind, word in enumerate(self.vocab)}
        self.value_to_word = {ind: word for ind, word in enumerate(self.vocab)}
        self.context_pairs = []

        self.word_to_value["<unk>"] = self.vocab_size
        self.value_to_word[self.vocab_size] = "<unk>"

        for sentence in self.data:
            for ind, word in enumerate(sentence):
                if word in self.vocab:
                    temp = sentence[max(0, ind-self.window_size):min(len(sentence), ind + 1 + self.window_size)]
                    self.context_pairs.extend([(word, c) for c in temp if c != word and c in self.vocab])
        print("Finished context pairs...")

    def __len__(self):
        return len(self.context_pairs)

    def __getitem__(self, ind):
        target, context = self.context_pairs[ind]

        target_value = self.word_to_value[target]
        context_value = self.word_to_value[context]
        return torch.tensor(target_value), torch.tensor(context_value)


def load_text_file(file_path: str) -> List[str]:
    """Reads a text file and returns a list of lines."""
    print(f"Loading text file from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        return [re.sub(r'[^\w\s]', '', re.sub(r'\d+', '', line)).strip().lower() for line in file]


def load_data_word2vec(window_size: int, num_neg_samples: int, min_freq: int, neg_exponent: float, use_additional_data: Optional[bool] = False):
    """
    Function for loading data for training or using vector representations
    (or embeddings) of words, for word2vec.

    Inputs:
        use_additional_data (bool): Whether to use additional data for training
    """
    # TODO

    train_data = load_text_file("data/training/training-data.1m")
    train_dataset = TrainDataDataset(train_data, window_size, num_neg_samples, min_freq, neg_exponent)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    cont_dev = pd.read_csv("data/contextual_similarity/contextual_dev_x.csv")
    cont_dev_word1 = cont_dev["word1"]
    cont_dev_word2 = cont_dev["word2"]
    cont_test = pd.read_csv("data/contextual_similarity/contextual_test_x.csv")
    cont_test_word1 = cont_test["word1"]
    cont_test_word2 = cont_test["word2"]
    isol_dev = pd.read_csv("data/isolated_similarity/isolated_dev_x.csv")
    isol_dev_word1 = isol_dev["word1"]
    isol_dev_word2 = isol_dev["word2"]
    isol_test = pd.read_csv("data/isolated_similarity/isolated_test_x.csv")
    isol_test_word1 = isol_test["word1"]
    isol_test_word2 = isol_test["word2"]
    cont_dev_data = torch.tensor([torch.tensor(train_dataset.word_to_value[word]) if word in train_dataset.word_to_value.keys() else train_dataset.vocab_size for word in cont_dev_word1]), torch.tensor([torch.tensor(train_dataset.word_to_value[word]) if word in train_dataset.word_to_value.keys() else train_dataset.vocab_size for word in cont_dev_word2])
    cont_test_data = torch.tensor([torch.tensor(train_dataset.word_to_value[word]) if word in train_dataset.word_to_value.keys() else train_dataset.vocab_size for word in cont_test_word1]), torch.tensor([torch.tensor(train_dataset.word_to_value[word]) if word in train_dataset.word_to_value.keys() else train_dataset.vocab_size for word in cont_test_word2])
    isol_dev_data = torch.tensor([torch.tensor(train_dataset.word_to_value[word]) if word in train_dataset.word_to_value.keys() else train_dataset.vocab_size for word in isol_dev_word1]), torch.tensor([torch.tensor(train_dataset.word_to_value[word]) if word in train_dataset.word_to_value.keys() else train_dataset.vocab_size for word in isol_dev_word2])
    isol_test_data = torch.tensor([torch.tensor(train_dataset.word_to_value[word]) if word in train_dataset.word_to_value.keys() else train_dataset.vocab_size for word in isol_test_word1]), torch.tensor([torch.tensor(train_dataset.word_to_value[word]) if word in train_dataset.word_to_value.keys() else train_dataset.vocab_size for word in isol_test_word2])

    return train_dataset, train_dataloader, cont_dev_data, cont_test_data, isol_dev_data, isol_test_data


def load_data_pretrained_models(model_type: str):
    """
    Function for loading and processing the evaluation datasets to be used
    by BERT or GPT-2.

    Inputs:
        model_type (str): One of BERT or GPT-2.
    """
    # TODO
    cont_dev_data = []
    cont_test_data = []
    isol_dev_data = []
    isol_test_data = []

    # Initialize tokenizer based on model type
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    else:
        raise ValueError("Invalid model type. Supported types are 'BERT' and 'GPT-2'.")
    print("set tokenizer...")

    # Load and process the contextual data
    def load_cont(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                word1_context = tokenizer.tokenize(row['word1_context'].lower())
                word2_context = tokenizer.tokenize(row['word2_context'].lower())
                context_soup = BeautifulSoup(row['context'].lower(), "html.parser").get_text()
                context = tokenizer.tokenize(context_soup)
                word1_span = find_sublist(context, word1_context, model_type)
                word2_span = find_sublist(context, word2_context, model_type)
                context_tokens = torch.tensor([tokenizer.encode(context)]) if model_type == 'bert' else torch.tensor(tokenizer.encode(context_soup))
                if file_path == 'data/contextual_similarity/contextual_dev_x.csv':
                    cont_dev_data.append((context_tokens, word1_span, word2_span, row['word1_context'].lower(), row['word2_context'].lower()))
                else:
                    cont_test_data.append((context_tokens, word1_span, word2_span, row['word1_context'].lower(), row['word2_context'].lower()))

    def load_isol(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                word1 = row['word1'].lower()
                word2 = row['word2'].lower()
                word1_tokens = torch.tensor([tokenizer.encode(tokenizer.tokenize(word1), add_special_tokens=False)]) if model_type == 'bert' else torch.tensor(tokenizer.encode(word1))
                word2_tokens = torch.tensor([tokenizer.encode(tokenizer.tokenize(word2), add_special_tokens=False)]) if model_type == 'bert' else torch.tensor(tokenizer.encode(word2))
                if file_path == "data/isolated_similarity/isolated_dev_x.csv":
                    isol_dev_data.append((word1_tokens, word2_tokens, word1, word2))
                else:
                    isol_test_data.append((word1_tokens, word2_tokens, word1, word2))

    load_cont("data/contextual_similarity/contextual_dev_x.csv")
    print("Finished loading cont_dev_data...")
    load_cont("data/contextual_similarity/contextual_test_x.csv")
    print("Finished loading cont_test_data...")
    load_isol("data/isolated_similarity/isolated_dev_x.csv")
    print("Finished loading isol_dev_data...")
    load_isol("data/isolated_similarity/isolated_test_x.csv")
    print("Finished loading isol_test_data...")

    return cont_dev_data, cont_test_data, isol_dev_data, isol_test_data


def find_sublist(main_list, sub_list, model_type):
    """
    Find the start index of sub_list in main_list.
    """
    if model_type == 'bert':
        word = ''.join(sub_list).replace('#', '')
        for i in range(len(main_list)):
            if main_list[i] == word:
                return (i, i+1)
            elif main_list[i] in word:
                start = i
                end = i+1
                while end < len(main_list) and ''.join(main_list[start:end+1]).replace('#', '') in word:
                    if ''.join(main_list[start:end+1]).replace('#', '') == word:
                        return (start, end+1)
                    end += 1
    else:
        word = ''.join(sub_list)
        for i in range(len(main_list)):
            if main_list[i] == word:
                return (i, i+1)
            elif main_list[i] in word:
                start = i
                end = i+1
                while end < len(main_list) and ''.join(main_list[start:end+1]) in word:
                    if ''.join(main_list[start:end+1]) == word:
                        return (start, end+1)
                    end += 1
            elif main_list[i].startswith('Ġ'):
                if main_list[i][1:] == word:
                    return (i, i+1)
                elif main_list[i][1:] in word:
                    start = i
                    end = i+1
                    while end < len(main_list) and ''.join(main_list[start:end+1])[1:] in word:
                        if ''.join(main_list[start:end+1])[1:] == word:
                            return (start, end+1)
                        end += 1
    print("failed for ", word)
    print("Tokens: ", main_list)
    print("Word tokens: ", sub_list)
    SystemExit
    return (-1, -1)
