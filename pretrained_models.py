import argparse
from typing import Optional, Any
import torch
import csv
import torch.nn as nn
from transformers import GPT2Model, BertModel
import torch.nn.functional as F
from load_data import load_data_pretrained_models
from utils import get_similarity_scores, compute_spearman_correlation

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PretrainedEmbeddingModel(nn.Module):
    """
    Pretrained model to extract embeddings from.
    """
    def __init__(self, model_type: str, layers: str, merge_strategy: str, layer_merging: str):
        """
        Inputs:
            model (str): Which model to load (GPT-2 or BERT)
            layers (str): The hidden states of which layers should we use?
            merge_stragegy (str): How do we merge subwords when a word is split into multiple subwords?
            layer_strategy (str): If we use multiple layers, how do we combine them?
        """
        # TODO
        super(PretrainedEmbeddingModel, self).__init__()
        self.layers = int(layers)
        if model_type == 'gpt2':
            self.model_type = model_type
            self.model = GPT2Model.from_pretrained(self.model_type)
        else:
            self.model_type = 'bert'
            self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.config.layer_strategy = layer_merging
        self.model.config.merge_strategy = merge_strategy
        self.model.config.output_hidden_states = True

    def forward(self, input_ids):
        """
        Performs a forward pass through the model

        Inputs:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Mask indicating which elements in the input sequence are padding tokens.
                Shape: (batch_size, sequence_length).

        Returns:
            torch.Tensor: The embeddings extracted from the model. The shape depends on the chosen model and
                layer selection strategy.
        """
        with torch.no_grad():
            return self.model(input_ids)

    def extract_embedding_from_outputs(self, model_type, data_type, model_output, word_span):
        """
        Extracts the embedding corresponding to the input word span according to whatever
        strategy the model has been initialized with.

        Input:
            model_output (Any): The output of the model following a forward pass.
            word_span (torch.LongTensor): A minibatch of word spans, where each span contains
                                          the start and end position of the word in the tokenized model input.
        """
        if data_type == "contextual":
            if model_type == 'bert':
                start, end = word_span
                if start == -1 or end == -1:
                    print("FAILED-------------------------------------------------------------")
                if self.model.config.merge_strategy == 'avg' and self.model.config.layer_strategy == 'avg':
                    return torch.mean(torch.mean(torch.stack(model_output.hidden_states), dim=0)[:, start:end + 1, :].squeeze(dim=0), dim=0)
                elif self.model.config.merge_strategy == 'sum' and self.model.config.layer_strategy == 'avg':
                    return torch.sum(torch.mean(torch.stack(model_output.hidden_states), dim=0)[:, start:end + 1, :].squeeze(dim=0), dim=0)

                elif self.model.config.merge_strategy == 'avg' and self.model.config.layer_strategy == 'last':
                    return torch.mean(model_output.last_hidden_state[:, start:end + 1, :].squeeze(dim=0), dim=0)
                elif self.model.config.merge_strategy == 'sum' and self.model.config.layer_strategy == 'last':
                    return torch.sum(model_output.last_hidden_state[:, start:end + 1, :].squeeze(dim=0), dim=0)

                elif self.model.config.merge_strategy == 'avg' and self.model.config.layer_strategy == 'sum':
                    return torch.mean(torch.sum(torch.stack(model_output.hidden_states), dim=0)[:, start:end + 1, :].squeeze(dim=0), dim=0)
                elif self.model.config.merge_strategy == 'sum' and self.model.config.layer_strategy == 'sum':
                    return torch.sum(torch.sum(torch.stack(model_output.hidden_states), dim=0)[:, start:end + 1, :].squeeze(dim=0), dim=0)
                else:
                    return torch.mean(torch.mean(torch.stack(model_output.hidden_states), dim=0)[:, start:end + 1, :].squeeze(dim=0), dim=0)
            else:
                start, end = word_span
                if start == -1 or end == -1:
                    print("FAILED-------------------------------------------------------------")
                if self.model.config.merge_strategy == 'avg':
                    return torch.mean(model_output.last_hidden_state[start:end, :], dim=0)
                elif self.model.config.merge_strategy == 'sum':
                    return torch.sum(model_output.last_hidden_state[start:end, :], dim=0)
                else:
                    return torch.mean(model_output.last_hidden_state[start:end, :], dim=0)

        elif data_type == "isolated":
            if model_type == 'bert':
                if self.model.config.merge_strategy == 'avg' and self.model.config.layer_strategy == 'avg':
                    return torch.mean(torch.mean(torch.stack(model_output.hidden_states), dim=0).squeeze(dim=0), dim=0)
                elif self.model.config.merge_strategy == 'sum' and self.model.config.layer_strategy == 'avg':
                    return torch.sum(torch.mean(torch.stack(model_output.hidden_states), dim=0).squeeze(dim=0), dim=0)

                elif self.model.config.merge_strategy == 'avg' and self.model.config.layer_strategy == 'last':
                    return torch.mean(model_output.last_hidden_state.squeeze(dim=0), dim=0)
                elif self.model.config.merge_strategy == 'sum' and self.model.config.layer_strategy == 'last':
                    return torch.sum(model_output.last_hidden_state.squeeze(dim=0), dim=0)

                elif self.model.config.merge_strategy == 'avg' and self.model.config.layer_strategy == 'sum':
                    return torch.mean(torch.sum(torch.stack(model_output.hidden_states), dim=0).squeeze(dim=0), dim=0)
                elif self.model.config.merge_strategy == 'sum' and self.model.config.layer_strategy == 'sum':
                    return torch.sum(torch.sum(torch.stack(model_output.hidden_states), dim=0).squeeze(dim=0), dim=0)

                else:
                    return torch.mean(torch.mean(torch.stack(model_output.hidden_states), dim=0).squeeze(dim=0), dim=0)
            else:
                if self.model.config.merge_strategy == 'avg':
                    return torch.mean(model_output.last_hidden_state, dim=0)
                elif self.model.config.merge_strategy == 'sum':
                    return torch.sum(model_output.last_hidden_state, dim=0)
                else:
                    return torch.mean(model_output.last_hidden_state, dim=0)

    def extract_isolated(self, isolated_data: Any):
        """
        Extracts word embeddings for isolated word pairs

        Inputs:
            isolated_data (Any): A dataset containing processed data for isolated word pairs. Recommended
                                 to be in a Dataloader format.

        Returns:
            word1_embeds (List): A list of embeddings for the first words of the word pairs in the dataset, 
                                 in the order they appear.
            word2_embeds (List): A list of embeddings for the second words of the word pairs in the dataset, 
                                 in the order they appear.
        """
        # TODO
        word1_embeds = []
        word2_embeds = []
        for word1, word2, w1, w2 in isolated_data:
            temp = self.extract_embedding_from_outputs(self.model_type, "isolated", self.forward(word1), None)
            word1_embeds.append(temp)
            word2_embed = self.extract_embedding_from_outputs(self.model_type, "isolated", self.forward(word2), None)
            word2_embeds.append(word2_embed.permute(*torch.arange(word2_embed.ndim - 1, -1, -1)))
            temp1 = word1_embeds[-1]
            temp2 = word2_embeds[-1]
            if len(temp1) != len(temp2):
                print("specific embeds for " + w1 + " and " + w2 + " are different lengths. " + w1 + " is length " + str(len(temp1)) + " and " + w2 + " is length " + str(len(temp2)))
                print(temp1)
                print(temp2)
        return word1_embeds, word2_embeds

    def extract_contextual(self, contextual_data: Any):
        """
        Extracts word embeddings for contextual word pairs

        Inputs:
            contextual_data (Any): A dataset containing processed data for contextual word pairs. Recommended
                                 to be in a Dataloader format.

        Returns:
            word1_embeds (List): A list of embeddings for the first words of the word pairs in the dataset,
                                 in the order they appear.
            word2_embeds (List): A list of embeddings for the second words of the word pairs in the dataset,
                                 in the order they appear.
        """
        # TODO
        word1_embeds = []
        word2_embeds = []
        for data, span1, span2, w1, w2 in contextual_data:
            output = self.forward(data)
            temp = self.extract_embedding_from_outputs(self.model_type, "contextual", output, span1)
            word1_embeds.append(temp)
            word2_embed = self.extract_embedding_from_outputs(self.model_type, "contextual", output, span2)
            word2_embeds.append(word2_embed.permute(*torch.arange(word2_embed.ndim - 1, -1, -1)))
            temp1 = word1_embeds[-1]
            temp2 = word2_embeds[-1]
            if len(temp1) != len(temp2):
                print("specific embeds for " + w1 + " and " + w2 + " are different lengths. " + w1 + " is length " + str(len(temp1)) + " and " + w2 + " is length " + str(len(temp2)))
                print(temp1)
                print(temp2)
        return word1_embeds, word2_embeds


def get_args():
    """
    You may freely add new command line arguments to this function, or change them.
    """
    parser = argparse.ArgumentParser(description='word2vec model')
    parser.add_argument('-m', '--model_type', type=str, choices=['gpt2', 'bert'],
                        help='Which pretrained model will we use?')

    parser.add_argument('-l', '--layers', type=str, default='12',
                        help="The hidden dimension outputs of which layers will we use?")
    parser.add_argument('-sm', '--subword_merging', type=str, default=None, choices=[None, 'avg', 'sum'],
                        help="How do we merge subwords?")
    parser.add_argument('-lm', '--layer_merging', type=str, default=None, choices=[None, 'avg', 'sum', 'last'],
                        help="How do we merge layers, if we do this at all?")

    parser.add_argument('-e', '--experiment_name', type=str, default='testing',
                        help="What should we name our experiment?")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model_type = args.model_type
    layers = args.layers
    merge_strategy = args.subword_merging
    layer_merging = args.layer_merging
    experiment_name = args.experiment_name

    # Load data
    cont_dev_data, cont_test_data, isol_dev_data, isol_test_data = load_data_pretrained_models(model_type)

    # Load model
    model = PretrainedEmbeddingModel(model_type, layers, merge_strategy, layer_merging)
    model.to(DEVICE)

    isol_dev_embeds_word1, isol_dev_embeds_word2 = model.extract_isolated(isol_dev_data)
    print("Finished extracting isolated for isol_dev_data...")
    isol_test_embeds_word1, isol_test_embeds_word2 = model.extract_isolated(isol_test_data)
    print("Finished extracting isolated for isol_test_data...")
    cont_dev_embeds_word1, cont_dev_embeds_word2 = model.extract_contextual(cont_dev_data)
    print("Finished extracting contextual for cont_dev_data...")
    cont_test_embeds_word1, cont_test_embeds_word2 = model.extract_contextual(cont_test_data)
    print("Finished extracting contextual for cont_test_data...")

    # Save the embeddings to text file
    isol_test_w1s = [word for _, _, word, _ in isol_test_data]
    isol_test_w2s = [word for _, _, _, word in isol_test_data]
    cont_test_w1s = [word for _, _, _, word, _ in cont_test_data]
    cont_test_w2s = [word for _, _, _, _, word in cont_test_data]

    def save_embeddings(file_path, words, word_embeddings):
        with open(file_path, "w") as file:
            for ind, word in enumerate(words):
                embedding = ' '.join(map(str, word_embeddings[ind].tolist()))
                file.write(f"{word} {embedding}\n")

    def touch_up(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        content = content.replace(',', '').replace('[', '').replace(']', '')
        with open(file_path, 'w') as file:
            file.write(content)

    if model_type == 'gpt2':
        save_embeddings("./results/gpt2_cont_test_words1_embeddings.txt", cont_test_w1s, cont_test_embeds_word1)
        save_embeddings("./results/gpt2_cont_test_words2_embeddings.txt", cont_test_w2s, cont_test_embeds_word2)
        save_embeddings("./results/gpt2_isol_test_words1_embeddings.txt", isol_test_w1s, isol_test_embeds_word1)
        save_embeddings("./results/gpt2_isol_test_words2_embeddings.txt", isol_test_w2s, isol_test_embeds_word2)
        touch_up("./results/gpt2_isol_test_words1_embeddings.txt")
        touch_up("./results/gpt2_isol_test_words2_embeddings.txt")
    else:
        save_embeddings("./results/bert_cont_test_words1_embeddings.txt", cont_test_w1s, cont_test_embeds_word1)
        save_embeddings("./results/bert_cont_test_words2_embeddings.txt", cont_test_w2s, cont_test_embeds_word2)
        save_embeddings("./results/bert_isol_test_words1_embeddings.txt", isol_test_w1s, isol_test_embeds_word1)
        save_embeddings("./results/bert_isol_test_words2_embeddings.txt", isol_test_w2s, isol_test_embeds_word2)
        touch_up("./results/bert_isol_test_words1_embeddings.txt")
        touch_up("./results/bert_isol_test_words2_embeddings.txt")

    # Compute word pair similarity scores using your embedding
    isol_dev_sim_scores = get_similarity_scores(isol_dev_embeds_word1, isol_dev_embeds_word2)
    print("Finished getting similarity scores isol_dev_data...")
    cont_dev_sim_scores = get_similarity_scores(cont_dev_embeds_word1, cont_dev_embeds_word2)
    print("Finished getting similarity scores cont_dev_data...")

    def load_y_csv_file(file_path: str):
        labels = []
        print(f"Loading CSV file from {file_path}...")
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'id':
                    continue
                labels.append(float(row[1]))
        return labels

    isol_dev_labels = load_y_csv_file("data/isolated_similarity/isolated_dev_y.csv")
    cont_dev_labels = load_y_csv_file("data/contextual_similarity/contextual_dev_y.csv")

    # Evaluate your similarity scores against human ratings
    isol_dev_corr = compute_spearman_correlation(isol_dev_sim_scores, isol_dev_labels)
    print("Finished computing spearman correlation for isol_dev_data")

    cont_dev_corr = compute_spearman_correlation(cont_dev_sim_scores, cont_dev_labels)
    print("Finished computing spearman correlation for cont_dev_data")

    print("\n\n\nEvaluating on: isolated word pairs")
    print("Correlation score on dev set:", isol_dev_corr)

    print("\n\n\nEvaluating on: contextual word pairs")
    print("Correlation score on dev set:", cont_dev_corr)


if __name__ == "__main__":
    main()
