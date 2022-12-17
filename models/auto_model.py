import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertPredictionHeadTransform

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_hidden_states = torch.stack(all_hidden_states)
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average[:, 0]

class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm, dropout_rate, is_lstm=True):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm

        if is_lstm:
            self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True)
        else:
            self.lstm = nn.GRU(self.hidden_size, self.hiddendim_lstm, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, all_hidden_states):
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out

class PredictionHead(nn.Module):
    '''
    A prediction head for a single objective of the SpeechGraderModel.

    Args:
        config (AutoConfig): the config for the the pre-trained BERT model
        num_labels (int): the number of labels that can be predicted

    Attributes:
        transform (transformers.modeling_bert.BertPredictionHeadTransform): a dense linear layer with gelu activation
            function
        decoder (torch.nn.Linear): a linear layer that makes predictions across the labels
        bias (torch.nn.Parameter): biases per label
    '''
    def __init__(self, config, num_labels):
        super(PredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, num_labels)
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class SpeechGraderModel(BertPreTrainedModel):
    '''
    BERT Model for automated speech scoring of transcripts.

    Attributes:
        model_dir (string) : directory to store/load this model to/from
        args : arguments from argparse
        bert_tokenizer : the tokenizer to use for the BERT model
        training_objectives_decoders (dict of str: (int, float)): a mapping of training objectives to their training
            parameters (a tuple containing the number of labels (i.e. the decoder output size) and the weight to give to
            this objective in the combined weighted loss function)
    '''
    def __init__(self, config):
        super(SpeechGraderModel, self).__init__(config)

        self.config = config
        self.bert = AutoModel.from_config(config)

        # Creates a prediction head per objective.
        self.decoder_objectives = config.training_objectives.keys()
        for objective, objective_params in config.training_objectives.items():
            num_predictions, _ = objective_params
            decoder = PredictionHead(self.config, num_predictions)
            setattr(self, objective + '_decoder', decoder)

        # The score scaler is used to force the result of the score prediction head to be within the range of possible
        # scores.
        self.score_scaler = nn.Hardtanh(min_val=0, max_val=config.max_score)
        self.init_weights()

    def forward(self, batch):
        """
        Returns:
        training_objective_predictions (dict of str: [float]): mapping of training objective to the predicted label
        """
        bert_sequence_output, bert_pooled_output = self.bert(**batch, return_dict=False)
        training_objective_predictions = {}

        for objective in self.decoder_objectives:
            scoring_input = bert_pooled_output if objective == 'score' else bert_sequence_output
            decoded_objective = getattr(self, objective + '_decoder')(scoring_input)
            decoded_objective = self.score_scaler(decoded_objective) if objective == 'score' else decoded_objective
            training_objective_predictions[objective] = decoded_objective.view(-1, decoded_objective.shape[2]) \
                if objective != 'score' else decoded_objective.squeeze()

        return training_objective_predictions

class PredictionPoolHead(nn.Module):
    '''
    A prediction head for a single objective of the SpeechGraderModel.

    Args:
        config (AutoConfig): the config for the the pre-trained BERT model
        num_labels (int): the number of labels that can be predicted

    Attributes:
        transform (transformers.modeling_bert.BertPredictionHeadTransform): a dense linear layer with gelu activation
            function
        decoder (torch.nn.Linear): a linear layer that makes predictions across the labels
        bias (torch.nn.Parameter): biases per label
    '''
    def __init__(self, config, num_labels, mode="mean"):
        super(PredictionPoolHead, self).__init__()
        self.config = config
        self.mode = mode
        if self.mode == "mean":
            self.pooler = MeanPooling()
        elif self.mode == "weighted":
            self.pooler = WeightedLayerPooling(config.num_hidden_layers)
        else:
            self.pooler = LSTMPooling(config.num_hidden_layers,
                 config.hidden_size, config.hidden_size, 0.1, is_lstm=True
            )
        self.fc = nn.Linear(self.config.hidden_size, num_labels)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, outputs, inputs):

        last_hidden_states = outputs['last_hidden_state']
        all_hidden_states = outputs['hidden_states']

        if self.mode == 'mean':
            feature = self.pooler(last_hidden_states, inputs['attention_mask'])
        elif self.mode == 'weighted':
            feature = self.pooler(all_hidden_states)
        elif self.mode in ['gru', 'lstm']:
            feature = self.pooler(all_hidden_states)
        else:
            raise ValueError('Unknown pooling type')

        outputs = self.fc(feature)

        return outputs

class SpeechGraderPoolModel(nn.Module):
    def __init__(self, config, num_classes=1, mode="mean"):
        super(SpeechGraderPoolModel,self).__init__()
        #self.model_name = config['model']
        #self.freeze = config['freeze_encoder']
        config.hidden_dropout = 0.
        config.hidden_dropout_prob = 0.
        config.attention_dropout = 0.
        config.attention_probs_dropout_prob = 0.
        #config.output_hidden_states = True
        #self.encoder = AutoModel.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_config(config)
 
        #if self.freeze:
        #    for param in self.encoder.base_model.parameters():
        #        param.requires_grad = False

        # Creates a pooling prediction head per objective.
        self.decoder_objectives = config.training_objectives.keys()
        for objective, objective_params in config.training_objectives.items():
            num_predictions, _ = objective_params
            decoder = PredictionPoolHead(self.encoder.config, num_predictions, mode)
            setattr(self, objective + '_decoder', decoder)

        # The score scaler is used to force the result of the score prediction head to be within the range of possible
        # scores.
        #self.score_scaler = nn.Hardtanh(min_val=0, max_val=config.max_score)

    def forward(self, inputs):
        outputs = self.encoder(**inputs, return_dict=True)

        training_objective_predictions = {}
        for objective in self.decoder_objectives:
            decoded_objective = getattr(self, objective + '_decoder')(outputs, inputs)
            #decoded_objective = self.score_scaler(decoded_objective) if objective == 'score' else decoded_objective
            training_objective_predictions[objective] = decoded_objective.view(-1, decoded_objective.shape[2]) \
                if objective != 'score' else decoded_objective.squeeze()

        return training_objective_predictions
