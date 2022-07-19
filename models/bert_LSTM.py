import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'bert_LSTM'
        self.train_path = dataset + '/data1/train.txt'
        self.dev_path = dataset + '/data1/dev.txt'
        self.test_path = dataset + '/data1/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data1/class.txt').readlines()]
        self.save_path = dataset + '/saved_model/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 5
        self.batch_size = 6
        self.pad_size = 35
        self.learning_rate = 5e-5
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')
        self.hidden_size = 768
        self.dropout = 0.1
        self.rnn_hidden = 256
        self.num_layers = 1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('asafaya/bert-base-arabic')
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=False, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        output = self.bert(context, attention_mask=mask)
        out, _ = self.lstm(output[0])
        out = self.dropout(out)
        out = self.fc_rnn(out[:, -1, :])
        return out
