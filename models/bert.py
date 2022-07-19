import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Config(object):

    def __init__(self, dataset):
        self.model_name = 'bert'
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
        self.bert_path = './bert_finetuned'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        output = self.bert(context, attention_mask=mask)
        out1 = self.fc(output[1])
        return out1

