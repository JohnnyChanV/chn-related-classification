import torch
import torch.nn as nn
from transformers import AutoModel
import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, embed_size, max_len, num_filters=64, filter_sizes=[2,4,4,2]):
        super(TextCNN, self).__init__()
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.hidden_size = num_filters * len(filter_sizes)

        # 卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embed_size, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        # 线性层
        self.fc = nn.Sequential(
            nn.Linear(num_filters * len(filter_sizes), max_len),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: (batch, len, embed_size)
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 卷积层处理
        x = x.permute(0, 2, 1)  # shape: (batch, embed_size, len)
        conv_outputs = []
        for conv in self.conv_layers:
            conv_output = conv(x)  # shape: (batch, num_filters, len - filter_size + 1)
            conv_output = nn.functional.relu(conv_output)
            conv_output, _ = torch.max(conv_output, dim=-1)  # shape: (batch, num_filters)
            conv_outputs.append(conv_output)

        # 拼接卷积层输出
        cnn_output = torch.cat(conv_outputs, dim=-1)  # shape: (batch, num_filters * len(filter_sizes))

        # 线性层处理
        fc_output = self.fc(cnn_output)  # shape: (batch, hidden_size)
        # fc_output = fc_output.unsqueeze(-1)
        return fc_output


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        if not args.use_llm:
            self.embedding = nn.Embedding(250002,args.embed_dim)
            self.hidden = args.embed_dim
        else:
            self.LLM = AutoModel.from_pretrained(args.LLM)
            self.hidden = self.LLM.config.hidden_size
        self.cnn = TextCNN(self.hidden,args.max_length)
        self.fc = nn.Linear(args.max_length,args.class_num)

    def forward(self, x,mask):
        if self.args.use_llm:
            x = self.LLM(input_ids=x,attention_mask=mask).last_hidden_state
        else:
            x = self.embedding(x)
        x = self.cnn(x)
        x = self.fc(x)
        return x
