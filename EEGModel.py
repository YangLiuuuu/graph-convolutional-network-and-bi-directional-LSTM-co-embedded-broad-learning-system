import torch
import torch.nn as nn
from ChebNet import ChebNet


torch.set_default_tensor_type(torch.cuda.DoubleTensor)

class Attention(nn.Module):
    """Attention mechanism.
        Parameters
        ----------
        dim : int
            The input and out dimension of per token features.
        n_heads : int
            Number of attention heads.
        qkv_bias : bool
            If True then we include bias to the query, key and value projections.
        attn_p : float
            Dropout probability applied to the query, key and value tensors.
        proj_p : float
            Dropout probability applied to the output tensor.
        Attributes
        ----------
        scale : float
            Normalizing consant for the dot product.
        qkv : nn.Linear
            Linear projection for the query, key and value.
        proj : nn.Linear
            Linear mapping that takes in the concatenated output of all attention
            heads and maps it into a new space.
        attn_drop, proj_drop : nn.Dropout
            Dropout layers.
        """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0, proj_p=0):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass.
               Parameters
               ----------
               x : torch.Tensor
                   Shape `(n_samples, n_patches + 1, dim)`.  class token is added in the first dimension
               Returns
               -------
               torch.Tensor
                   Shape `(n_samples, n_patches + 1, dim)`.
               """
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches+1, 3*dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_samples, n_patches+1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches+1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1) # (n_samples, n_heads, head_dim, n_patches+1)
        dp = (
            q @ k_t
        ) * self.scale  # (n_samples, n_heads, n_patches+1, n_patches+1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches+1, n_patches+1)
        attn = self.attn_drop(attn)
        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches+1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches+1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches+1, dim)
        x = self.proj(weighted_avg)  # (n_samples, n_patches+1, dim)
        x = self.proj_drop(x)
        return x

# LSTM + Graph
class EEGModel(nn.Module):
    def __init__(self, in_chan,class_num=4 ,conv_out_chan=32, graph_hidden_dim=128,K = 4, num_layers=2, adj = None, graph_pooling_type="sum"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_chan, out_channels = 32, kernel_size=5, stride=1, padding=2),  # 256*32
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=32, out_channels = 32, kernel_size=3, stride=1, padding=1), # 256*32
            nn.BatchNorm1d(32),

            nn.Dropout(0.3),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1), # 128*32
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
        )

        self.lstm1 = nn.LSTM(input_size = 32, hidden_size = 64, num_layers = 3, batch_first=True, bidirectional=True, )
        # self.att1 = Attention(128,n_heads=4,attn_p=0.2)
        self.act = nn.LeakyReLU()
        self.lstm2 = nn.LSTM(128, 32, 3, batch_first=True, bidirectional=True)
        # self.att2 = Attention(64, n_heads=4,attn_p=0.2)
        self.lstm3 = nn.LSTM(64, 32, 3, batch_first=True, bidirectional=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(128*32, 128)
        self.flatten = nn.Flatten()
        self.GCNs = torch.nn.ModuleList()
        self.GCNs.append(ChebNet(conv_out_chan, graph_hidden_dim, K))
        for i in range(num_layers-1):
            self.GCNs.append(ChebNet(graph_hidden_dim, graph_hidden_dim, K))
        self.adj = adj
        self.graph_pooling_type=graph_pooling_type
        self.classifier = nn.Linear(384,class_num)
    def forward(self, x):
        
        x = x.permute(0,2,1)
        conv_out = self.conv(x)
        lstm_out = conv_out.permute(0,2,1)
        lstm_out,_ = self.lstm1(lstm_out)
        lstm_out = self.act(lstm_out)
        # lstm_out = self.att1(lstm_out)
        lstm_out = self.bn1(lstm_out.permute(0,2,1)).permute(0,2,1)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.act(lstm_out)
        # lstm_out = self.att2(lstm_out)
        lstm_out = self.bn2(lstm_out.permute(0,2,1)).permute(0,2,1)
        lstm_out,_ = self.lstm3(lstm_out)
        lstm_hidden = self.flatten(lstm_out)
        lstm_hidden = self.fc(lstm_hidden)
        
        graph_hidden = conv_out.permute(0,2,1)
        for layer in self.GCNs:
            graph_hidden = self.act(layer(graph_hidden,self.adj))
        
        if(self.graph_pooling_type == 'mean'):
            pooled = torch.mean(graph_hidden,dim=1)
        if (self.graph_pooling_type == 'max'):
            pooled = torch.max(graph_hidden,dim=1)[0]
        if (self.graph_pooling_type == 'sum'):
            pooled = torch.sum(graph_hidden,dim=1)
        
        hidden = torch.cat((lstm_hidden,pooled),dim=1)
        # print(hidden.shape)
        return hidden,self.classifier(hidden)