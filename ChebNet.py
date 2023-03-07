from torch._C import device
import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F

torch.set_default_tensor_type(torch.cuda.DoubleTensor)

class ChebConv(nn.Module):
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c)).cuda()  # [K+1, 1, in_c, out_c]
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c).cuda())
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N]
        mul_L = self.cheb_polynomial(L).unsqueeze(1)   # [K, 1, N, N]

        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]
        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result
    # def forward(self, inputs, graph):
    #     b,N,N = graph.shape
    #     mul_L = torch.zeros((self.K,b,N,N))
    #     for i in range(b):
    #         L = ChebConv.get_laplacian(graph[i], self.normalize)
    #         mul_L[:,i] = self.cheb_polynomial(L)
    #     # L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N]
    #     # mul_L = self.cheb_polynomial(L).unsqueeze(1)   # [K, 1, N, N]

    #     result = torch.matmul(mul_L, inputs)  # [K, B, N, C]
    #     result = torch.matmul(result, self.weight)  # [K, B, N, D]
    #     result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

    #     return result

        

    def cheb_polynomial(self, laplacian):
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.double)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.double)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        if normalize:

            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L


class ChebNet(nn.Module):
    def __init__(self, in_c, hid_c, K):

        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c, K=K)
        self.act = nn.LeakyReLU()

    def forward(self, data, A):

        A = A.cuda()
        
        flow_x = data.cuda()  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]

        output_1 = self.act(self.conv1(flow_x, A))

        return output_1




