# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init


class ChebConv(nn.Module):

    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize 

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c)) 
 
        init.xavier_normal_(self.weight) 

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))  
            init.zeros_(self.bias)  
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):

        L = ChebConv.get_laplacian(graph, self.normalize)  
        mul_L = self.cheb_polynomial(L).unsqueeze(1)   

        result = torch.matmul(mul_L, inputs)  
        result = torch.matmul(result, self.weight) 
        result = torch.sum(result, dim=0) + self.bias  

        return result

    def cheb_polynomial(self, laplacian): 
   
        N = laplacian.size(0)  
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float) 

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
    def __init__(self, in_c, hid_c, out_c, K):
 
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c, K=K) 
        self.conv2 = ChebConv(in_c=hid_c, out_c=out_c, K=K) 
        self.act = nn.ReLU()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  
        flow_x = data["flow_x"].to(device)  

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  

        output_1 = self.act(self.conv1(flow_x, graph_data))
        output_2 = self.act(self.conv2(output_1, graph_data))

        return output_2.unsqueeze(2)  
