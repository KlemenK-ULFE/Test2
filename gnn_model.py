
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, depth=2, act=nn.ReLU):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth-1):
            layers += [nn.Linear(d, hidden), act()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class GraphConv(nn.Module):
    # Lightweight message passing layer with edge attributes.
    # h_i' = W_self h_i + AGG_j ( W_msg [h_j || e_ji] )
    def __init__(self, hdim, edim, hidden=128):
        super().__init__()
        self.lin_self = nn.Linear(hdim, hdim)
        self.lin_msg  = nn.Linear(hdim + edim, hdim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        h_src = x[src]
        m_in = torch.cat([h_src, edge_attr], dim=1)
        m = self.lin_msg(m_in)
        N = x.size(0)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)
        h = self.lin_self(x) + agg
        return self.act(h)

class LVGNN(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, hidden=128, layers=3, out_dim=3):
        super().__init__()
        self.embed = MLP(in_node_dim, hidden, hidden=hidden, depth=2)
        self.convs = nn.ModuleList([GraphConv(hidden, in_edge_dim, hidden=hidden) for _ in range(layers)])
        self.head  = MLP(hidden, out_dim, hidden=hidden, depth=2)

    def forward(self, x, edge_index, edge_attr):
        h = self.embed(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        out = self.head(h)
        return out

def physics_voltage_drop_loss(V_pred, edge_index, edge_attr, P_node, parents):
    # Optional physics loss: encourage V_p - V_c ≈ k * R * P_downstream (per phase).
    device = V_pred.device
    src, dst = edge_index
    mask = (src < dst)
    src = src[mask]; dst = dst[mask]
    R = edge_attr[mask][:,0]
    N = P_node.size(0)
    par = parents.clone().to(device)
    children = [[] for _ in range(N)]
    for v in range(1, N):
        p = int(par[v].item())
        children[p].append(v)
    P_down = torch.zeros_like(P_node)
    for v in reversed(range(N)):
        tot = P_node[v]
        for c in children[v]:
            tot = tot + P_down[c]
        P_down[v] = tot
    k = 0.006
    Vp = V_pred[src]
    Vc = V_pred[dst]
    Pd = P_down[dst]
    drop_pred = Vp - Vc
    drop_phys = k * R.view(-1,1) * Pd
    return torch.mean((drop_pred - drop_phys)**2)
