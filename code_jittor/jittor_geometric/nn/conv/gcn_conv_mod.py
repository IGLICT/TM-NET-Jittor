from typing import Optional, Tuple
from jittor_geometric.typing import Adj, OptVar

import jittor as jt
from jittor import Var
from jittor_geometric.nn.conv import MessagePassing
from jittor_geometric.utils import add_remaining_self_loops
from jittor_geometric.utils.num_nodes import maybe_num_nodes

from ..inits import glorot, zeros


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, Var):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = jt.ones((edge_index.size(1), ))

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        shape = list(edge_weight.shape)
        shape[0] = num_nodes
        deg = jt.zeros(shape)
        deg = jt.scatter(deg, 0, col, src=edge_weight, reduce='add')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConvMod(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    """

    _cached_edge_index: Optional[Tuple[Var, Var]]

    def __init__(self, in_channels: int, out_channels: int, edge_index, 
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConvMod, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = jt.random((in_channels, out_channels))

        if bias:
            self.bias = jt.random((out_channels,))

        else:
            self.bias = None

        self.reset_parameters()

        edge_weight = None
        if self.normalize:
            if isinstance(edge_index, Var):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(edge_index, 
                                                    edge_weight, 
                                                    int(jt.max(edge_index).data)+1, 
                                                    self.improved, 
                                                    self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def execute(self, x: Var) -> Var:
        """"""

        x = x @ self.weight
        x = x.transpose(1, 0, 2)
        out = self.propagate(jt.int32(self.edge_index), 
                            x=x, 
                            edge_weight=self.edge_weight)
        out = out.transpose(1, 0, 2)
        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Var, edge_weight: OptVar) -> Var:
        # print('{} {}'.format(x_j.shape, edge_weight.shape))
        return x_j if edge_weight is None else (edge_weight.view(-1, 1) * x_j.transpose(1, 0, 2)).transpose(1, 0, 2)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
