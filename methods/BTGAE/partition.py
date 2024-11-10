from scipy.sparse import csr_matrix
import community as community_louvain
import networkx as nx
import numpy as np
import copy
import logging
from collections import defaultdict
import cvxpy as cp
from scipy.stats import linregress
import pymetis
import infomap
from scipy.sparse import lil_matrix

########## Graph Partitioning ##########
class GraphPartitioner:
    def __init__(self, graph, partition_type='louvain', size_bound=None):
        """
        Initialize the GraphPartitioner class.

        Parameters:
        - graph: csr_matrix
            The complete graph to be partitioned.
        - partition_type: str
            The type of partitioning method to be used. Default is 'louvain'.
        - size_bound: int, optional
            The upper bound of the size for each subgraph. If None, no bound is applied.
        """
        self.graph = graph
        self.partition_type = partition_type
        self.size_bound = size_bound
        self.partition_methods = {
            'louvain': self._louvain_partition,
            'balanced_lpa': self._balanced_lpa,
            'metis': self._metis_partition,
            'infomap': self._infomap_partition
        }
    def permute_adjacency_matrix(self,adj_matrix, communities_list):
        """
        Permute the adjacency matrix so that nodes within the same community are adjacent.

        Parameters:
        - adj_matrix (csr_matrix): The original adjacency matrix of the graph.
        - communities_list (List[List[int]]): A list of lists, where each inner list contains the node IDs within a community.

        Returns:
        - csr_matrix: The permuted adjacency matrix.
        """
        # Flatten the communities list to get a single list of node indices
        permutation = [node for community in communities_list for node in community]
        
        # Permute the rows of the adjacency matrix
        permuted_matrix = adj_matrix[permutation, :]
        
        # Permute the columns of the permuted row matrix
        permuted_matrix = permuted_matrix[:, permutation]
        
        return permuted_matrix    

    def partition(self):
        """
        Partition the graph based on the specified partitioning method.

        Returns:
        - List[csr_matrix]
            A list of csr_matrix, each representing a partitioned subgraph.
        """
        partition_method = self.partition_methods.get(self.partition_type)
        if not partition_method:
            raise ValueError(f"Partition method {self.partition_type} not recognized.")
        return partition_method()
    
    def _infomap_partition(self):
        """
        Apply the Infomap community detection method to partition the graph.

        Returns:
        - List[csr_matrix]
            A list of csr_matrix, each representing a partitioned subgraph.
        """
        G = nx.from_scipy_sparse_array(self.graph)

        # Initialize Infomap with the 'unsigned int' flag for nodes
        im = infomap.Infomap("--two-level")

        # Map networkx nodes to integers if they are not already
        node_to_int = {node: i for i, node in enumerate(G.nodes())}

        # Add edges to Infomap, adjust for weighted graph if necessary
        for edge in G.edges(data=True):
            src, dest = edge[:2]
            weight = edge[2].get('weight', 1.0)  # Default weight to 1.0 if not weighted
            im.addLink(node_to_int[src], node_to_int[dest], weight)

        # Run the Infomap search algorithm
        im.run()

        # Retrieve the communities
        subgraphs_nodes = {}
        for node_module in im.iterTree():
            if node_module.isLeaf():
                node = node_module.physicalId
                original_node = list(node_to_int.keys())[list(node_to_int.values()).index(node)]
                module = node_module.moduleIndex()
                if module not in subgraphs_nodes:
                    subgraphs_nodes[module] = []
                subgraphs_nodes[module].append(original_node)

        subgraph_list = []
        communities_list = []
        for module, nodes in subgraphs_nodes.items():
            if self.size_bound is None:  # Assuming size_bound logic is similar
                subgraph = G.subgraph(nodes)
                subgraph_csr = nx.to_scipy_sparse_array(subgraph)
                subgraph_list.append(subgraph_csr)
                communities_list.append(nodes)

        return subgraph_list,communities_list
    
    def _louvain_partition(self):
        """
        Apply the Louvain community detection method to partition the graph.

        Returns:
        - Tuple[List[List[int]], List[csr_matrix]]:
            A tuple where the first element is a list of lists, each containing the node IDs that belong to the same community,
            and the second element is a list of csr_matrix, each representing a partitioned subgraph corresponding to these communities.
        """

        G = nx.from_scipy_sparse_array(self.graph)
        partition = community_louvain.best_partition(G)
        subgraphs_nodes = {}

        for node, community in partition.items():
            if community not in subgraphs_nodes:
                subgraphs_nodes[community] = []
            subgraphs_nodes[community].append(node)

        communities_list = []
        subgraph_list = []
        for community, nodes in sorted(subgraphs_nodes.items()):
            if self.size_bound is None:
                subgraph = G.subgraph(nodes)
                subgraph_csr = nx.to_scipy_sparse_array(subgraph)
                subgraph_list.append(subgraph_csr)
                communities_list.append(nodes)

        return subgraph_list,communities_list 
    
    def _metis_partition(self, num_partitions=20):
        """
        Partition a graph using METIS.
        """
        N = self.graph.shape[0]
        # num_partitions = max(num_partitions, int(np.cbrt(N))+1)
        G = nx.from_scipy_sparse_array(self.graph)
        xadj = self.graph.indptr
        adjncy = self.graph.indices
        num_partitions = max(num_partitions, N//1000)
        # Use pymetis to partition the graph
        _, parts = pymetis.part_graph(num_partitions, xadj=xadj.tolist(), adjncy=adjncy.tolist())
            # Create subgraphs based on METIS partitioning
        subgraphs_nodes = {i: [] for i in range(num_partitions)}
        for node, part in enumerate(parts):
            subgraphs_nodes[part].append(node)
        communities_list = []
        subgraph_list = []
        for _, nodes in subgraphs_nodes.items():
            # Extract subgraph for the given set of nodes
            subgraph = G.subgraph(nodes)
            # Convert the NetworkX subgraph back to a CSR matrix
            subgraph_csr = nx.to_scipy_sparse_array(subgraph, dtype=int, format='csr')
            subgraph_list.append(subgraph_csr)
            communities_list.append(nodes)

        return subgraph_list,communities_list    
    def _balanced_lpa(self):
        """
        Apply a balanced label propagation algorithm for graph partitioning.

        Returns:
        - List[csr_matrix]
            A list of csr_matrix, each representing a partitioned subgraph.
        """
        adj = self.graph.toarray().astype(bool)
        # print(adj.shape)
        # print(adj.shape)
        num_nodes = adj.shape[0]
        # n^(1/2)
        num_communities = int(np.sqrt(num_nodes))
        var = 4
        node_threshold = num_nodes//num_communities * var # Define based on your partitioning strategy
        print("node threshold: ",node_threshold)
        terminate_delta = 0.005 # Define a termination criteria for the LPA iterations

        lpa = ConstrainedLPABase(adj, num_communities, node_threshold, terminate_delta)

        lpa.initialization()

        # Run the community detection
        num_iterations = 100
        communities, lpa_deltas = lpa.community_detection(num_iterations)  # Adjust this call according to your adapted method

        # Convert communities to subgraphs
        subgraph_list = []
        communities_list = []
        adj = csr_matrix(adj)
        G = nx.from_scipy_sparse_array(adj)
        for community_nodes in communities.values():
            subgraph = G.subgraph(community_nodes)
            subgraph_csr = nx.to_scipy_sparse_array(subgraph, dtype=int)
            subgraph_list.append(subgraph_csr)
            communities_list.append(community_nodes)

        return subgraph_list,communities_list
    
    def extract_global_adj_matrix(self,subgraph_list,adj):
        """
        Partitions the padded graph's adjacency matrix into smaller subgraphs and abstracts
        it into a super graph with weighted edges.
        """
        num_subgraphs = len(subgraph_list)
        super_adj_matrix = lil_matrix((num_subgraphs, num_subgraphs))

        row = 0
        for i in range(num_subgraphs-1):
            ilen = subgraph_list[i].shape[0]
            col = row + ilen
            
            for j in range(i+1,num_subgraphs):
                jlen = subgraph_list[j].shape[0]
                sub_ij = adj[row:row+ilen, col:col+jlen]
                # Update super node adj matrix edge weight
                super_adj_matrix[i, j] = sub_ij.sum()
                col+=jlen
            row += ilen
        
        super_adj_matrix = super_adj_matrix.tocsr()
        super_adj_matrix += super_adj_matrix.T
        
        return super_adj_matrix

########## Balanced Label Propagation Algorithm ##########
class ConstrainedLPABase:
    def __init__(self, adj, num_communities, node_threshold, terminate_delta):
        self.logger = logging.getLogger('constrained_lpa_base')

        self.adj = adj
        self.num_nodes = adj.shape[0]
        self.num_communities = num_communities
        self.node_threshold = node_threshold
        self.terminate_delta = terminate_delta

    def initialization(self):
        self.logger.info('initializing communities')

        random_nodes = np.arange(self.num_nodes)
        # np.random.shuffle(random_nodes)
        self.communities = defaultdict(set)
        self.node_community = np.zeros(self.adj.shape[0])

        # each node use node is as its community label
        for community, nodes in enumerate(np.array_split(random_nodes, self.num_communities)):
            self.communities[community] = set(nodes)
            self.node_community[nodes] = community

    def community_detection(self, iterations=100):
        self.logger.info('detecting communities')

        communities = copy.deepcopy(self.communities)
        lpa_deltas = []

        for i in range(iterations):
            self.logger.info('iteration %s' % (i,))

            ## Step 1: calculate desired move
            desire_move = self._determine_desire_move()
            relocation = {}
            utility_func = {}

            ## Step 2: calculate parameters for linear programming problem
            for src_community in range(self.num_communities):
                for dst_community in range(self.num_communities):
                    move_node = desire_move[np.where(np.logical_and(desire_move[:, 1] == src_community, desire_move[:, 2] == dst_community))[0]]

                    if src_community != dst_community and move_node.size != 0:
                        move_node = move_node[np.flip(np.argsort(move_node[:, 3]))]
                        relocation[(src_community, dst_community)] = move_node

                        if move_node.shape[0] == 1:
                            utility_func[(src_community, dst_community)] = np.array([[0, move_node[0, 3]]])
                        else:
                            cum_sum = np.cumsum(move_node[:, 3])
                            utility_func_temp = np.zeros([move_node.shape[0] - 1, 2])
                            for k in range(move_node.shape[0] - 1):
                                utility_func_temp[k, 0], utility_func_temp[k, 1], _, _, _ = linregress([k, k+1], [cum_sum[k], cum_sum[k+1]])
                                utility_func[(src_community, dst_community)] = utility_func_temp

            ## Step 3: solve linear programming problem
            x = cp.Variable([self.num_communities, self.num_communities])
            z = cp.Variable([self.num_communities, self.num_communities])

            objective = cp.Maximize(cp.sum(z))
            constraints = []
            for src_community in range(self.num_communities):
                const = 0
                for dst_community in range(self.num_communities):
                    if (src_community, dst_community) in relocation:
                        if src_community == dst_community:
                            constraints.append(x[src_community, dst_community] == 0)
                            constraints.append(z[src_community, dst_community] == 0)
                        else:
                            ## Constraint 2 of Theorem 2
                            constraints.append(x[src_community, dst_community] >= 0)
                            constraints.append(x[src_community, dst_community] <= relocation[(src_community, dst_community)].shape[0])

                            ## Constraint 1 of Theorem 2
                            if (dst_community, src_community) in relocation:
                                const += x[src_community, dst_community] - x[dst_community, src_community]

                        ## Constraint 3 of Theorem 2
                        for utility_func_value in utility_func[(src_community, dst_community)]:
                            constraints.append(- utility_func_value[0] * x[src_community, dst_community] + z[src_community, dst_community] <= utility_func_value[1])

                    else:
                        constraints.append(x[src_community, dst_community] == 0)
                        constraints.append(z[src_community, dst_community] == 0)

                ## Constraint 1 of Theorem 2
                constraints.append(len(self.communities[src_community]) + const <= self.node_threshold)

            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS)
            ## Step 4: parse linear programming problem results
            if problem.status == 'optimal':
                x_value = np.floor(np.abs(x.value)).astype(np.int64)
                for src_community in range(self.num_communities):
                    for dst_community in range(self.num_communities):
                        if (src_community, dst_community) in relocation and x_value[src_community, dst_community] != 0:
                        # if (src_community, dst_community) in relocation:
                            relocation_temp = relocation[(src_community, dst_community)][:, 0].astype(np.int64)
                            move_node = relocation_temp[:x_value[src_community, dst_community] - 1]
                            if isinstance(move_node, np.int64):
                                self.communities[src_community].remove(move_node)
                                self.communities[dst_community].add(move_node)
                                self.node_community[move_node] = dst_community
                            else:
                                # move_node = set(move_node)
                                self.communities[src_community].difference_update(move_node)
                                self.communities[dst_community].update(move_node)
                                for node in move_node:
                                    self.node_community[node] = dst_community
            else:
                self.logger.info("No optimal solution, break!")
                break

            ## Check the number of moved nodes
            delta = self._lpa_delta(communities, self.communities)
            lpa_deltas.append(delta)
            self.logger.info("%d" % delta)
            communities = copy.deepcopy(self.communities)
            if delta <= self.terminate_delta:
                break

        return self.communities, lpa_deltas

    def _determine_desire_move(self):
        desire_move = []

        for i in range(self.num_nodes):
            # neighbor_community = self.node_community[np.nonzero(self.adj[i])[0]]  # for non-bool adj
            neighbor_community = self.node_community[self.adj[i]] # for bool adj
            unique_community, unique_count = np.unique(neighbor_community, return_counts=True)

            src_relocation = unique_count[np.where(unique_community == self.node_community[i])[0]]
            for community in unique_community:
                if community != self.node_community[i]:
                    dst_relocation = unique_count[np.where(unique_community == community)[0]]
                    if dst_relocation - src_relocation >= 0:
                        desire_move_temp = np.zeros(4)
                        desire_move_temp[0] = i
                        desire_move_temp[1] = self.node_community[i]
                        desire_move_temp[2] = community
                        desire_move_temp[3] = dst_relocation - src_relocation

                        desire_move.append(desire_move_temp)

        return np.stack(desire_move)

    def _lpa_delta(self, lpa_pre, lpa_cur):
        delta = 0.0
        for i in range(len(lpa_cur)):
            delta += len((lpa_cur[i] | lpa_pre[i]) - (lpa_cur[i] & lpa_pre[i]))

        return delta
