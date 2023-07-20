import numpy as np
import pandas as pd
from collections import deque
from itertools import combinations
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns.fpgrowth import fpgrowth

class VectorToTransactions:
    """
    Transformers vectorized data into transaction style. 
    Floating type data is converted into bins. Classes and Integers are kept.
    """
    def __init__(self, n_bins=5):
        self.metadata = {}
        self.n_bins = n_bins
        self.bin_step = 1/(n_bins-2)
        
    def fit(self, X):
        for column in X:
            if X[column].dtype==float:
                metadata = {
                    "type":"numeric",
                    "bins":[],
               }
                #generate {n_bins}, including "outlier ranges" with -inf and inf
                low = -float("inf")
                for i in range(self.n_bins - 1):
                    high = np.round(
                        np.quantile( X[column], i * self.bin_step ), 3
                    )
                    window = (low, high)
                    metadata["bins"].append(window)
                    low = high
                    
                window = (low, float("inf"))
                metadata["bins"].append(window)
                self.metadata[column] = metadata
            else:
                self.metadata[column] = {
                    "type":"categorical",
                    "categories": frozenset(np.unique(X[column]))
                }

        return self
                
    def _apply_transformation(self, row):
        trans = []
        for col, value in row.items():
            if col in self.metadata:
                metadata = self.metadata[col]
                
                if metadata["type"] == "numeric":
                    
                    for bin_range in metadata["bins"]:
                        if bin_range[0]<=value<bin_range[1]:
                            trans.append(
                                f"{col}={bin_range}"
                            )
                            break
                else:
                    if value in metadata["categories"]:
                        trans.append(
                            f"{col}={int(value)}"
                        )
                    else: trans.append(f"{col}=other")
        return trans
    
    def transform(self, X):
        transitions = []
        for i in X.index:
            transitions.append(
                self._apply_transformation(X.loc[i])
            )
        return transitions
    
    
    
""" #######################################

    Implementation of Apriori Algorithm

####################################### """  

class Apriori:
    """Implementation of the Apriori Algorithm"""
    def __init__(self, min_support):
        self.freq_itemsets = {}
        self.min_support = min_support
        self.is_fit = False
        self.unique_items = None
        
    def filter_itemsets(self, items):
        """Filters out the itemsets with a support < {min_support}"""
        return dict(
            filter(
                lambda x: x[1]>=self.min_support, items.items()
            )
        )
        
    def create_base_itemsets(self, X):
        """Generates 1-length frequent itemsets from X"""
        N = len(X)
        itemsets = dict(
            map(
                lambda item, count: (frozenset([item]), count/N), *np.unique(X, return_counts=True)
            )
        )        
        itemsets = self.filter_itemsets(itemsets)
        self.freq_itemsets|=itemsets
        self.unique_items = frozenset(itemsets.keys())
    
    def fit(self, X):
        self.create_base_itemsets(X)
        self.is_fit=True
        return self 
    
    def get_support(self, itemset, X):
        if itemset in self.freq_itemsets:
            return self.freq_itemsets[itemset]
        else:
            N = len(X)
            supp=0
            for transaction in X:
                if itemset.issubset(transaction):
                    supp+=1/N
            return supp
    
    def generate_new_itemsets(self, X, current_items):
        """Generates l+1 lenght itemsets from the itemsets with minimum support threshold"""
        N = len(X)
        itemsets = {}
        for item_pair in combinations(current_items, 2):
            if len(item_pair[0].difference(item_pair[1])) == 1:
                itemsets[frozenset(
                    item_pair[0]|item_pair[1]
                )] = 0
        
        for transaction in X:
            for item in itemsets:
                if item.issubset(transaction):
                    itemsets[item] += 1/N
                    
        return itemsets
        
    def mine(self, X, verbose=0):
        if self.is_fit:
            i=1
            previous_items = self.freq_itemsets.copy()
                
            if verbose==1: print(f"previous length = {len(previous_items)} | generating {i}-length ...")
            
            current_items = self.generate_new_itemsets(X, previous_items)
            current_items = self.filter_itemsets(current_items)
            i+=1
            while len(current_items)>0:
                self.freq_itemsets|=current_items

                previous_items = current_items
                if verbose==1: print(f"preivous length = {len(previous_items)} | generating {i}-length ...")
                current_items = self.generate_new_itemsets(X, previous_items)
                current_items = self.filter_itemsets(current_items)
                i+=1

            print("Generated the frequent itemsets!")
            
        else:
            print("Model needs to be fit to the database before mining it!")

            
""" #######################################

    Implementation of FP-Growth Algorithm

####################################### """            
            
class FPNode:
    def __init__(self, node_count, depth, node_value=None, parent=None, children=dict()):
        self.parent = parent
        self.children = children
        self.node_value = node_value
        self.node_count = node_count
        self.depth = depth
        

class FPGrowth:
    def __init__(self, min_support):
        self.freq_items = {}
        self.min_support = min_support
        self.is_fit = False
        self.root = FPNode(0, 0)
        
    def fit(self, X):
        v,c = np.unique(X, return_counts=True)
        self.item_nodes = {value:[] for value in v}
        self.order = dict(zip(v,c))
        self.is_fit = True
        return self
    
    def get_pattern(self, node):
        pattern = frozenset()
        while node.parent!=None:
            pattern|=frozenset([node.node_value])
            node = node.parent
        return pattern
    
    def parse_tree(self, node, prefix):
        prefix |= frozenset([node.node_value])
        for i in range(1, len(prefix)+1):
            for pattern in combinations(prefix, i):
                pattern = frozenset(pattern)
                self.freq_items[pattern] = self.freq_items.get(pattern, 0) + node.node_count
        
        for child in node.children.values():
            self.parse_tree(node, prefix)
        
    def generate_patterns(self):
        node = self.root
        
        print("Parsing Tree..")
        for node in self.root.children.values():
            self.parse_tree(node, frozenset())
            
    def mine(self, X):
        if self.is_fit:
            N = len(X)
            for transaction in X:
                sorted_trans = sorted(
                    transaction, 
                    key=lambda x: self.order[x], 
                    reverse=True
                )
                node = self.root

                for item in sorted_trans:
                    if item in node.children:
                        child_node = node.children[item]
                        child_node.node_count += 1/N
                        
                    else:
                        child_node = FPNode(
                            node_count=1/N, 
                            depth = node.depth+1,
                            node_value=item, 
                            parent=node,
                            children=dict()
                        )
                        self.item_nodes[item].append(child_node)
                        node.children[item] = child_node
                        
                    node = child_node
            
            print("Generating Patterns...")
            self.generate_patterns()
        else:
            print("Model needs to be fit to the database before mining it!")
            
            

            
class FrequentPattern:
    def __init__(self, dataset, min_supp):
        self.ds = dataset
        self.mine(min_supp)
        self.closed_items = None
        self.max_items = None
        
    def mine(self, supp):
        self.enc = TransactionEncoder().fit(self.ds)
        self.enc_ds = self.enc.transform(self.ds)
        self.enc_ds = pd.DataFrame(data=self.enc_ds, columns=self.enc.columns_)
        self.freq_items = fpgrowth(self.enc_ds, min_support=supp, use_colnames=True)
    
    def get_maximal(self):
        if self.max_items is None:
            fi = self.get_closed()
            
            set_size = fi.itemsets.apply(len)
            min_set_size, max_set_size = set_size.min(), set_size.max()
            max_fi = pd.DataFrame(columns=["support", "itemsets"])
            
            for size in range(min_set_size, max_set_size+1):
                sets = fi[set_size==size]
                super_sets = fi[set_size==size+1]
                for i in sets.index:
                    row = sets.loc[i]
                    itemset = row.itemsets
                    matching_supersets = super_sets[
                        super_sets.itemsets.apply(lambda item: (itemset&item) == itemset)
                    ]
                    if len(matching_supersets)==0:
                        max_fi.loc[i] = row
            self.max_items = max_fi
        return self.max_items
    
    def get_closed(self):
        if self.closed_items is None:
            fi = self.freq_items
            set_size = fi.itemsets.apply(len)
            min_set_size, max_set_size = set_size.min(), set_size.max()
            closed_fi = pd.DataFrame(columns=["support", "itemsets"])
            
            for size in range(min_set_size, max_set_size+1):
                sets = fi[set_size==size]
                super_sets = fi[set_size==size+1]
                for i in sets.index:
                    row = sets.loc[i]
                    itemset = row.itemsets
                    matching_supersets = super_sets[
                        super_sets.itemsets.apply(lambda item: (itemset&item) == itemset)
                    ]
                    if (len(matching_supersets)==0) or \
                    (matching_supersets.support!=row.support).all():
                        closed_fi.loc[i] = row   
                        
            self.closed_items = closed_fi
        return self.closed_items
    
    
    def get_support(self, pattern):
        if pattern in self.freq_items:
            idx = self.freq_items.itemsets==pattern
            return self.freq_items[idx]
        
        N=len(self.ds)
        supp=0
        for transaction in self.ds:
            if pattern.issubset(transaction):
                supp+=1/N
        return supp
    
    def __iter__(self):
        self.index=-1
        return self
    
    def __next__(self):
        self.index+=1
        if self.index == len(self.freq_items):
            raise StopIteration    
        return self.freq_items.loc[self.index, "itemsets"]