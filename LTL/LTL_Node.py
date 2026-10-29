from typing import List, Tuple
from sortedcontainers import SortedSet

class LTL_Node:
    def __init__(self,name,index):
        self.name = name
        self.index = index
        self.leaves = SortedSet(self.get_leaves())
        self.level = 0
    
    def __str__(self):
        return self.name
    
    def __lt__(self, other):
        return self.index < other.index
    
    # These methods need to be implemented by the Leaf and Group subclasses
    def get_leaves(self):
        raise NotImplement
    def verify_correctness(self,leaves,groups):
        raise NotImplement

class LTL_LeafNode(LTL_Node):
    def __init__(self,name,index):
        super().__init__(name,index)
        
    def get_leaves(self):
        return [self]
    
    def verify_correctness(self,leaves,groups):
        if self not in leaves:
            raise ValueError("Group verification failed - leaf "+self.name+" missing from list")
    
class LTL_GroupNode(LTL_Node):
    def __init__(self,name,index,children):
        self.children = [c for c in children]
        super().__init__(name,index)
        self.level = max([c.level for c in self.children])+1
    
    def __str__(self):
        stringname = self.name + " {"
        for c in self.children:
            stringname += c.name+", "
        return stringname[:-2] + "}"
    
    def __lt__(self, other):
        return self.level < other.level
            
    def __len__(self):
        return len(self.children)
    
    def get_leaves(self):
        ret_list = []
        for c in self.children:
            ret_list += c.get_leaves()
        return ret_list
    
    #verify groups are in list and form a DAG
    def verify_correctness(self,leaves,groups):
        if self.flag:
            raise ValueError("Group verification failed - not a DAG")
        if self not in groups:
            raise ValueError("Group verification failed - group "+self.name+" missing from list")
        self.flag = True
        for c in self.children:
            if not issubclass(type(c),LTL_Node):
                raise ValueError("Group verification failed - group "+self.name+" contains invalid child")
    
class LTL_GroupingStructure:
    def __init__(self, leaves : List[LTL_LeafNode],
                       group_list : List[LTL_GroupNode]):
        self.leaves = SortedSet(leaves)
        self.groups = SortedSet(group_list)
        
        #verify index correctness
        indices = SortedSet()
        for l in self.leaves:
            if l.index not in indices:
                indices.add(l.index)
            else:
                raise ValueError("Leaf index collision : "+str(l.index))
        for i in range(len(self.leaves)):
            if i not in indices:
                raise ValueError("Leaf index missing: "+str(i))
        for g in self.groups:
            if g.index not in indices:
                indices.add(g.index)
            else:
                raise ValueError("Group index collision : "+str(g.index))
        for i in range(len(self.leaves),len(self.leaves)+len(self.groups)):
            if i not in indices:
                raise ValueError("Group index missing: "+str(i))
        
        #verify groups are all included and only reference other groups and leaves - should raise error if incorrect
        for g in self.groups:
            for n in self.groups:
                g.flag=False
            g.verify_correctness(self.leaves,self.groups)
    
    def __str__(self):
        stringname = "Leaves : {"
        for l in self.leaves:
            stringname += l.name+", "
        stringname = stringname[:-2]+"}\nGroups: {"
        for g in self.groups:
            stringname += g.name+",\n         "
        stringname = stringname[:-11]+"}"
        return stringname
            
    def __len__(self):
        return len(self.leaves)+len(self.groups)
        
    def check_vars_dims(self, d, r):
        if r is not None and d.shape[0] != r.shape[0]:
            raise ValueError("Batch size needs to be the same for both the data and regularisation terms")
        if r is not None and d.shape[1] != r.shape[1]:
            raise ValueError("Number of channels needs to be the same for both the data and regularisation terms")
        if r is not None and d.shape[2] != r.shape[2]+1:
            raise ValueError("The length of the regularisation signal must be one less than that of the data signal")
        if d.shape[1] != len(self):
            raise ValueError("Number of channels disagrees with number of leaves and groups in the hierarchy")

    def copy(self):
        return LTL_GroupingStructure(list(self.leaves),list(self.groups))