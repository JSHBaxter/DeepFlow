#include "hmf_trees.h"

#include <iostream>
#include <deque>

//guarantee that leaves appear first in the bottom up list and last in the top-down list
void TreeNode::build_tree(TreeNode* &node, TreeNode** &children, TreeNode** &bottom_up_list, TreeNode** &top_down_list, const int* parent_t, const int n_r, const int n_c){
        
    //create tree to structure things
    node = new TreeNode[n_r+1];
    children = new TreeNode*[n_r];
    
    //start building tree (note it is top-down by pre-condition)
    node[0].parent = NULL;
    node[0].r = -1;
    node[0].d = -1;
    node[0].c = 0;
    for(int n = 0; n < n_r; n++){
        node[n+1].parent = &(node[parent_t[n]+1]);
        node[n+1].r = n;
        node[n+1].d = (n < n_r-n_c) ? -1 : n - (n_r-n_c);
        node[n+1].c = 0;
    }
    for(int n = 1; n < n_r+1; n++)
        node[n].parent->c++;
    
    //populate children lists
    int bump = 0;
    for(int n = 0; n < n_r+1; n++){
        node[n].children = children+bump;
        bump += node[n].c;
        node[n].c = 0;
    }
    for(int n = 0; n < n_r; n++){
        TreeNode* parent = node[n+1].parent;
        parent->children[parent->c++] = &(node[n+1]);
    }
    
    //get orderings
    bottom_up_list = new TreeNode*[2*(n_r+1)];
    top_down_list = bottom_up_list + (n_r+1);
    for(int n = 0; n < n_r+1; n++){
        top_down_list[n] = &(node[n]);
        bottom_up_list[n] = &(node[n_r-n]);
    }
    
}

void TreeNode::free_tree(TreeNode* &node, TreeNode** &children, TreeNode** &bottom_up_list, TreeNode** &top_down_list){
    delete [] node;
    delete [] bottom_up_list;
    delete [] children;
}

void TreeNode::print_list(TreeNode** list, const int n_r){
    for(int c = 0; c < n_r; c++)
        std::cout << "Node " << list[c]->r << " (" << list[c]->d << ")" << std::endl;
}

void TreeNode::print_tree(){
    std::cout << "Self: " << this->r << "(" << this->d << ")" << std::endl;
    if (this->parent == NULL)
        std::cout << "  Parent: NULL" << std::endl;
    else
        std::cout << "  Parent: " << this->parent->r << std::endl;
    for(int i = 0; i < this->c; i++)
        std::cout << "  Child: " << this->children[i]->r << std::endl;
    for(int i = 0; i < this->c; i++)
        this->children[i]->print_tree();
}


void DAGNode::build_tree(DAGNode* &node, DAGNode** &children, DAGNode** &bottom_up_list, DAGNode** &top_down_list, const float* parent_t, const int n_r, const int n_c){
    
    //get number of connections
    int count = 0;
    int source_child_count = 0;
    for(int c = 0; c < n_r; c++){
        int p_count = 0;
        float w_count = 0.0f;
        for(int p = 0; p < n_r; p++){
            p_count += (parent_t[p*n_r + c] > 0) ? 1 : 0;
            w_count += parent_t[p*n_r + c];
        }
        if(w_count < 1.0f){
            p_count += 1;
            source_child_count++;
        }
        count += p_count;
    }
    children = new DAGNode*[2*count];
    
    //initialise essentially empty nodes with just parent and child lists populated
    int leaf_count = 0;
    node = new DAGNode[n_r+1];
    DAGNode** children_ptr = children + source_child_count;
    DAGNode** parent_ptr = children + count;
    node[0].c = 0;
    node[0].p = 0;
    node[0].r = -1;
    node[0].d = -1;
    node[0].parents = 0;
    node[0].children = children;
    for(int r = 0; r < n_r; r++){
        node[r+1].c = 0;
        node[r+1].p = 0;
        node[r+1].r = r;
        node[r+1].d = -1;
     
        //populate child lists
        node[r+1].children = children_ptr;
        for(int c = 0; c < n_r; c++){
            if(parent_t[r*n_r+c] == 0)
                continue;
            *children_ptr = &(node[c+1]);
            children_ptr++;
            node[r+1].c++;
        }
        
        //if there are no children, set data term index
        if(node[r+1].c == 0){
            node[r+1].d = leaf_count;
            leaf_count++;
        }
        
        //populate parent list (if no parent, add source as parent)
        node[r+1].parents = parent_ptr;
        float w_count = 0.0;
        for(int p = 0; p < n_r; p++){
            if(parent_t[p*n_r+r] == 0)
                continue;
            w_count += parent_t[p*n_r+r];
            *parent_ptr = &(node[p+1]);
            parent_ptr++;
            node[r+1].p++;
        }
        
        //if there is not enough weight in graph, add source as sole parent
        if(w_count < 1.0){
            node[r+1].p++;
            node[0].children[node[0].c] = &(node[r+1]);
            node[0].c++;
            *parent_ptr = &(node[0]);
            parent_ptr++;
        }
    }
    
    //make containers for weights
    for(int r = 0; r < n_r+1; r++){
        node[r].parent_weight = (node[r].p > 0) ? new float[node[r].p] : 0;
        node[r].child_weight  = (node[r].c > 0) ? new float[node[r].c] : 0;
    }
    
    //make the tree's parent weights
    for(int r = 0; r < n_r; r++){
        float total_weight = 0.0f;
        for(int p = 0; p < node[r+1].p; p++){
            int pr = node[r+1].parents[p]->r;
            if(pr < 0)
                continue;
            node[r+1].parent_weight[p] = parent_t[pr*n_r+r];
            total_weight += parent_t[pr*n_r+r];
        }
        for(int p = 0; p < node[r+1].p; p++){
            int pr = node[r+1].parents[p]->r;
            if(pr >= 0)
                continue;
            node[r+1].parent_weight[p] = 1-total_weight;
        }
    }
    
    //and child weights
    for(int r = 0; r < n_r+1; r++){
        node[r].sum_child_weight_sqr = 0;
        for(int c = 0; c < node[r].c; c++){
            int nr = node[r].r;
            int cr = node[r].children[c]->r;
            if( nr > -1 ){
                node[r].child_weight[c] = parent_t[nr*n_r+cr];
                node[r].sum_child_weight_sqr += parent_t[nr*n_r+cr]*parent_t[nr*n_r+cr];
                continue;
            }
        }
    }
    for(int nc = 0; nc < node[0].c; nc++){
        DAGNode* child = node[0].children[nc];
        for(int np = 0; np < child->p; np++){
            if(child->parents[np] == &(node[0])){
                node[0].child_weight[nc] = child->parent_weight[np];
                node[0].sum_child_weight_sqr += child->parent_weight[np]*child->parent_weight[np];
                break;
            }
        }
    }
            
    
    
    //start a DAG traversal to get the bottom-up lists
    top_down_list = new DAGNode*[2*(n_r+1)];
    int top_down_list_idx = 0;
    bool* in_list = new bool[n_r+1];
    for(int r = 0; r < n_r+1; r++)
        in_list[r] = false;
    std::deque<DAGNode*> queue;
    queue.push_front(node);
    while( !(queue.empty()) ){
        DAGNode* curr_node = queue.front();
        queue.pop_front();
        if(in_list[curr_node->r+1])
            continue;
        for(int p = 0; p < curr_node->p; p++)
            if(!in_list[curr_node->parents[p]->r+1]){
                queue.push_front(curr_node);
                queue.push_front(curr_node->parents[p]);
            }
        
        for(int c = 0; c < curr_node->c; c++)
            queue.push_back(curr_node->children[c]);
        top_down_list[top_down_list_idx] = curr_node;
        top_down_list_idx++;
        in_list[curr_node->r+1] = true;
    }
    delete [] in_list;
    
    //reverse top-down list to get the bottom-up one
    bottom_up_list = top_down_list + n_r + 1;
    for(int r = 0; r < n_r+1; r++)
        bottom_up_list[r] = top_down_list[n_r-r];
    
    
    
}

void DAGNode::free_tree(DAGNode* &node, DAGNode** &children, DAGNode** &bottom_up_list, DAGNode** &top_down_list){
    //traverse tree to free internal arrays
    
    
    //free overall structure
    delete [] node;
    delete [] top_down_list;
    delete [] children;
}

void DAGNode::print_list(DAGNode* list, const int n_r){
    for(int r = 0; r < n_r+1; r++){
        std::cout << "Node [" << r << "]" << std::endl;
        std::cout << "r:" << list[r].r << std::endl;
        std::cout << "d:" << list[r].d << std::endl;
        std::cout << "p:" << list[r].p;
        for(int p = 0; p<list[r].p; p++)
            std::cout << "\t" << list[r].parents[p]->r;
        std::cout << std::endl;
        for(int p = 0; p<list[r].p; p++)
            std::cout << "\t" << list[r].parent_weight[p];
        std::cout << std::endl;
        std::cout << "c:" << list[r].c;
        for(int c = 0; c<list[r].c; c++)
            std::cout << "\t" << list[r].children[c]->r;
        std::cout << std::endl;
        for(int c = 0; c<list[r].c; c++)
            std::cout << "\t" << list[r].child_weight[c];
        std::cout << std::endl;
        std::cout << std::endl;
    }
}