#include "hmf_trees.h"

#include <iostream>

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
    delete node;
    delete bottom_up_list;
    delete children;
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