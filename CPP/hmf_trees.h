
#ifndef HMF_TREES
#define HMF_TREES

//guarantee that leaves appear first in the bottom up list and last in the top-down list
struct TreeNode {
    TreeNode* parent;
    int r;
    int d;
    int c;
    TreeNode** children;
    
    void print_tree();
    
    static void build_tree(TreeNode* &node, TreeNode** &children, TreeNode** &bottom_up_list, TreeNode** &top_down_list, const int* parent_t, const int n_r, const int n_c);
    static void free_tree(TreeNode* &node, TreeNode** &children, TreeNode** &bottom_up_list, TreeNode** &top_down_list);
    static void print_list(TreeNode** list, const int n_r);
};

//guarantee that leaves appear first in the bottom up list and last in the top-down list
struct DAGNode {
    int r;
    int d;
    int c;
    int p;
    DAGNode** parents;
    DAGNode** children;
    float* parent_weight;
    float* child_weight;
    float sum_child_weight_sqr;
    
    void print_tree();
    
    static void build_tree(DAGNode* &node, DAGNode** &children, DAGNode** &bottom_up_list, DAGNode** &top_down_list, const float* parent_t, const int n_r, const int n_c);
    static void free_tree(DAGNode* &node, DAGNode** &children, DAGNode** &bottom_up_list, DAGNode** &top_down_list);
    static void print_list(DAGNode* list, const int n_r);
};

#endif //#ifndef HMF_TREES
