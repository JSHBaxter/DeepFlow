#ifndef HMF_TREES
#define HMF_TREES

struct TreeNode {
    TreeNode* parent;
    int r;
    int d;
    int c;
    TreeNode** children;
    
    void print_tree();
    
    static void build_tree(TreeNode* &node, TreeNode** &children, TreeNode** &bottom_up_list, TreeNode** &top_down_list, const int* parent_t, const int* index_t, const int n_r, const int n_c);
    static void free_tree(TreeNode* &node, TreeNode** &children, TreeNode** &bottom_up_list, TreeNode** &top_down_list);
    static void print_list(TreeNode** list, const int n_r);
};

//guarantee that leaves appear first in the bottom up list and last in the top-down list

#endif //#ifndef HMF_TREES