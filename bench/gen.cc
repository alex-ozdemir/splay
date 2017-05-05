#include <algorithm>
#include <assert.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdlib.h>
#include <utility>
#include <vector>


#define usize std::size_t // Too used to rust.
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;

uint32_t pcg32_random_r(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Directions that one might want to rotate
enum Direction {
    CCW, CW, NONE
};

template <typename T>
struct tree_node;

template<typename T>
tree_node<T> * link(typename std::vector<tree_node<T>*>::const_iterator& internal_nodes,
                    typename std::vector<tree_node<T>*>::const_iterator& leaf_nodes,
                    std::vector<bool>::const_iterator& result);

std::ostream& operator<<(std::ostream& out, const std::vector<bool>& bitvec)
{
    for (bool bit : bitvec) {
        out << (bit ? 'X' : 'O');
    }
    return out;
}


struct ruleset {
    std::vector<bool> result;
    ruleset * left;
    ruleset * right;

    ruleset()
        : left(nullptr),
          right(nullptr)
    { /* Defaults are fine */ }

    static void deallocate(ruleset * n)
    {
        if (n != nullptr) {
            deallocate(n->left);
            deallocate(n->right);
            delete n;
        }
    }

    template <typename T>
    static ruleset * make_and_get(ruleset * root, T str_iter, T str_end)
    {
        if (str_iter == str_end) {
            return root;
        } else if (*str_iter == 'L') {
            if (root->left == nullptr) root->left = new ruleset();
            str_iter++;
            return make_and_get(root->left, str_iter, str_end);
        } else if (*str_iter == 'R') {
            if (root->right == nullptr) root->right = new ruleset();
            str_iter++;
            return make_and_get(root->right, str_iter, str_end);
        } else {
            str_iter++;
            return make_and_get(root, str_iter, str_end);
        }
    }

    // Verifies that every path has a rule and that the rules are of the
    // correct size.
    bool are_rules_okay()
    {
        std::string path;
        return check_missing_rules_helper(path);
    }

    bool check_missing_rules_helper(std::string& path)
    {
        // We exclude zero length paths because the root will not have a result.
        if (path.size() > 0 && result.size() == 0) {
            std::cerr << "Found missing rule at path {" << path << "}" << std::endl;
            return false;
        }
        if (2 * path.size() + 3 != result.size()) {
            std::cerr << "The result [" << result << "] at path {"
                      << "} has length " << result.size()
                      << " but the path has length " << path.size()
                      << std::endl;
            return false;
        }
        int zeros = std::count(result.begin(), result.end(), false);
        int ones = std::count(result.begin(), result.end(), true);
        if (zeros + 1 != ones) {
            std::cerr << "Found " << zeros << " internal nodes and " << ones 
                      << " leaves in the result " << result << " at path {"
                      << path << "}" << std::endl;
            return false;
        }
        if (left != nullptr) {
            path.push_back('L');
            if (!left->check_missing_rules_helper(path)) {
                return false;
            }
            path.pop_back();
        }
        if (right != nullptr) {
            path.push_back('R');
            if (!right->check_missing_rules_helper(path)) {
                return false;
            }
            path.pop_back();
        }
        return true;
    }
};

// Format: a path followed by a RPN description of the result.
//
// LL OXOXOXX
// RR OOOXXXX
// LR OOXXOXX
// RL OOXXOXX
//
// The fields are expected to be whitespace delimited. The O's are node, the
// X's are null terminators.
ruleset make_ruleset(const std::string& s)
{
    std::stringstream in_stream(s, std::ios_base::in);
    ruleset root;
    root.result = {0, 1, 1};
    while (!in_stream.eof()) {
        std::string path, result;
        in_stream >> path >> result;
        if (result.size() > 0) {
            ruleset * template_ = ruleset::make_and_get(&root, path.begin(), path.end());
            if (template_->result.size() > 0) {
                std::cerr << "Tried to assign path {" << path << "} to result ["
                          << result << "], but it was already set to ["
                          << template_->result << "]" << std::endl;
                std::cerr << "  -> Overwriting" << std::endl;
                template_->result.clear();
            }
            for (char c : result) template_->result.push_back(c == 'X');
        }
    }
    if (!root.are_rules_okay()) exit(1);
    return root;
}

ruleset make_inverted_ruleset(const std::string& s)
{
    std::stringstream in_stream(s, std::ios_base::in);
    ruleset root;
    root.result = {0, 1, 1};
    while (!in_stream.eof()) {
        std::string path, result;
        in_stream >> path >> result;
        if (result.size() > 0) {
            ruleset * template_ = ruleset::make_and_get(&root, path.rbegin(), path.rend());
            if (template_->result.size() > 0) {
                std::cerr << "Tried to assign path {" << path << "} to result ["
                          << result << "], but it was already set to ["
                          << template_->result << "]" << std::endl;
                std::cerr << "  -> Overwriting" << std::endl;
                template_->result.clear();
            }
            for (char c : result) template_->result.push_back(c == 'X');
        }
    }
    if (!root.are_rules_okay()) exit(1);
    return root;
}

template <typename T>
struct tree_node {
    T data;
    tree_node<T> * left;
    tree_node<T> * right;

    tree_node(T data)
        : data(data),
          left(nullptr),
          right(nullptr)
    { /* Nothing left to do */ }

    static void deallocate(tree_node<T> * n)
    {
        if (n != nullptr) {
            deallocate(n->left);
            deallocate(n->right);
            delete n;
        }
    }

    static void print(std::ostream& out, tree_node<T> * n)
    {
        if (n != nullptr) {
            if (n->left != nullptr || n->right != nullptr) {
                out << '(' << n->data << ' ';
                print(out, n->left);
                out << ' ';
                print(out, n->right);
                out << ')';
            } else {
                out << n->data;
            }
        } else
            out << '-';
    }

    // We hold pointers into our tree and our ruleset and traverse them in
    // lockstep till we find the item, fail to find it, or run out of rules.
    //
    // At that point we apply the rule we're at.
    //
    // Assumes a non-empty tree.
    //
    // Returns whether the item/a nullptr was found and which rule (really
    // which result) was employed.
    static std::pair<bool, const std::vector<bool> *> apply_rule_at(
            tree_node<T> * & root,
            const T& item,
            const ruleset& rules)
    {
        tree_node<T> * tree = root;
        const ruleset * rule = &rules;
        std::vector<tree_node<T> *> left_internal_nodes;
        std::vector<tree_node<T> *> right_internal_nodes;
        std::vector<tree_node<T> *> left_leaf_nodes;
        std::vector<tree_node<T> *> right_leaf_nodes;
        bool all_done = false; // Did you find `item` or a nullptr?
        while (true) {
            if (tree->data > item) {
                // Look left
                right_internal_nodes.push_back(tree);
                right_leaf_nodes.push_back(tree->right);
                if (tree->left == nullptr || rule->left == nullptr) {
                    left_leaf_nodes.push_back(tree->left);
                    all_done = (tree->left == nullptr);
                    break;
                } else {
                    tree = tree->left;
                    rule = rule->left;
                }
            } else if (item > tree->data) {
                // Look right
                left_internal_nodes.push_back(tree);
                left_leaf_nodes.push_back(tree->left);
                if (tree->right == nullptr || rule->right == nullptr) {
                    right_leaf_nodes.push_back(tree->right);
                    all_done = (tree->right == nullptr);
                    break;
                } else {
                    tree = tree->right;
                    rule = rule->right;
                }
            } else {
                // Found the item!
                left_internal_nodes.push_back(tree);
                left_leaf_nodes.push_back(tree->left);
                right_leaf_nodes.push_back(tree->right);
                all_done = true;
                break;
            }
        }
        left_internal_nodes.insert(left_internal_nodes.end(),
                                   right_internal_nodes.rbegin(),
                                   right_internal_nodes.rend());
        left_leaf_nodes.insert(left_leaf_nodes.end(),
                               right_leaf_nodes.rbegin(),
                               right_leaf_nodes.rend());
        typename std::vector<tree_node<T>*>::const_iterator a = left_internal_nodes.cbegin();
        typename std::vector<tree_node<T>*>::const_iterator b = left_leaf_nodes.cbegin();
        std::vector<bool>::const_iterator c = rule->result.cbegin();
        root = link<T>(a, b, c);
        //root = link<T>(left_internal_nodes.cbegin(), left_leaf_nodes.cbegin(), rule->result.cbegin());
        return std::make_pair(all_done, &rule->result);
    }
};

void step_left(std::vector<bool>::const_iterator& iter) {
    assert(!*iter);
    iter++;
}

void step_right(std::vector<bool>::const_iterator& iter) {
    assert(!*iter);
    iter++;
    int ones_less_zeros = 0;
    while (ones_less_zeros < 1) {
        if (*iter) {
            ones_less_zeros++;
        } else {
            ones_less_zeros--;
        }
        iter++;
    }
}

template <typename T>
struct td_searcher {

    // The item we're looking for.
    const T& item;

    // The rules we're using
    const ruleset * rules;

    tree_node<T> * center_tree,
                 * left,    // Where we're growing the left tree.
                 * right;   // Where we're growing the right tree.

    // Let "left" tree will hang off of the right of this, and the "right" tree
    // will hang off the left.
    tree_node<T> lr_dummy;

    td_searcher(const T& item, tree_node<T> * root, const ruleset& rules)
        : item(item),
          rules(&rules),
          center_tree(root),
          left(nullptr),
          right(nullptr),
          lr_dummy(item)
    {
        left = &lr_dummy;
        right = &lr_dummy;
    }

    // We hold pointers into our tree and our ruleset and traverse them in
    // lockstep till we find the item, fail to find it, or run out of rules.
    //
    // At that point we apply the rule we're at.
    //
    // Assumes a non-empty tree.
    //
    // Returns whether the search should end
    bool step() {
        // std::cout << "Current tree:\t";
        // print(std::cout);
        // std::cout << std::endl;

        auto res = tree_node<T>::apply_rule_at(center_tree, item, *rules);
        const std::vector<bool> * rule = res.second;
        bool all_done = res.first;

        // std::cout << "Post Xfm trees:\t";
        // print(std::cout);
        // std::cout << std::endl;

        // std::cout << "Used rule " << *rule << std::endl;

        auto left_tree_additions = extract_left(center_tree, *rule);
        if (left_tree_additions.first != nullptr) {
            left->right = left_tree_additions.first;
            left = left_tree_additions.second;
        }

        // std::cout << "Post Left Extraction trees:\t";
        // print(std::cout);
        // std::cout << std::endl;

        auto right_tree_additions = extract_right(center_tree, *rule);
        if (right_tree_additions.first != nullptr) {
            right->left = right_tree_additions.first;
            right = right_tree_additions.second;
        }

        return all_done;
    }

    void print(std::ostream& out) {
        tree_node<T>::print(out, lr_dummy.right);
        out << '\t';
        tree_node<T>::print(out, center_tree);
        out << '\t';
        tree_node<T>::print(out, lr_dummy.left);
    }

    tree_node<T> * reassemble () {
        left->right = center_tree->left;
        right->left = center_tree->right;
        center_tree->left = lr_dummy.right;
        center_tree->right = lr_dummy.left;
        return center_tree;
    }

    tree_node<T> * do_search() {
        while (!step());
        return reassemble();
    }

    static std::pair<tree_node<T> *, tree_node<T> *> extract_left(tree_node<T> * root, const std::vector<bool>& rule) {
        tree_node<T> * left_child = root->left;
        tree_node<T> * tree_iter = left_child;
        auto rule_iter = rule.begin();
        step_left(rule_iter);

        if (*rule_iter) {
            return std::make_pair(nullptr, nullptr);
        }

        // While we haven't reached a terminal node
        while (true) {
            // See if our right child is a terminal
            step_right(rule_iter);
            if (*rule_iter) {
                // We hit the terminator for the rule.
                root->left = tree_iter->right;
                tree_iter->right = nullptr;
                return std::make_pair(left_child, tree_iter);
            } else {
                tree_iter = tree_iter->right;
            }
        }
    }

    static std::pair<tree_node<T> *, tree_node<T> *> extract_right(tree_node<T> * root, const std::vector<bool>& rule) {
        tree_node<T> * right_child = root->right;
        tree_node<T> * tree_iter = right_child;
        auto rule_iter = rule.begin();
        step_right(rule_iter);

        if (*rule_iter) {
            return std::make_pair(nullptr, nullptr);
        }

        // While we haven't reached a terminal node
        while (true) {
            // See if our left child is a terminal
            step_left(rule_iter);
            if (*rule_iter) {
                root->right = tree_iter->left;
                tree_iter->left = nullptr;
                return std::make_pair(right_child, tree_iter);
            } else {
                tree_iter = tree_iter->left;
            }
        }
    }
};

// This takes in:
// * a list of internal nodes `internal_nodes[internal_node_idx:]`
// * a list of leaf nodes `leaf_nodes[leaf_node_idx:]`
// * a description of a tree structure with a cursor `result` with `result_idx`
//   in which a `false` corresponds to an internal node
//
// It then relinks all the nodes into the tree structure described, and returns
// a pointer to the root of the resultin construct.
//
// When it exits, the `internal_node_idx` has been advanced past the last node
// consumed as has the `leaf_node_idx`. The `result_idx` has been advanced past
// the description of the tree constructed.
template<typename T>
tree_node<T> * link(typename std::vector<tree_node<T>*>::const_iterator& internal_nodes,
                    typename std::vector<tree_node<T>*>::const_iterator& leaf_nodes,
                    std::vector<bool>::const_iterator& result) {
    if (*result) {
        // We're building just a leaf node
        result++;
        auto res = *leaf_nodes;
        leaf_nodes++;
        return res;
    } else {
        // We're building an internal node
        result++;
        tree_node<T> * left_child = link<T>(internal_nodes, leaf_nodes, result);
        tree_node<T> * parent = *internal_nodes;
        internal_nodes++;
        tree_node<T> * right_child = link<T>(internal_nodes, leaf_nodes, result);
        parent->left = left_child;
        parent->right = right_child;
        return parent;
    }
}

template <typename T>
struct tree_set {

    tree_node<T> * root;

    static const int INVERSION_DEPTH = 2;

    tree_set()
        : root(nullptr)
    {}

    ~tree_set() { tree_node<T>::deallocate(root); }

    void insert(const T& item)
    {
        insert(root, item);
    }

    static void insert(tree_node<T>*& n, const T& item)
    {
        if (n == nullptr)
            n = new tree_node<T>{item};
        else if (n->data > item)
            insert(n->left, item);
        else if (item > n->data)
            insert(n->right, item);
    }

    std::ostream& print(std::ostream& out) const
    {
        tree_node<T>::print(out, root);
        return out << std::endl;
    }

    void td_splay_gen(const T& item, const ruleset& rules) {
        if (root != nullptr) {
            td_searcher<T> searcher(item, root, rules);
            root = searcher.do_search();
        }
    }

    void bu_splay_gen(const T& item, const ruleset& rules, const ruleset& inverse_rules) {
        std::vector<tree_node<T> *> stack;
        tree_node<T> dummy_node(item);
        dummy_node.right = root;
        tree_node<T> * dummy_node_ptr = &dummy_node;
        stack.push_back(dummy_node_ptr);
        stack.push_back(root);
        while (stack.back() != nullptr) {
            if (stack.back()->data > item) {
                stack.push_back(stack.back()->left);
            } else if (item > stack.back()->data) {
                stack.push_back(stack.back()->right);
            } else {
                stack.push_back(nullptr); // Escape the loop with a dummy to pop off
            }
        }
        stack.pop_back();
        // Now we pull the bottom of the stack to the root
        while (stack.size() > 2) {
            // We first have to identify the correct rule, using the inverse tree
            const ruleset * current_rule = &inverse_rules;
            tree_node<T> * bottom_node = stack.back();
            stack.pop_back();
            tree_node<T> * current_node = bottom_node;
            // When we exit this loop the node to use as the root of the rule
            // should be `current_node` and the rule to use should be
            // `current_rule`.
            // During the loop `current_node` will be compated to `stack.back()`
            while (stack.size() > 1) {
                assert(stack.back()->left == current_node || stack.back()->right == current_node);
                if (stack.back()->left == current_node) {
                    if (current_rule->left != nullptr) {
                        current_node = stack.back();
                        stack.pop_back();
                        current_rule = current_rule->left;
                    } else break;
                } else {
                    if (current_rule->right != nullptr) {
                        current_node = stack.back();
                        stack.pop_back();
                        current_rule = current_rule->right;
                    } else break;
                }
            }
            // Now we apply the rule
            // we need a reference to the parent's point to `current_node`
            assert(stack.back()->left == current_node || stack.back()->right == current_node);
            if (stack.back()->left == current_node) {
                tree_node<T>::apply_rule_at(stack.back()->left, item, rules);
                assert(stack.back()->left == bottom_node);
            } else {
                tree_node<T>::apply_rule_at(stack.back()->right, item, rules);
                assert(stack.back()->right == bottom_node);
            }
            stack.push_back(bottom_node);
        }
        root = stack[0]->right;
    }

};

/* ## API
 * This function generates a full tree of height `height`, where `height=0`
 * means a single node.
 *
 * `height=2` would be:
 *
 *                     100
 *                   /     \
 *                 010     110
 *                /   \   /   \
 *               001 011 101 111
 *
 * where we present the value of each item in binary.
 *
 * ## Implementation
 *
 * We insert (without rotation) each item in the tree, one layer at a time from
 * top to bottom.
 *
 * We observe that if the top layer is `level=0` then each layer starts with `1
 * << (height - level)` and the items in each layer are an arithmetic sequence
 * with difference `1 << (height - level + 1)`.
 */
tree_set<usize> full(usize height)
{
    tree_set<usize> set;
    usize number_items = (1 << (height + 1)) - 1;
    for (usize level = 0; level <= height; level++) {
        usize increment = 1 << (height - level + 1);
        for (usize item = 1 << (height - level); item <= number_items; item += increment) {
            set.insert(item);
        }
    }
    return set;
}

usize n_full(usize height)
{
    return (1 << (height + 1)) - 1;
}


// In a complete tree with `n` nodes, how many are in the left subtree?
usize complete_left_count(usize n)
{
    usize l;
    for (l = 0; (1u << l) < n + 2; l++) /* Do nothing */;
    // The number of nodes in the complete subtrees.
    usize n_complete_sub = (l > 1) ? (1u << (l - 2)) - 1 : 0;
    usize n_bottom_level = n - 2*n_complete_sub - 1;
    return n_complete_sub + MIN(n_complete_sub + 1, n_bottom_level);
}

// Outputs the string/rule representation of a complete tree with `n` nodes.
void complete(std::ostream & stream, usize n)
{
    if (n == 0) {
        stream << 'X';
    } else {
        stream << 'O';
        usize n_left = complete_left_count(n);
        usize n_right = n - 1 - n_left;
        complete(stream, n_left);
        complete(stream, n_right);
    }
}

// Given a binary number `path` outputs the 0's a L's and the 1's as R's.
// Assumes the number has length, `length`
//
// Returns ths number of non-zero entries.
usize out_path(std::ostream & stream, usize path, usize length)
{
    usize cnt = 0;
    for (usize bit = length; bit > 0; bit--) {
        stream << (((path >> (bit - 1)) & 1) ? 'R' : 'L');
        cnt += ((path >> (bit - 1)) & 1);
    }
    return cnt;
}


// The set of rules that reach as deep as `h` and always produce complete
// children.
std::string complete_rules(usize h)
{
    std::ostringstream out;
    for (usize level = 1; level <= h; level++) {
        for (usize path = 0; path < (1u << level); path++) {
            usize n_less_than = out_path(out, path, level);
            usize n_greater_than = level - n_less_than;
            out << '\t';
            out << 'O';
            complete(out, n_less_than);
            complete(out, n_greater_than);
            out << std::endl;
        }
    }
    return out.str();
}

void test_complete_left_count() {
    assert(complete_left_count(1)  == 0);
    assert(complete_left_count(2)  == 1);
    assert(complete_left_count(3)  == 1);
    assert(complete_left_count(4)  == 2);
    assert(complete_left_count(5)  == 3);
    assert(complete_left_count(6)  == 3);
    assert(complete_left_count(7)  == 3);
    assert(complete_left_count(8)  == 4);
    assert(complete_left_count(9)  == 5);
    assert(complete_left_count(10) == 6);
    assert(complete_left_count(11) == 7);
    assert(complete_left_count(12) == 7);
    assert(complete_left_count(13) == 7);
    assert(complete_left_count(14) == 7);
}

void test_complete_rules() {
    std::string expected(
            "L\tOXOXX\n"
            "R\tOOXXX\n"
            "LL\tOXOOXXX\n"
            "LR\tOOXXOXX\n"
            "RL\tOOXXOXX\n"
            "RR\tOOOXXXX\n"
            );
    std::string actual = complete_rules(2);
    assert(actual == expected);

    std::string expected2(
            "L\tOXOXX\n"
            "R\tOOXXX\n"
            "LL\tOXOOXXX\n"
            "LR\tOOXXOXX\n"
            "RL\tOOXXOXX\n"
            "RR\tOOOXXXX\n"
            "LLL\tOXOOXXOXX\n"
            "LLR\tOOXXOOXXX\n"
            "LRL\tOOXXOOXXX\n"
            "LRR\tOOOXXXOXX\n"
            "RLL\tOOXXOOXXX\n"
            "RLR\tOOOXXXOXX\n"
            "RRL\tOOOXXXOXX\n"
            "RRR\tOOOXXOXXX\n"
            );
    std::string actual2 = complete_rules(3);
    assert(actual2 == expected2);
}

void test_result_step() {
    // The result is 0 011 011
    std::vector<bool> result{false, false, true, true, false, true, true};

    auto root1 = result.cbegin();
    auto left_c = result.cbegin();
    std::advance(left_c, 1);
    step_left(root1);
    assert(root1 == left_c);
    step_right(root1);
    assert(*root1);

    auto root2 = result.cbegin();
    auto right_c = result.cbegin();
    std::advance(right_c, 4);
    step_right(root2);
    assert(root2 == right_c);
    step_left(root2);
    assert(*root2);
}

void test_left_extraction() {
    auto tree = full(2);
    tree_node<usize> * orig_left = tree.root->left;
    assert(tree.root->data == 0b100);
    std::vector<bool> rule{0, 0, 1, 1, 1};

    auto output = td_searcher<usize>::extract_left(tree.root, rule);

    assert(tree.root->data == 0b100);
    assert(tree.root->left->data == 0b011);
    assert(output.first = orig_left);
    assert(output.second = orig_left);

    auto tree2 = full(2);
    tree_node<usize> * orig_left2 = tree2.root->left;
    tree_node<usize> * orig_lr2 = tree2.root->left->right;
    assert(tree2.root->data == 0b100);
    std::vector<bool> rule2{0, 0, 1, 0, 1, 1, 1};
    auto output2 = td_searcher<usize>::extract_left(tree2.root, rule2);
    assert(tree2.root->data == 0b100);
    assert(tree2.root->left == nullptr);
    assert(output2.first = orig_left2);
    assert(output2.second = orig_lr2);
}

void test_right_extraction() {
    auto tree = full(2);
    tree_node<usize> * orig_right = tree.root->right;
    assert(tree.root->data == 0b100);
    std::vector<bool> rule{0, 1, 0, 1, 1};

    auto output = td_searcher<usize>::extract_right(tree.root, rule);
    assert(tree.root->data == 0b100);
    assert(tree.root->right->data == 0b101);
    assert(output.first = orig_right);
    assert(output.second = orig_right);

    auto tree2 = full(2);
    tree_node<usize> * orig_right2 = tree2.root->right;
    tree_node<usize> * orig_rl2 = tree2.root->right->left;
    assert(tree2.root->data == 0b100);
    std::vector<bool> rule2{0, 1, 0, 0, 1, 1, 1};
    auto output2 = td_searcher<usize>::extract_right(tree2.root, rule2);
    assert(tree2.root->data == 0b100);
    assert(tree2.root->right == nullptr);
    assert(output2.first = orig_right2);
    assert(output2.second = orig_rl2);
}

void test_td_splay_gen_2() {
    /*
     *                     100
     *                   /     \
     *                 010     110
     *                /   \   /   \
     *               001 011 101 111
     */
    tree_set<usize> general = full(2);
    std::string normal_rules_str(
            "L  OXOXX   "
            "R  OOXXX   "
            "LL OXOXOXX "
            "RR OOOXXXX "
            "LR OOXXOXX "
            "RL OOXXOXX");
    ruleset normal_rules = make_ruleset(normal_rules_str);
    general.td_splay_gen(0b101, normal_rules);
    /*
     *                       101
     *                      /   \
     *                    100   110
     *                   /         \
     *                 010         111
     *                /   \
     *               001 011
     */
    assert(general.root->data == 0b101);
    assert(general.root->left->data == 0b100);
    assert(general.root->right->data == 0b110);
    assert(general.root->left->right == nullptr);
    assert(general.root->right->left == nullptr);
    assert(general.root->left->left->data == 0b010);
    assert(general.root->right->right->data == 0b111);

    general.td_splay_gen(0b011, normal_rules);
    /*
     *                    011
     *                   /   \
     *                010     100
     *               /           \
     *             001           101
     *                              \
     *                              110
     *                                 \
     *                                 111
     */
    assert(general.root->data == 0b011);
    assert(general.root->left->data == 0b010);
    assert(general.root->right->data == 0b100);
    assert(general.root->left->right == nullptr);
    assert(general.root->right->left == nullptr);
    assert(general.root->left->left->data == 0b001);
    assert(general.root->left->left->left == nullptr);
    assert(general.root->left->left->right == nullptr);
    assert(general.root->right->right->left == nullptr);
    assert(general.root->right->right->data == 0b101);
    assert(general.root->right->right->right->left == nullptr);
    assert(general.root->right->right->right->data == 0b110);
}

void test_bu_splay_gen_2() {
    /*
     *                     100
     *                   /     \
     *                 010     110
     *                /   \   /   \
     *               001 011 101 111
     */
    tree_set<usize> general = full(2);
    std::string normal_rules_str(
            "L  OXOXX   "
            "R  OOXXX   "
            "LL OXOXOXX "
            "RR OOOXXXX "
            "LR OOXXOXX "
            "RL OOXXOXX");
    ruleset normal_rules = make_ruleset(normal_rules_str);
    ruleset inverse_rules = make_inverted_ruleset(normal_rules_str);
    general.bu_splay_gen(0b101, normal_rules, inverse_rules);
    /*
     *                       101
     *                      /   \
     *                    100   110
     *                   /         \
     *                 010         111
     *                /   \
     *               001 011
     */
    assert(general.root->data == 0b101);
    assert(general.root->left->data == 0b100);
    assert(general.root->right->data == 0b110);
    assert(general.root->left->right == nullptr);
    assert(general.root->right->left == nullptr);
    assert(general.root->left->left->data == 0b010);
    assert(general.root->right->right->data == 0b111);

    general.bu_splay_gen(0b011, normal_rules, inverse_rules);
    /*
     *                    011
     *                   /   \
     *                010     101
     *               /       /   \
     *             001     100   110
     *                              \
     *                              111
     */
    assert(general.root->data == 0b011);
    assert(general.root->left->data == 0b010);
    assert(general.root->right->data == 0b101);
    assert(general.root->left->right == nullptr);
    assert(general.root->right->left->data == 0b100);
    assert(general.root->left->left->data == 0b001);
    assert(general.root->left->left->left == nullptr);
    assert(general.root->left->left->right == nullptr);
    assert(general.root->right->right->left == nullptr);
    assert(general.root->right->right->data == 0b110);
    assert(general.root->right->right->right->left == nullptr);
    assert(general.root->right->right->right->data == 0b111);
}

void test_td_splay_gen_8() {
    tree_set<usize> general = full(8);
    usize n_items = n_full(8);
    pcg32_random_t rng = {0, 0};
    std::string normal_rules_str(
            "L  OXOXX   "
            "R  OOXXX   "
            "LL OXOXOXX "
            "RR OOOXXXX "
            "LR OOXXOXX "
            "RL OOXXOXX");
    ruleset normal_rules = make_ruleset(normal_rules_str);
    for (usize i = 0; i < 1 << 8; ++i) {
        usize t = pcg32_random_r(&rng) % n_items + 1;
        general.td_splay_gen(t, normal_rules);
        assert(general.root->data == t);
    }
}

void test_td_splay_gen_8_weird_rules() {
    tree_set<usize> general = full(8);
    usize n_items = n_full(8);
    pcg32_random_t rng = {0, 0};
    std::string normal_rules_str(
            "L   OXOXX     "
            "R   OOXXX     "
            "LL  OXOXOXX   "
            "RR  OOOXXXX   "
            "LR  OOXXOXX   "
            "LLL OXOXOXOXX "
            "RRR OOOOXXXXX");
    ruleset normal_rules = make_ruleset(normal_rules_str);
    for (usize i = 0; i < 1 << 8; ++i) {
        usize t = pcg32_random_r(&rng) % n_items + 1;
        general.td_splay_gen(t, normal_rules);
        assert(general.root->data == t);
    }
}

void test_bu_splay_gen_8_weird_rules() {
    tree_set<usize> general = full(8);

    std::string normal_rules_str(
            "L   OXOXX     "
            "R   OOXXX     "
            "LL  OXOXOXX   "
            "RR  OOOXXXX   "
            "LR  OOXXOXX   "
            "LLL OXOXOXOXX "
            "RRR OOOOXXXXX");
    ruleset normal_rules = make_ruleset(normal_rules_str);
    ruleset inverse_rules = make_inverted_ruleset(normal_rules_str);
    usize t = 1;
    for (int i = 0; i < 1 << 8; ++i) {
        t = (t * 27) % ((1 << 9) - 1) + 1;
        general.bu_splay_gen(t, normal_rules, inverse_rules);
        assert(general.root->data == t);
    }
}

int error(char * pgmname) {
    std::cout << "Usage:\n\
    " << pgmname << " <tree_height> <lookups>\n\n\
        tree_height: The program with use a full tree with this height.\n\
                     A single node has height 0.\n\
        lookups: How many splays to do." << std::endl;
    return 2;
}

void test() {
    std::cerr << "Starting tests" << std::endl;
    test_result_step();
    test_left_extraction();
    test_right_extraction();
    test_td_splay_gen_2();
    test_td_splay_gen_8();
    test_td_splay_gen_8_weird_rules();
    test_bu_splay_gen_2();
    test_complete_left_count();
    test_complete_rules();
    std::cerr << "Ending tests" << std::endl;

}

void benchmark_td_complete_rules(usize r_height, usize n, usize times)
{
    tree_set<usize> general = full(n);
    usize n_items = n_full(n);

    std::string rulestr = complete_rules(r_height);
    ruleset normal_rules = make_ruleset(rulestr);
    pcg32_random_t rng = {0, 0};

    double start = omp_get_wtime();
    for (usize i = 0; i < times; ++i) {
        usize t = pcg32_random_r(&rng) % n_items + 1;
        general.td_splay_gen(t, normal_rules);
        assert(general.root->data == t);
    }
    double end = omp_get_wtime();
    double time_per_op = (end - start) / times;
    std::cout << "TD\t" << r_height << '\t' << n << '\t' << (end-start)
              << '\t' << time_per_op << std::endl;
}

void benchmark_bu_complete_rules(usize r_height, usize n, usize times)
{
    tree_set<usize> general = full(n);
    usize n_items = n_full(n);

    std::string rulestr = complete_rules(r_height);
    ruleset normal_rules = make_ruleset(rulestr);
    ruleset inverse_rules = make_inverted_ruleset(rulestr);
    pcg32_random_t rng = {0, 0};

    double start = omp_get_wtime();
    for (usize i = 0; i < times; ++i) {
        usize t = pcg32_random_r(&rng) % n_items + 1;
        general.bu_splay_gen(t, normal_rules, inverse_rules);
        assert(general.root->data == t);
    }
    double end = omp_get_wtime();
    double time_per_op = (end - start) / times;
    std::cout << "BU\t" << r_height << '\t' << n << '\t' << (end-start)
              << '\t' << time_per_op << std::endl;
}

void benchmark_original_td(usize n, usize times)
{
    tree_set<usize> general = full(n);
    usize n_items = n_full(n);

    std::string rulestr =
        "L  OXOXX "
        "R  OOXXX "
        "LL OXOXOXX "
        "LR OOXXOXX "
        "RL OOXXOXX "
        "RR OOOXXXX";

    ruleset normal_rules = make_ruleset(rulestr);
    pcg32_random_t rng = {0, 0};

    double start = omp_get_wtime();
    for (usize i = 0; i < times; ++i) {
        usize t = pcg32_random_r(&rng) % n_items + 1;
        general.td_splay_gen(t, normal_rules);
        assert(general.root->data == t);
    }
    double end = omp_get_wtime();
    double time_per_op = (end - start) / times;
    std::cout << "O-TD\t" << 2 << '\t' << n << '\t' << (end-start)
              << '\t' << time_per_op << std::endl;
}

void benchmark_original_bu(usize n, usize times)
{
    tree_set<usize> general = full(n);
    usize n_items = n_full(n);

    std::string rulestr =
        "L  OXOXX "
        "R  OOXXX "
        "LL OXOXOXX "
        "LR OOXXOXX "
        "RL OOXXOXX "
        "RR OOOXXXX";

    ruleset normal_rules = make_ruleset(rulestr);
    ruleset inverse_rules = make_inverted_ruleset(rulestr);
    pcg32_random_t rng = {0, 0};

    double start = omp_get_wtime();
    for (usize i = 0; i < times; ++i) {
        usize t = pcg32_random_r(&rng) % n_items + 1;
        general.bu_splay_gen(t, normal_rules, inverse_rules);
        assert(general.root->data == t);
    }
    double end = omp_get_wtime();
    double time_per_op = (end - start) / times;
    std::cout << "O-BU\t" << 2 << '\t' << n << '\t' << (end-start)
              << '\t' << time_per_op << std::endl;
}

void benchmarks() {
    int n = 14;
    usize t = 1 << 22;
    std::cout << "algorithm\trule height\tstarting tree height\t"
                 "total seconds\tseconds per splay" << std::endl;
    int n_benchmarks = 2 * (n+4 - 2 + 1);
    int i = 0;
    benchmark_original_td(n, t);
    for (int d = 2; d <= n+4; d++) {
        ++i;
        benchmark_td_complete_rules(d, n, t);
        std::cerr << "Benchmark " << i << '/' << n_benchmarks << std::endl;
    }
    benchmark_original_bu(n, t);
    for (int d = 2; d <= n+4; d++) {
        ++i;
        benchmark_bu_complete_rules(d, n, t);
        std::cerr << "Benchmark " << i << '/' << n_benchmarks << std::endl;
    }
}

int main() {
    test();
    benchmarks();
}

