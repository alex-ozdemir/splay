#include <algorithm>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <utility>
#include <vector>

#define usize std::size_t

enum Direction {
    CCW, CW, NONE
};

template <typename T>
struct tree_node {
    T data;
    T deepest;
    tree_node<T> * left;
    tree_node<T> * right;
    usize depth;

    tree_node(T data)
        : data(data),
          deepest(data),
          left(nullptr),
          right(nullptr),
          depth(0)
    {}

    static void deallocate(tree_node<T> * n)
    {
        if (n != nullptr) {
            deallocate(n->left);
            deallocate(n->right);
            delete n;
        }
    }

    static void fix_all_deepest_tracking(tree_node<T>* n)
    {
        if (n != nullptr) {
            fix_all_deepest_tracking(n->left);
            fix_all_deepest_tracking(n->right);
            n->fix_deepest_tracking();
        }
    }

    void fix_deepest_tracking()
    {
        deepest = data;
        depth = 0;
        if (left != nullptr) {
            depth = left->depth + 1;
            deepest = left->deepest;
        }
        if (right != nullptr && (right->depth + 1) > depth) {
            depth = right->depth + 1;
            deepest = right->deepest;
        }
    }


    static std::pair<bool,usize> move_to_root(tree_node<T>*& n, const T& item, int depth)
    {
        if (n == nullptr) // If we passed a leaf
            return std::make_pair(true, 0);
        if (depth == 0) // If we hit search depth, return whether we found item
            return std::make_pair(!( (item < n->data) || (n->data < item) ), 0);
        if (item < n->data) { // We should look to the left for item
            std::pair<bool,usize> result = move_to_root(n->left, item, depth - 1);
            rotate_right(n);
            result.second++;
            return result;
        }
        else if (n->data < item) { // We should look right for item
            std::pair<bool,usize> result = move_to_root(n->right, item, depth - 1);
            rotate_left(n);
            result.second++;
            return result;
        }
        else // We found the item!
            return std::make_pair(true, 0);
    }

    static std::pair<bool,usize> old_splay(tree_node<T>*& n, const T& item, Direction dir)
    {
//        std::cout << "Splaying at Node: ";
//        print(std::cout, n);
//        std::cout << " for item: " << item << " with dir " << dir << std::endl;
        // If we searched past the leaves
        if (n == nullptr)
            return std::make_pair(false, 0);

        Direction our_dir;
        std::pair<bool, usize> result;
        if (n->data > item) {
            our_dir = CW;
            result = old_splay(n->left, item, our_dir);
        }
        else if (item > n->data) {
            our_dir = CCW;
            result = old_splay(n->right, item, our_dir);
        }
        else
            return std::make_pair(false, 0);

//        std::cout << "Rotate child? " << result.first << std::endl;
//        std::cout << "our dir " << our_dir << " original " << dir << std::endl;

        if (result.first) {
            rotate(n, our_dir);
            rotate(n, our_dir);
            return std::make_pair(false, result.second + 2);
        }
        else {
            if (our_dir == dir)
                return std::make_pair(true, result.second);
            else {
                rotate(n, our_dir);
                return std::make_pair(false, result.second + 1);
            }
        }
    }

    static void rotate(tree_node<T>*& n, Direction dir)
    {
//        if (dir != NONE) {
//            print(std::cout, n);
//        }
        switch (dir) {
        case CCW:
            rotate_left(n);
            break;
        case CW:
            rotate_right(n);
            break;
        case NONE:
            break;
        }
//        if (dir != NONE) {
//            print(std::cout, n);
//            usize i;
//            std::cin >> i;
//        }
    }

    static void rotate_left(tree_node<T> * & node)
    {
        if (node == nullptr || node->right == nullptr) return;
        tree_node<T> * old_right = node->right;
        node->right = old_right->left;
        old_right->left = node;
        node = old_right;
        node->left->fix_deepest_tracking();
        node->fix_deepest_tracking();
    }

    static void rotate_right(tree_node<T> * & node)
    {
        if (node == nullptr || node->left == nullptr) return;
        tree_node<T> * old_left = node->left;
        node->left = old_left->right;
        old_left->right = node;
        node = old_left;
        node->right->fix_deepest_tracking();
        node->fix_deepest_tracking();
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
};

template <typename T>
struct tree_set {

    tree_node<T> * root;

    static const int INVERSION_DEPTH = 2;

    tree_set()
        : root(nullptr)
    {}

    ~tree_set() { tree_node<T>::deallocate(root); }

    void fix_all_deepest_tracking()
    {
        tree_node<T>::fix_all_deepest_tracking(root);
    }

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

    // Undefined behavior if empty
    T deepest() const
    {
        return root->deepest;
    }

    usize old_splay(const T& item)
    {
        return tree_node<T>::old_splay(root, item, NONE).second;
    }

    usize new_splay(const T& item)
    {
        usize rotations = 0;
        // Try to move item to root untill we find it or we hit the leaves
        while (true) {
//            std::cout << "Looking for "  << item << " in ";
//            print(std::cout);
//            usize i;
//            std::cin >> i;
            auto result = tree_node<T>::move_to_root(root, item, INVERSION_DEPTH);
//            std::cout << "looking for " << item << std::endl;
            rotations += result.second;
            if (result.first) {
//                std::cout << "Looking for "  << item << " in ";
//                print(std::cout);
//                std::cout << " FOUND " << item << std::endl;
                return rotations;
            }
        }
    }

    std::ostream& print(std::ostream& out) const
    {
        tree_node<T>::print(out, root);
        return out << std::endl;
    }
};

/* ## API
 * This function generates a complete tree of height `height`, where `height=0`
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

tree_set<usize> complete(usize height)
{
    tree_set<usize> set;
    usize number_items = (1 << (height + 1)) - 1;
    for (usize level = 0; level <= height; level++) {
        usize increment = 1 << (height - level + 1);
        for (usize item = 1 << (height - level); item <= number_items; item += increment) {
            set.insert(item);
        }
    }
    set.fix_all_deepest_tracking();
    return set;
}

int error(char * pgmname) {
    std::cout << "Usage:\n\
    " << pgmname << " <tree_height> <lookups>\n\n\
        tree_height: The program with use a complete tree with this height.\n\
                     A single node has height 0.\n\
        lookups: How many splays to do." << std::endl;
    return 2;
}

int main(int argc, char * argv[]) {
    if (argc != 3) return error(argv[0]);
    usize height = atoi(argv[1]);
    usize lookups = atoi(argv[2]);
    if (lookups == 0) lookups = 3 * ((1 << (height + 1)) - 1);
    if (height == 0 || lookups == 0) return error(argv[0]);

    tree_set<usize> old_tree = complete(height);
    usize old_rotations = 0;
    for (usize i = 0; i < lookups; i++) {
//        std::cout << "Deepest: " << old_tree.deepest() << std::endl;
//        std::cout << "Dump: ";
//        old_tree.print(std::cout);
        usize rotations = old_tree.old_splay(old_tree.deepest());
        old_rotations += rotations;
//        std::cout << "\nRotations: " << rotations << '/' << old_rotations << std::endl;
    }
//        std::cout << "Deepest: " << old_tree.deepest() << std::endl;
//        std::cout << "Dump: ";
//        old_tree.print(std::cout);
//        std::cout << "DONE" << std::endl;


    tree_set<usize> new_tree = complete(height);
    usize new_rotations = 0;
    for (usize i = 0; i < lookups; i++) {
//        std::cout << "Deepest: " << new_tree.deepest() << std::endl;
//        std::cout << "Dump: ";
//        new_tree.print(std::cout);
        usize rotations = new_tree.new_splay(new_tree.deepest());
        new_rotations += rotations;
//        std::cout << "Rotations: " << rotations << '/' << new_rotations << std::endl;
    }
    std::cout << height << '\t' << lookups << '\t'
              << old_rotations << '\t' << new_rotations << std::endl;
}
