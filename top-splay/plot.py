import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
    filename = sys.argv[1]
    img_filename = sys.argv[2]
    data = np.loadtxt(filename, delimiter='\t')
    tree_height = data[:,0]
    lookups = data[:,1].astype(float)
    tree_size = 2 ** (tree_height + 1) - 1
    log_size = np.log2(tree_size)
    old_rotations = data[:,2].astype(float)
    new_rotations = data[:,3].astype(float)
    old_ratio = old_rotations / lookups
    new_ratio = new_rotations / lookups
    space = np.linspace(1, 27, 100)

    old_plot, = plt.plot(log_size, old_ratio, marker='d', color='g')
    N = 5
    print('Ignoring lowest %d sizes' % N)
    old_fit = np.poly1d(np.polyfit(log_size[N:], old_ratio[N:], 2))
    plt.plot(space, old_fit(space), color='c')
    print('Fit for original strategy')
    print(old_fit)

    new_plot, = plt.plot(log_size, new_ratio, marker='o', color='b')
    new_fit = np.poly1d(np.polyfit(log_size[N:], new_ratio[N:], 2))
    plt.plot(space, new_fit(space), color='r')
    print('Fit for new strategy')
    print(new_fit)

    plt.title('Rotations per lookup')
    plt.xlabel('$\log_2$(size)')
    plt.ylabel('Rotations per lookup, average')
    plt.legend([old_plot, new_plot], ['Bottom Up Splay', 'Top Splay'])

    print(new_ratio / old_ratio)

    plt.savefig(img_filename)

    plt.show()


if __name__ == '__main__':
    main()
