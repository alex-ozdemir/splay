import numpy as np
import matplotlib.pyplot as plt
import sys

TD = 0
BU = 1

def algorithm_str_to_code(alg_str):
    return { b'TD': TD, b'BU': BU }[alg_str]

def main():
    data_filename = sys.argv[1]
    img_filename = sys.argv[2]
    data = np.loadtxt(data_filename,
                      delimiter='\t',
                      skiprows=1,
                      converters={0: algorithm_str_to_code})
    rule_heights = [rule_h for (alg, rule_h, _, _, _) in data if alg == TD]
    td_avg_times = [avg_sec*10**6 for (alg, _, t_, _, avg_sec) in data if alg == TD]
    bu_avg_times = [avg_sec*10**6 for (alg, _, t_, _, avg_sec) in data if alg == BU]
    space = np.linspace(1, 27, 100)

    td_plot, = plt.plot(rule_heights, td_avg_times, marker='d', color='g')
    bu_plot, = plt.plot(rule_heights, bu_avg_times, marker='o', color='b')

    plt.title('Generalized Splay Time')
    plt.xlabel('Rule height')
    plt.ylabel(u'Average Time per Lookup (\u03BCs)')
    plt.legend([td_plot, bu_plot], ['Top-Down', 'Bottom-Up'])
    plt.savefig(img_filename)
    plt.show()


if __name__ == '__main__':
    main()
