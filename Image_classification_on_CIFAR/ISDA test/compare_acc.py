import numpy as np
import matplotlib.pyplot as plt

NumCluster = 2
AccEachCluster = np.zeros((NumCluster, 160))
for i in range(NumCluster):
    max_num = 0
    filename = 'cifar10'\
        + '-32_/' \
        + 'no_1_lambda_0_0.' + ("0" if i > 4 else "5") + '_cluster_'\
        + str(0)\
        + ("" if i > 0 else "_noise")\
        + "/accuracy_epoch_test.txt"
    f = open(filename, "r")
    for j, line in enumerate(f):
        if max_num < float(line):
            max_num = float(line)
        AccEachCluster[i][j] = (float(line))
    print(max_num)
    f.close()

for i in range(NumCluster):
    if i == 0:
        label_name = "ISDA"
    else:
        label_name = 'noise'
    plt.plot(AccEachCluster[i], label=label_name)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
