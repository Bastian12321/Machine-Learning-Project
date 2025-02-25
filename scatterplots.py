from load import *
import matplotlib.pyplot as plt

#method for iterating over attributes in dataset 
# and drawing pairwise scatterplots
def make_scatterplots(start, end):
    plt.figure(figsize=(12, 10))
    for m1 in range(start, end):
        for m2 in range(start, end):
            plt.subplot(end - start, end - start, (m1 - start) * (end - start) + (m2 - start) + 1)
            for c in range(C):
                class_mask = classLabels == c
                plt.plot(np.array(X[class_mask, m2]), np.array(X[class_mask, m1]), ".")
                if m1 == (end - 1):
                    plt.xlabel(attributeNames[m2])
                else:
                    plt.xticks([])
                if m2 == start:
                    plt.ylabel(attributeNames[m1])
                else:
                    plt.yticks([])          
    plt.legend(classNames)
    plt.show()

make_scatterplots(0, 4)
make_scatterplots(4, M)
