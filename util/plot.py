import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatter_2d(dataset):
    def scatter_value_color(value, color):
        x = []
        y = []
        for (inpt, output) in dataset:
            if output == value:
                x.append(inpt[0])
                y.append(inpt[1])
        plt.scatter(x,y, color=color)
    
    scatter_value_color(1, "b")
    scatter_value_color(0, "r")
    plt.show()


def scatter_3d(dataset):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def scatter_value_color(value, color):
        x = []
        y = []
        z = []
        for (inpt, output) in dataset:
            if output == value:
                x.append(inpt[0])
                y.append(inpt[1])
                z.append(inpt[2])
        import pdb; pdb.set_trace()
        ax.scatter3D(x,y,z, color=color)
    
    scatter_value_color(1, "b")
    scatter_value_color(0, "r")
    plt.show()


