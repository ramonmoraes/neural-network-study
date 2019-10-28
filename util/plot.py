import matplotlib.pyplot as plt

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

