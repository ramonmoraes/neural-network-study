import csv


class Dataset:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __iter__(self):
        return zip(self.inputs, self.outputs)

    def __getitem__(self, key):
        return (self.inputs[key], self.outputs[key])


OR_DATASET = Dataset([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 1])

AND_DATASET = Dataset([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1])

XOR_DATASET = Dataset([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0])

FIVE_OR_MORE_DATASET = Dataset(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    [0, 0, 0, 0, 0, 1, 1, 1],
)

XOR_DATASET = Dataset([[0, 0], [0, 1], [1, 0], [1, 1]], [1, 0, 0, 1])


class MnistDataset(Dataset):
    file_path = "dataset/mnist_train.csv"

    def __init__(self):
        super().__init__([], [])

    def load(self):
        print("[Loading MnistDataset...]")
        with open(self.file_path, "r") as f:
            csv_reader = csv.reader(f, delimiter=",")
            read_header = False
            for row in csv_reader:
                if not read_header:
                    read_header = True
                    continue

            self.inputs.append(self.format_input(row[1:]))
            self.outputs.append(self.numeral_to_output(row[0]))
        print("[Loaded]")

    def numeral_to_output(self, numeral: str):
        number = int(numeral)
        inpt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        inpt[number + 1] = 1
        return inpt

    def format_input(self, output):
        return [int(x)/255 for x in output]