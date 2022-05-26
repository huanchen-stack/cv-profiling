import csv
import torch

f = open("csvtrial.csv", "w")
if __name__ == "__main__":
    f.write("0, 1, 2, 3, 4, 5, 6, 7, 8, 9\n")
    f.write("a, b, c, d, e, f, g, h, i, j\n")
    tmp = torch.rand(1, 3, 224, 224)
    f.write(str(tmp.shape))
f.close()