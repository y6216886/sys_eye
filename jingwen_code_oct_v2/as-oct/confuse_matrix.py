from sklearn.metrics import confusion_matrix


def example():
    y_true = [2, 1, 0, 1, 2, 0]
    y_pred = [2, 0, 0, 1, 2, 1]

    C=confusion_matrix(y_true, y_pred)
    print(C, end='\n\n')


    y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    C2 = confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    print(C2)
#tn,    fp,     fn,     tp
def makeMatrixFromtxt(path):
    with open(path,"r") as f:
        while True:
            line = f.readline()
            templist = line.split()[-4:]
            confuseMatrix = [[templist[0],templist[1]],[templist[2],templist[3]]]
            if not line:
                break
            print(confuseMatrix)

if __name__ == '__main__':
    path = "I:/log.txt"
    makeMatrixFromtxt(path)
    # example()