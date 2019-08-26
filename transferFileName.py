# /home/yangyifan/jpg/20170417.1727702684-84-1_1.jpg
def transferTxtPath(path,path1):
    f = open(path)
    g = open(path1,'a')
    for line in f:
        line_trans = line.split('/')[-1]
        g.write("/home/yangyifan/data/asoct/jpg/"+line_trans)

        # print(line)

def retrain():
    str1 = '/home/yangyifan/code/jingwen_code_oct_v2/as-oct/log_asoct_VGG_11_onevsall_bs8_type=2-lr=0.001/model/checkpoint'
    retrain = []  # path to model to retrain with
    for i in range (33):
        model = str1 + str(i) + '.pkl'
        retrain.append (model)
    print(retrain)
if __name__ == "__main__":
    # path = 'as-oct/test.txt'
    # path1 = 'G:/CODE/jingwen_code_oct_v2/testAfter.txt'
    # path = "as-oct/train.txt"
    # path1 = 'I:/trainAfter.txt'
    # transferTxtPath(path,path1)
    retrain()