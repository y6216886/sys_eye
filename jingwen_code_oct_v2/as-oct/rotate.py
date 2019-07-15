def rotate(matrix):
    """
    :type matrix: List[List[int]]
    :rtype: None Do not return anything, modify matrix in-place instead.
    """
    n = len(matrix[0])
    l1 =  []
    l2 =  []
    l3 =  []
    l4 =  []
    rotatedMatrix = [[0]*n]*n
    for i in range (int(n/2)):
        for j in range (i, n-i):
            l1.append (matrix[i][j])
            l2.append (matrix[j][n - i - 1])
            l3.append (matrix[n-i-1][j])
            l4.append (matrix[j][i])
        for j in range (i, n - i):
            print ("i,j",i, j)
            print(l4)
            l4=l4[::-1]
            print(l4)
            rotatedMatrix[i][j] = l4[j]
            rotatedMatrix[j][n - i - 1] = l1[j]
            l2 = l2[::-1]
            rotatedMatrix[n-i-1][j] = l2[j]
            rotatedMatrix[j][i] = l3[j]
        # print (l1, l2, l3, l4)
        l1 = []
        l2 = []
        l3 = []
        l4 = []
    # print(rotatedMatrix)

if __name__ == '__main__':
    matrix = [
  [ 5, 1, 9,11, 76, 111],
  [ 2, 4, 8,10, 65,333],
  [13, 3, 6, 7, 34,113],
  [15,14,12,16, 12,656],
  [75,112,34,53,66,7654],
  [123,444,435,246,432,565]

]
    rotate(matrix)