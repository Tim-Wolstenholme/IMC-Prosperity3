max_returns = 1
max_combo = []
convertion_table = [[1, 1.45, 0.52, 0.72], [0.7, 1, 0.31, 0.48], [1.95, 3.1, 1, 1.49], [1.34, 1.98, 0.64, 1]]
for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                returns = (1 * convertion_table[3][i] * convertion_table[i][j] * convertion_table[j][k] *
                           convertion_table[k][l] * convertion_table[l][3])
                if returns > max_returns:
                    max_returns = returns
                    max_combo = [i,j,k,l]

print(max_returns,max_combo)
