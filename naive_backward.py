import numpy as np
def naive_backward(gO, A, B, X):
    # gO [b x n]
    # A [m x r]
    # B [r x n]
    # X [b x m]
    # forward X @ A @ B

    # aim to reduce the load of X and gO

    gA = X.T @ (gO @ B.T)
    gB = (A.T @ X.T) @ gO

    gX = gO @ B.T @ A.T
    return gX, gA, gB


def lora_backward_gA_gB(gO, A, B, X, block_M, block_N, block_K):
    # gO [b x n]
    # A [m x r]
    # B [r x n]
    # X [b x m]

    # AT [r x m]
    # BT [n x r]
    # XT [m x b] 

    # each kernel compute X.T gO block
    # it multiplies by B.T to get gA block
    # it multiplies by A.T to get gB block

    AT = A.T
    BT = B.T
    XT = X.T

    M, R = A.shape
    N = B.shape[1]
    K = X.shape[0]

    gA = np.zeros((M, R), dtype=np.float32)
    gB = np.zeros((R, N), dtype=np.float32)

    for ii in range(0, M, block_M):
        for jj in range(0, N, block_N):
            lora_backward_gA_gB_kernel(gO, AT, BT, XT, ii, jj, block_M, block_N, block_K, K, gA, gB)

    return gA, gB

def lora_backward_gA_gB_kernel(gO, AT, BT, XT, ii, jj, block_M, block_N, block_K, K, gA, gB):
    # each kernel compute X.T gO block
    # it multiplies by B.T to get gA block
    # it multiplies by A.T to get gB block

    # gO [b x n]
    # AT [r x m]
    # BT [n x r]
    # XT [m x b] 

    acc = np.zeros((block_M, block_N), dtype=np.float32)
    for k in range(K // block_K):
        # XT_block = XT[k*block_K:k*block_K+block_K, ii:ii+block_M]
        XT_block = XT[ ii:ii+block_M, k*block_K:k*block_K+block_K]
        gO_block = gO[k*block_K:k*block_K+block_K, jj:jj+block_N]

        acc += XT_block @ gO_block

    BT_block = BT[jj:jj+block_N, :]
    gA[ii:ii+block_M, :] += acc @ BT_block

    AT_block = AT[:, ii:ii+block_M]
    gB[:, jj:jj+block_N] += AT_block @ acc 


def test_backward():
    np.random.seed(0)
    b, m, r, n = 128, 64, 16, 256
    block_M, block_N, block_K = 8, 8, 8

    gO = np.random.randn(b, n).astype(np.float32)
    A = np.random.randn(m, r).astype(np.float32)
    B = np.random.randn(r, n).astype(np.float32)
    X = np.random.randn(b, m).astype(np.float32)

    gX_naive, gA_naive, gB_naive = naive_backward(gO, A, B, X)
    gA, gB = lora_backward_gA_gB(gO, A, B, X, block_M, block_N, block_K)

    # print(np.allclose(gX, gX_naive))
    # print precision 0.01
    np.set_printoptions(precision=2)
    print(gA)
    print(gA_naive)
    assert np.allclose(gA, gA_naive, atol=1e-1)
    assert np.allclose(gB, gB_naive, atol=1e-1)


if __name__ == "__main__":
    test_backward()


