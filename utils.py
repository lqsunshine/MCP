import torch
import numpy as np
def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH

def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1]  # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH

def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind.cpu()]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


class DCG(object):

    def __init__(self, k=10, gain_type='exp2'):
        """
        :param k: int DCG@k
        :param gain_type: 'exp2' or 'identity'
        """
        self.k = k
        self.discount = self._make_discount(256)
        if gain_type in ['exp2', 'identity']:
            self.gain_type = gain_type
        else:
            raise ValueError('gain type not equal to exp2 or identity')

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        gain = self._get_gain(targets)
        discount = self._get_discount(min(self.k, len(gain)))
        return np.sum(np.divide(gain, discount))

    def _get_target(self, qu_B, re_B, qu_L, re_L, topk):

        num_query = qu_L.shape[0]
        topkmap = 0

        for iter in range(num_query):
            gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
            # gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
            tsum = np.sum(gnd)
            if tsum == 0:
                continue
            hamm = calculate_hamming(qu_B[iter, :], re_B)
            ind = np.argsort(hamm)
            gnd = gnd[ind]

            gain = self._get_gain(gnd)
            discount = self._get_discount(min(self.k, len(gain)))
            topkmap_ = np.sum(np.divide(gain, discount))
            topkmap = topkmap + topkmap_
        return topkmap

    def _get_gain(self, targets):
        t = targets[:self.k]
        if self.gain_type == 'exp2':
            return np.power(2.0, t) - 1.0
        else:
            return t

    def _get_discount(self, k):
        if k > len(self.discount):
            self.discount = self._make_discount(2 * len(self.discount))
        return self.discount[:k]

    @staticmethod
    def _make_discount(n):
        x = np.arange(1, n + 1, 1)
        discount = np.log2(x + 1)
        return discount


class NDCG(DCG):

    def __init__(self, k=10, gain_type='exp2'):
        """
        :param k: int NDCG@k
        :param gain_type: 'exp2' or 'identity'
        """
        super(NDCG, self).__init__(k, gain_type)

    def get_target(self, qu_B, re_B, qu_L, re_L):
        qu_B,re_B,qu_L,re_L = qu_B.cpu().numpy(),re_B.cpu().numpy(),qu_L.cpu().numpy(),re_L.cpu().numpy()

        num_query = qu_L.shape[0]
        ndcg = 0

        for iter in range(num_query):
            gnd = (np.dot(qu_L[iter, :], re_L.transpose())).astype(np.float32)
            # gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
            tsum = np.sum(gnd)
            if tsum == 0:
                continue
            hamm = calculate_hamming(qu_B[iter, :], re_B)
            ind = np.argsort(hamm)
            gnd = gnd[ind]

            dcg = super(NDCG, self).evaluate(gnd)
            ideal = np.sort(gnd)[::-1]
            idcg = super(NDCG, self).evaluate(ideal)
            ndcg_ = dcg / idcg
            # gain = self._get_gain(gnd)
            # discount = self._get_discount(min(self.k, len(gain)))
            # topkmap_ = np.sum(np.divide(gain, discount))
            ndcg = ndcg + ndcg_
        ndcg = ndcg / num_query

        return ndcg

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        dcg = super(NDCG, self).evaluate(targets)
        ideal = np.sort(targets)[::-1]
        idcg = super(NDCG, self).evaluate(ideal)
        return dcg / idcg

    def maxDCG(self, targets):
        """
        :param targets: ranked list with relevance
        :return:
        """
        ideal = np.sort(targets)[::-1]
        return super(NDCG, self).evaluate(ideal)




if __name__ == '__main__':
    qB = torch.Tensor([[1, -1, 1, 1],
                       [-1, -1, -1, 1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1]])
    rB = torch.Tensor([[1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [1, 1, -1, -1],
                       [-1, 1, -1, -1],
                       [1, 1, -1, 1]])
    query_L = torch.Tensor([[0, 1, 0, 0],
                            [1, 1, 0, 0],
                            [1, 0, 0, 1],
                            [0, 1, 0, 1]])
    retrieval_L = torch.Tensor([[1, 0, 0, 1],
                                [1, 1, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 1, 0]])

    ndcg10 = NDCG(10)
    ndcg = ndcg10.get_target(qB,rB,query_L,retrieval_L)
    print(ndcg)

