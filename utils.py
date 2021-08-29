import numpy as np
import time
import os
import torch
np.random.seed(43)
from multiprocessing import Pool


def get_n_params(model):
    """
    get the number of total model's parameter
    :param model: the target model, e.g GCN, HGT
    :return: the number
    """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def calculate_pathsim(m):
    """
    calculate PathSim by definition
    :param m:
    :return:
    """
    m_diag = m.diagonal()
    m_xx = np.array([list(m_diag), ] * m.shape[0])
    m_xx = m_xx + m_xx.transpose()
    m_pathsim = np.nan_to_num(2 * m * 1/m_xx)
    return m_pathsim


# def compute_ndcg(xs, ys):
#     """
#     compute the average ndcg
#     :param xs:
#     :param ys:
#     :return:
#     """
#     l_ndcg = []
#     xs = xs.tolist()
#     ys = ys.tolist()
#     for i in range(len(ys)):
#         l_ndcg.append(NDCG(xs[i], ys[i]))
#     return np.average(l_ndcg)

def compute_ndcg(xs, ys):
    """
    compute the average ndcg
    :param xs:
    :param ys:
    :return:
    """
    xs = xs.tolist()
    ys = ys.tolist()
    pool = Pool()
    xs_ys = list(zip(xs, ys))
    l_ndcg = pool.starmap(NDCG, xs_ys)
    pool.close()
    return np.average(l_ndcg)


def NDCG(gt, pred, use_graded_scores=False):
  score = 0.0
  for rank, item in enumerate(pred):
    if item in gt:
      if use_graded_scores:
        grade = 1.0 / (gt.index(item) + 1)
      else:
        grade = 1.0
      score += grade / np.log2(rank + 2)

  norm = 0.0
  for rank in range(len(gt)):
    if use_graded_scores:
      grade = 1.0 / (rank + 1)
    else:
      grade = 1.0
    norm += grade / np.log2(rank + 2)
  return score / max(0.3, norm)
