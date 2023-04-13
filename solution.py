import pandas as pd
import numpy as np
from hyppo.ksample import Energy, MMD, DISCO
from scipy.stats import laplace, norm, ks_2samp, anderson_ksamp, cramervonmises_2samp
from statsmodels.stats.weightstats import ztest
from statsmodels.distributions.empirical_distribution import ECDF


chat_id = 382319199 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    answers = []
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    if MMD(compute_kernel="poly").test(x, y)[1] < 0.01:
        answers.append(True)
    else:
        answers.append(False)
        
    if MMD(compute_kernel="rbf", gamma=1/10).test(x, y)[1] < 0.01:
        answers.append(True)
    else:
        answers.append(False)
        
    if MMD(compute_kernel="rbf", gamma=1).test(x, y)[1] < 0.01:
        answers.append(True)
    else:
        answers.append(False)
        
    if MMD(compute_kernel="rbf", gamma=10).test(x, y)[1] < 0.01:
        answers.append(True)
    else:
        answers.append(False)
    
    if MMD(compute_kernel="laplacian", gamma=1/10).test(x, y)[1] < 0.01:
        answers.append(True)
    else:
        answers.append(False)
    
    if MMD(compute_kernel="laplacian", gamma=1).test(x, y)[1] < 0.01:
        answers.append(True)
    else:
        answers.append(False)
    
    if MMD(compute_kernel="laplacian", gamma=10).test(x, y)[1] < 0.01:
        answers.append(True)
    else:
        answers.append(False)
    
    if answers.count(True) > 3:
        return True
    else:
        return False
