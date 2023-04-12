import pandas as pd
import numpy as np
from hyppo.ksample import MMD


chat_id = 382319199 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    if MMD(compute_kernel="poly").test(x, y)[1] < 0.01:
        answer = True
    else:
        answer = False
    return answer
