import pandas as pd
import numpy as np
from scipy.integrate import trapz

def tristimulus(I_color, cmf, step):

    # 求出XYZ三色刺激值的系数，100/(I与y的积分)，y为颜色匹配函数中的一支
    Coefficient = 100/trapz(np.multiply(I_color, (cmf['y'].to_numpy())[::step]), dx=step)
    # XYZ三色刺激值的分子与系数的乘积，注意I的首尾元素已经在前面乘以0.5
    X0 = np.multiply(I_color, (cmf['x'].to_numpy())[::step]) * Coefficient
    Y0 = np.multiply(I_color, (cmf['y'].to_numpy())[::step]) * Coefficient
    Z0 = np.multiply(I_color, (cmf['z'].to_numpy())[::step]) * Coefficient

    return [X0, Y0, Z0]