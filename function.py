import numpy as np
from scipy.integrate import trapz
from scipy.stats import norm
from math import *

wavelength = np.arange(360, 761)

def func_cool(x, I, step=1):
    if type(x) == np.ndarray:
        # r = dr(x, I,  step=step)  # 这里传入的x数组，其最后一个元素代表了吸收截止波长，dr(x)返回的数组只含有反射率，且为已经增量的反射率
        r = x  # 这里的x只有反射率，无截止波长
    else:
        x = np.asarray(x)
        # r = dr(x, I,  step=step)
        r = x  # 这里的x只有反射率，无截止波长
    fun = trapz(np.multiply(-I, r), dx=step)
    return fun

def func_hot(x, I):
    # r = dr(x)  # 这里传入的x数组，其最后一个元素代表了吸收截止波长，dr(x)返回的数组只含有反射率，且为已经增量的反射率
    r = x  # 这里的x只有反射率，无截止波长
    fun = np.sum(I*r)
    return fun

def dr(x, I, step=1):
    r = x[:-1]  # 除了最后一个元素是截止波长，之前的都是反射率
    l = x[-1]  # 获得截止波长
    # extr转换系数，0.25是折射率为1.5的情况下表面为平面的光提取率，
    # QY是量子产率(Quantum Yield,QY)或可以说是内量子效率(Internal Quantum Efficiency, IQE)
    # SS((360+l)/(760+l))是斯托克斯位移，假设吸收峰发射峰不相交，无自吸收效应
    extr = 0.25
    QY = 1
    SS = (360+l)/(760+l)
    l_index = floor((l - 360)/step) + 1  # 截止波长数索引，先向上取整，后减去360是映射过程
    # 初始化PL过程的波长的索引序列
    wavelength_step = wavelength[::step]
    x_conversion = wavelength_step[l_index:]
    # 吸收并转换的能量，将要以该能量进行正态分布
    P_conversion = trapz(np.multiply(I[:l_index], (1 - r[:l_index])), dx=step) * QY * SS * extr
    miu = (l + 760) / 2  # 正态分布的均值
    sigma = (760 - l) / 4  # 正态分布的标准差
    GPD = P_conversion * norm.pdf(x_conversion, miu, sigma)  # 初始化高斯概率密度函数
    delta_r = np.divide(GPD, I[l_index:])  # 将高斯分布的光谱与AM1.5对应波段光谱相除，得到反射率增量
    try:
        delta_r[-1] = delta_r[-1]
    except IndexError:  # 此时l,即吸收截止波长已经到了760nm这个临界值
        pass
    r = np.append(r[:l_index], np.add(r[l_index:], delta_r))  # 将吸收截止波长前的反射率与吸收截止波长后的反射率拼接
    return r

def NLcnstrnt(x, arg, I, step=1):
    if type(x) == np.ndarray:
        # r = dr(x, I,  step=step)  # 这里传入的x数组，其最后一个元素代表了吸收截止波长，dr(x)返回的数组只含有反射率，且为已经增量的反射率
        r = x  # 这里的x只有反射率，无截止波长
    else:
        x = np.asarray(x)
        # r = dr(x, I,  step=step)
        r = x  # 这里的x只有反射率，无截止波长
    X = trapz(np.multiply(arg[0], r), dx=step)
    Y = trapz(np.multiply(arg[1], r), dx=step)
    Z = trapz(np.multiply(arg[2], r), dx=step)
    x = X/(X+Y+Z)
    y = Y/(X+Y+Z)
    d = pow((arg[3] - x), 2)+pow((arg[4] - y), 2)
    # func = d
    return d

# 这是对Hot的xy坐标的限制
def NLcnstrnt_hot_xy(x, arg, step=1):
    # r = dr(x)   # 这里传入的x数组，其最后一个元素代表了吸收截止波长，dr(x)返回的数组只含有反射率，且为已经增量的反射率
    r = x  # 这里的x只有反射率，无截止波长
    X = trapz(np.multiply(arg[0], r), dx=step)
    Y = trapz(np.multiply(arg[1], r), dx=step)
    Z = trapz(np.multiply(arg[2], r), dx=step)
    x = X/(X+Y+Z)
    y = Y/(X+Y+Z)
    d = sqrt(pow((arg[3] - x), 2)+pow((arg[4] - y), 2))-0.0001
    func = d
    return func

# 这是对Hot的Y坐标的限制
def NLcnstrnt_hot_Y(x, Y0, Y_cool, step):
    # r = dr(x)   # 这里传入的x数组，其最后一个元素代表了吸收截止波长，dr(x)返回的数组只含有反射率，且为已经增量的反射率
    r = x  # 这里的x只有反射率，无截止波长
    Y = trapz(np.multiply(Y0, r), dx=step)
    d = 0.1-sqrt(pow((Y_cool - Y), 2))
    func = d
    return func

def Smoothcons(x):
    # r = dr(x)   # 这里传入的x数组，其最后一个元素代表了吸收截止波长，dr(x)返回的数组只含有反射率，且为已经增量的反射率
    r = x  # 这里的x只有反射率，无截止波长
    # np.linalg.norm()默认是二范数,即sqrt(x1^2+x2^2+...+xn^2)
    # diff()表示相邻元素的差分
    func = 0.05 - np.linalg.norm(np.diff(r))**2
    # func = 0.2 - integral
    return func

def Delta(r, arg, step=1):
    X = trapz(np.multiply(arg[0], r), dx=step)
    Y = trapz(np.multiply(arg[1], r), dx=step)
    Z = trapz(np.multiply(arg[2], r), dx=step)
    cal_x = X / (X + Y + Z)
    cal_y = Y / (X + Y + Z)
    delta = sqrt(pow((arg[3] - cal_x), 2) + pow((arg[4] - cal_y), 2))
    return delta, cal_x, cal_y, Y

def XYZ_cal(r, arg, step=1):
    X = trapz(np.multiply(arg[0], r), dx=step)
    Y = trapz(np.multiply(arg[1], r), dx=step)
    Z = trapz(np.multiply(arg[2], r), dx=step)
    cal_x = X / (X + Y + Z)
    cal_y = Y / (X + Y + Z)
    return [cal_x, cal_y, X, Y, Z]

def Lab2CIEXY(L, a, b):
    Xn, Yn, Zn = 95.0489, 100, 108.8840
    fy = (L+16)/116
    fx = (a/500)+fy
    fz = fy-(b/200)

    if fy > (24/116):
        Y = Yn*(fy**3)
    elif fy <= (24/116):
        Y = (fy-(16/116))*(108/841)*Yn
    if fx > (24 / 116):
        X = Xn * (fx ** 3)
    elif fx <= (24 / 116):
        X = (fx-(16/116))*(108/841)*Xn
    if fz > (24 / 116):
        Z = Zn * (fz ** 3)
    elif fz <= (24 / 116):
        Z = (fz-(16/116))*(108/841)*Zn
    #coordinat in CIExyY
    x0 = X/(X+Y+Z)
    y0 = Y/(X+Y+Z)

    return x0, y0