import function
from math import *
from scipy.integrate import trapz
import numpy as np
import time
import pandas as pd

class Record_process:
    def __init__(self):
        self.P = []  # 理论上某特定颜色的最冷热负载集合
        self.Delta = []   # 色差
        self.R = []  # 最冷热负载对应的反射率解集
        self.coordinate = []  # 待遍历的色彩坐标
        self.real_crdnt = []  # 通过解出来的反射率求得的色彩坐标
        self.Lambda = []  # 吸收截止波长的集合
        self.optimized_success = []
        self.optimized_failure_reason = []
        self.index = []
        self.Y = []

    def record(self, res, cls, I_souce, step, cooling_power, color_relate, i):
        # 优化结果记录，成功与否，若失败，失败的的理由是什么
        self.optimized_success.append(res['success'])
        if res['success'] == False:
            self.optimized_failure_reason.append(res['message'])
        else:
            self.optimized_failure_reason.append('None')

        # 反射率记录
        r = function.dr(res.x, I=I_souce, step=step)  # 获取优化后的反射率反谱,注意res_cool.x里边的反射率是没有加上反射率增量的
        # r_cool = res_cool.x # 只有反射率，无反射率增量
        self.R.append(r.reshape(-1, 1))  # 注意R_cool是记录优化求解的的全部颜色的反射率解集，r_cool为当前循环优化求解的反射率谱,
                                         # reshape(-1, 1)代表将行向量转置成列向量

        # 吸收截止波长记录
        lambda0 = ceil((res.x)[-1])  # 获取优化后的吸收截止波长
        self.Lambda.append(lambda0)

        # 热负载记录
        p = trapz(np.multiply(I_souce, (1 - r)))+0-cooling_power  # 记录理论上最冷的热负载
        self.P.append(p)  # 注意P_cool是记录遍历的全部热负载，p_cool为当前循环计算的热负载
        # 色差记录
        delta, cal_x, cal_y, y = function.Delta(r, arg=color_relate, step=step)# 根据优化求解的反射率，计算色差与颜色坐标
        # smoothness = np.linalg.norm(np.diff(r_cool))
        # smoothness = trapz((np.gradient(np.gradient(r_cool)))**2)
        self.Y.append(y)
        self.Delta.append(delta)
        # 色彩坐标记录（数据集值，非计算值）
        coordinate_xy = (color_relate[3], color_relate[4])
        self.coordinate.append(coordinate_xy)
        # 色彩坐标记录（计算值，非数据集值）
        coordinate_cool = (cal_x, cal_y)
        self.real_crdnt.append(coordinate_cool)
        # 索引号记录
        self.index.append(i)
        time.sleep(0.1)
        # 将计算的数据转存为excel
        writer_properties = pd.ExcelWriter('Thermal_properties.xlsx')
        writer_R = pd.ExcelWriter('R_CIExyY.xlsx')
        Optimized_success_save = pd.DataFrame(self.optimized_success)
        Failure_Reason_save = pd.DataFrame(self.optimized_failure_reason)
        Random_index = pd.DataFrame(self.index)
        P_save = pd.DataFrame(self.P)
        Y_save = pd.DataFrame(self.Y)
        Lambda_save = pd.DataFrame(self.Lambda)
        R_save = np.hstack(self.R)  # 若不进行hstack操作，则R_cool是个三维列表，之后pd.DataFrame会报错
        R_save = pd.DataFrame(R_save)
        delta_save = pd.DataFrame(self.Delta)
        coordinate_save = pd.DataFrame(self.coordinate)
        real_crdnt_save = pd.DataFrame(self.real_crdnt)
        save_data = pd.concat([P_save,
                               Y_save,
                               Lambda_save,
                               delta_save, coordinate_save, real_crdnt_save,
                               Random_index,
                               Optimized_success_save, Failure_Reason_save], axis=1)
        save_data.columns = ['P_{}'.format(cls),
                             'Y_{}'.format(cls),
                             'cut-off wavelength',
                             'delta{}'.format(cls),
                             'data_x', 'data_y',
                             'real_x', 'real_y',
                             'index',  # index是色坐标的索引
                             'Optimize success',
                             'Optimize failure reason']
        save_data.to_excel(writer_properties, sheet_name='{}'.format(cls), index=False)
        writer_properties.save()
        #  将计算得到的R进行保存
        R_save.to_excel(writer_R, sheet_name='R_{}'.format(cls))
        writer_R.save()