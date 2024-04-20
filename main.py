from scipy.optimize import linprog, minimize
from functools import partial
import pandas as pd
import numpy as np
import mpmath as mp
from tqdm import trange
import function
from tristimulus_cal import tristimulus
from AdaptiveDeffev2 import AdaptiveDESolver, CustomMonitor
import Record
import mystic.penalty as pnty
from mystic.termination import ChangeOverGeneration,VTR
from mystic.strategy import Rand1Bin,RandToBest1Bin,Best1Bin
import mystic.strategy as strategy
# 设置计算精度与数据路径

global solver

mp.dps = 20
step = 10
num_variable = int(((760-360)/step)) + 1
cooling_power = 130
I_path = './Source_CMF_CIExy_data/astmg173.xls'
d65_path = './Source_CMF_CIExy_data/CIE_std_illum_D65.xlsx'
cmf_path = './Source_CMF_CIExy_data/colorMatchFcn.xlsx'
CIExy_path = './Source_CMF_CIExy_data/CIExy_vector.xlsx'

# I_souce是AM1.5的光谱,用于计算thermal load
I_souce = pd.read_excel(I_path, sheet_name='Sheet1')
wvlngth_idx = int(I_souce.loc[I_souce['wavelength/nm'] == 359.5].index.values) + 1# 找360nm波长对应的索引，将此索引与对应的值作为起始
I_souce = np.array(I_souce['I'][wvlngth_idx::step])
# # D65,用于计算颜色
# I_color = pd.read_excel(d65_path)
I_color = I_souce
# 读入颜色匹配函数
cmf = pd.read_excel(cmf_path)
cmf.drop('wavelength', axis=1, inplace=True)
CIExy = pd.read_excel(CIExy_path)
CIEx = CIExy['x'].to_numpy()
CIEy = CIExy['y'].to_numpy()

# 获取与反射率数值积分前的三色刺激值向量
if I_color.all() == I_souce.all():
    tri_val = tristimulus(I_color, cmf, step)
else:
    I_color = np.array(I_color['I'][::step])
    tri_val = tristimulus(I_color, cmf, step)
color_relate = {}
color_relate[0] = tri_val[0]
color_relate[1] = tri_val[1]
color_relate[2] = tri_val[2]

# r0为初值, bounds_r是每个变量的边界((0 ,1), (0, 1), (0, 1),... ..., (0, 1)), tuple类型，元素也是tuple
r0_cool = 0.9 * np.ones(num_variable)
# r0_hot = 0.1 * np.ones(num_variable)
# # 添加截止频率初值l0
# l0 = 360
# r0_cool = np.append(r0_cool, l0)

# 设置反射率的上下界
bounds_r = []
for i in range(num_variable):
    bounds_r.append((0, 1))

# # 设置吸收截止波长的上下界
# bounds_r.append((360, 760))
# bounds_r = tuple(bounds_r)

# 初始化各种参数
# # Cool parameter
Cool = Record.Record_process()
# # Hot parameter
# 要多记录一个CIExyY的Y要保持和计算cool的色彩中的Y一致,并且没有截止波长的计算,非线性约束也要多一条对Y的约束
# Y_hot = []
count = 0  # 遍历次数计数
random_numbers = list(range(3100))
iter_time = len(random_numbers)  # 与random_number中的index数量一样
for i in trange(iter_time, desc='Processing', leave=True):
    count += 1  # 每进一次循环计数加一

    # 优化求解过程
    # i = random_numbers[i]
    i=49
    coordinate_x = CIEx[i]
    coordinate_y = CIEy[i]
    color_relate[3] = coordinate_x
    color_relate[4] = coordinate_y
    # 定义目标函数，例如使用 func_cool
    def objective(x):
        try:
            result = function.func_cool(x, I_souce, step)
            if not np.isfinite(result):
                print(f"Non-finite result obtained for input: {x}")
            return result
        except Exception as e:
            print(f"Error with input {x}: {e}")
            return np.inf  # 返回一个巨大的数，避免程序崩溃
    # 定义约束条件
    def constraints(x):
        try:
            result = function.NLcnstrnt(x, color_relate, I_souce, step)
            # if not all(np.isfinite(result)):
            #     print(f"Constraint non-finite for input {x}")
            #     return np.array([np.inf])  # 返回一个使这个点无效的结果
            solver.last_constraint = result
            return result
        except Exception as e:
            print(f"Error in constraints with input {x}: {e}")
            return np.array([np.inf])  # 使这个点无效
    @pnty.quadratic_equality(constraints,k=1e4,h=5)
    def penalty(x):
        return 0.0
    ndim = num_variable
    npop = 10*ndim  # Population size
    maxiter = 2000  # Maximum number of iterations
    stepmon = CustomMonitor(10)
    solver = AdaptiveDESolver(ndim, npop)
    solver.SetGenerationMonitor(stepmon)
    solver.SetRandomInitialPoints(min=[b[0] for b in bounds_r], max=[b[1] for b in bounds_r])
    solver.SetStrictRanges(min=[b[0] for b in bounds_r], max=[b[1] for b in bounds_r])
    termination = ChangeOverGeneration(tolerance=1e-3, generations=50)
    solver.SetTermination(termination)
    # solver.SetConstraints(constraints)
    solver.SetPenalty(penalty)
    solver.Solve(objective, strategy=Rand1Bin, CrossProbability=0.9)

    print("Best Solution: {}".format(solver.bestSolution))
    print("Best Objective: {}".format(solver.bestEnergy))

    # NLcnstrnt_partial = partial(function.NLcnstrnt, arg=color_relate, I=I_souce, step=step)
    # cons = {'type': 'eq', 'fun': NLcnstrnt_partial}
    #         # {'type': 'eq', 'fun': Smoothcons}]    # 加入非线性限制
    # # cons_hot = [{'type': 'eq', 'fun': NLcnstrnt_hot_xy},
    # #             {'type': 'eq', 'fun': NLcnstrnt_hot_Y},
    # #             {'type': 'eq', 'fun': Smoothcons},
    # #             ]  # 加入非线性限制
    # # Cool
    # res_cool = minimize(function.func_cool, r0_cool, args=(I_souce, step), method='SLSQP',
    #                     constraints=cons, bounds=bounds_r, options={'maxiter': 10000})  # 开始优化
    # # Recording
    # delta = function.NLcnstrnt(res_cool.x, arg=color_relate, I=I_color, step=step)
    # print("Optimal Solution:", np.trapz(np.multiply(-I_souce, res_cool.x), dx=step), "delta_r:", delta)
    # Cool.record(res=res_cool, cls='Cool', I_souce=I_souce, step=step,
    #             cooling_power=cooling_power, color_relate=color_relate, i=i)

