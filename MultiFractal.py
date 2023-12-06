import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
from numpy import polyfit

def Fitting(x, y, n=1):
    return polyfit(x, y, n)

def fractal(image_path, q_min=-5,q_max=5):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = img_array.shape[:2]
    # print(height, width)

    with open(label_path, 'r') as f:
        data = json.load(f)
    shapes = data['shapes']
    albite_points = []
    for shape in shapes:
        label = shape['label']
        if label == 'albite':
            points = shape['points']
            albite_points.append(points)

    region_mask = np.zeros((height, width), dtype=np.uint8)
    for polygon_points in albite_points:
        polygon_points = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(region_mask, [polygon_points], 255)

    region = np.where(region_mask, img_array, 0)  # 将掩码应用于区域
    binary_image = np.where(region > 0, 255, 0).astype(np.uint8)

    # 权重因子
    ql = np.linspace(q_min,q_max,21)

    # print(ql)
    # 配分函数
    xl=[]
    # 质量指数
    tl=[]
    # 标度指数
    al=[]
    # 奇异谱函数
    fl=[]
    # 广义维数
    dl=[]

    for q in ql:
        # 观测尺度
        rl = []
        # 质量指数
        xl_t=[]
        # 标度指数
        xl_a=[]
        # 奇异普函数
        xl_f=[]
        # 广义维数
        xl_d=[]

        rl = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
        for r in rl:
            N_sum = 0
            Nl = []

            for h in range(height // r):
                for w in range(width // r):
                    sub_image = binary_image[h * r:h * r + r, w * r:w * r + r]
                    N = np.sum(sub_image == 255)
                    N_sum += N
                    Nl.append(N)

            Pil = [N / N_sum for N in Nl if N != 0]

            # 配分函数
            X_t=0
            for pi in Pil:
                X_t += pi ** q
            X_a =0
            X_f=0
            for pi in Pil:
                X_a += pi ** q / X_t  * np.log(pi)
                X_f += pi ** q / X_t * np.log(
                    pi ** q / X_t
                )
            xl_t.append(X_t)
            xl_a.append(X_a)
            xl_f.append(X_f)

            X_d=0
            if q==1:
                for pi in Pil:
                    X_d += pi*np.log(pi)
                xl_d.append(X_d)

        fit = Fitting(np.log(rl), np.log(xl_t), n=1)
        t = fit[0]  # 斜率
        tl.append(t)
        # print(tl)
        fitted_x = np.log(rl)
        fitted_y = np.polyval(fit, fitted_x)
        xl.append([fitted_x, fitted_y, q])

        a = Fitting(np.log(rl), xl_a)[0]
        al.append(a)

        f = Fitting(np.log(rl), xl_f)[0]
        fl.append(f)

        if q == 1:
            dl.append(Fitting(np.log(rl), xl_d)[0])
        else:
            dl.append(t / (q - 1))

        al = list(al)
        ql = list(ql)
        dl = list(dl)

    coeff = Fitting(al, fl, 2)

    out_data={
        'coeff[0]': [coeff[0]],
        'coeff[1]': [coeff[1]],
        'coeff[2]': [coeff[2]],
        'H': [(1 + dl[ql.index(2)]) / 2],
        'D0': [dl[ql.index(0)]],
        'D1': [dl[ql.index(1)]],
        'D2': [dl[ql.index(2)]],
        'D({})'.format(q_min): [dl[ql.index(q_min)]],
        'D(+{})'.format(q_max): [dl[ql.index(q_max)]],
        'D({})-D(+{})'.format(q_min, q_max): [dl[ql.index(q_min)] - dl[ql.index(q_max)]],
        'α(q=0)': [al[ql.index(0)]],
        'α(q=1)': [al[ql.index(1)]],
        'α(q=2)': [al[ql.index(2)]],
        'α(q={})'.format(q_min): [al[ql.index(q_min)]],
        'α(q=+{})'.format(q_max): [al[ql.index(q_max)]],
        'α(q={})-α(q=+{})'.format(q_min, q_max): [al[ql.index(q_min)] - al[ql.index(q_max)]],
        'f(q=0)': [fl[ql.index(0)]],
        'f(q=1):': [fl[ql.index(1)]],
        'f(q=2)': [fl[ql.index(2)]],
        'f(q={})'.format(q_min): [fl[ql.index(q_min)]],
        'f(q=+{})'.format(q_max): [fl[ql.index(q_max)]],
        'f(q={})-f(q=+{})'.format(q_min, q_max): [fl[ql.index(q_min)] - fl[ql.index(q_max)]],
    }
    figure_data={
        'q': ql,
        'τ(q)': tl,
        'α(q)': al,
        'f(α)': fl,
        'D(q)': dl,
    }
    for i,item in enumerate(ql):
        figure_data['q={}_r'.format(item)] = list(xl[i][0])
        figure_data['q={}_X'.format(item)] = list(xl[i][1])


    for key, value in out_data.items():
        print(key, value)
    for key, value in figure_data.items():
        print(key, value)

    return [
        binary_image,
        image_path,
        xl,
        al,
        ql,
        dl,
        tl,
        fl,
        coeff,
        out_data,
        figure_data]


def Draw(para):
    fdir = 'images'
    os.makedirs(fdir, exist_ok=True)
    binary_image,image_path, xl, al, ql, dl, tl, fl, coeff= para[:9]
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].imshow(binary_image,cmap='gray')
    axes[0, 0].set_title('gray')
    axes[0, 0].axis('off')

    axes[0, 1].plot(al, fl, 'b', marker='o')
    #axes[0, 1].plot([al[ql.index(0)] for _ in fl], fl, linestyle='--')
    axes[0, 1].set_xlabel('α')
    axes[0, 1].set_ylabel('f')
    axes[0, 1].set_title('f-α\nf = {:.4f}α^2 + {:.4f}α + {:.4f}'.format(coeff[0], coeff[1], coeff[2]))

    axes[1, 0].plot(ql, dl, 'm', marker='o')
    axes[1, 0].set_xlabel('q')
    axes[1, 0].set_ylabel('D')
    axes[1, 0].set_title('D-q')

    for i, item in enumerate(ql):
        if item < -0.5 or item > 0.5:
            axes[1, 1].plot(xl[i][0], xl[i][1], label='q={:.4f}'.format(xl[i][2]))
        else:
            axes[1, 1].plot(xl[i][0], xl[i][1])

    axes[1, 1].set_xlabel('log(r)')
    axes[1, 1].set_ylabel('log(X)')
    axes[1, 1].set_title('X-r')

    plt.tight_layout()
    # fig.savefig(f'{fdir}/{image_path}_muscovite.png', dpi=300)
    fig.savefig(f'{fdir}/{image_path}_albite.png', dpi=300)
    # fig.savefig(f'{fdir}/{image_path}_quartz.png', dpi=300)
    plt.show()

