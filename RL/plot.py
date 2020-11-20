import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# left = np.array([1, 2])
# height = np.array([136.02, 111.16])
# label = ["DQN", "DDQN"]
# plt.bar(left, height, width=0.6, tick_label=label, align="center")
# plt.title("学習終了時のエピソード")
# # plt.xlabel("reward clipping")
# plt.ylabel("episode")
# plt.ylim(0, 200)
# # plt.grid(True)
# plt.savefig('./img/graph/double.png')


r = np.loadtxt('reward_list_1.txt')
rlist = []
for i in range(500):
    rlist.append(np.mean(r[0:i+1]))

plt.plot(rlist)
plt.show()

# 移動平均はちょっと結果が微妙
# num = 100
# b=np.ones(num)/float(num)
# y=np.convolve(r, b, mode='same')
# plt.plot(y)
# plt.show()