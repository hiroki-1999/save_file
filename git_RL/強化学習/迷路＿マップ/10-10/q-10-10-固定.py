# 使用するパッケージの宣言
import numpy as np
import matplotlib.pyplot as plt
import math


# 初期の方策を決定するパラメータtheta_0を設定　移動できるところの設定

# 行は状態0～7、列は移動方向で↑、→、↓、←を表す
theta_0 = np.array([[np.nan, 1, 1, np.nan],  # s0 右下
                    [np.nan, 1, np.nan, 1],  # s1 左右
                    [np.nan, 1, 1, 1],  # s2 左右下
                    [ np.nan,np.nan,np.nan,1],  # s3 左
                    [np.nan, 1, 1, np.nan],  # s4 右下
                    [np.nan,1, np.nan,1],  # s5 左右
                    [np.nan,1, np.nan,1],  # s6 左右
                    [np.nan,1, np.nan,1],  # s7 左右
                    [np.nan,1, np.nan,1],  # s8 左右
                    [np.nan,np.nan, np.nan,1],  # s9 左
                    
                    [1, 1, 1, np.nan],  # s10 上下右
                    [np.nan,1, np.nan,1],  # s11 左右
                    [1,1, np.nan,1],  # s12 上左右
                    [np.nan, np.nan, 1, 1],  # s13 下左
                    [1 ,np.nan, 1, np.nan],  # s14 上下
                    [np.nan, 1, 1, np.nan],  # s15 右下
                    [np.nan, 1, np.nan, 1],  # s16 左右
                    [np.nan, 1, np.nan, 1],  # s17 左右
                    [np.nan, 1, np.nan, 1],  # s18 左右
                    [np.nan, np.nan, 1, 1],  # s19 下左
                    
                    [1, 1, 1, np.nan],  # s20 上下右
                    [np.nan,np.nan, np.nan,1],  # s21 左
                    [np.nan, 1, 1, np.nan],  # s22 右下
                    [1 ,np.nan, 1, 1],  # s23上下左
                    [np.nan, 1, 1, np.nan],  # s24右下
                    [1 ,np.nan, 1, 1],  # s25上下左
                    [np.nan, 1, 1, np.nan],  # s26右下
                    [np.nan, 1, 1, 1],  # s27 左右下
                    [np.nan, 1, 1, 1],  # s28 左右下
                    [1 ,np.nan, 1, 1],  # s29上下左
                    
                    [1, 1, np.nan, np.nan],  # s30上右
                    [np.nan, 1, 1, 1],  # s31 左右下
                    [1, np.nan, np.nan, 1],  # s32上左
                    [1, 1, np.nan, np.nan],  # s33上右
                    [np.nan,np.nan, np.nan,1],  # s34 左
                    [1 ,np.nan, 1, np.nan],  # s35 上下
                    [1 ,np.nan, 1, np.nan],  # s36 上下
                    [1 ,np.nan, 1, np.nan],  # s37 上下
                    [1, 1, 1, np.nan],  # s38 上下右
                    [1 ,np.nan, 1, 1],  # s39上下左
                    
                    [np.nan ,np.nan, 1, np.nan],  # s40 下
                    [1 ,np.nan, 1, np.nan],  # s41 上下
                    [np.nan, 1, 1, np.nan],  # s42 右下
                    [np.nan, 1, np.nan, 1],  # s43 左右
                    [np.nan, np.nan, 1, 1],  # s44 下左
                    [1, 1, 1, np.nan],  # s45 上下右
                    [1,1, np.nan,1],  # s46 上左右
                    [1,1, np.nan,1],  # s47 上左右
                    [1,1, np.nan,1],  # s48 上左右
                    [1 ,np.nan, 1, 1],  # s49上下左
                    
                    [1, 1, 1, np.nan],  # s50 上下右
                    [1 ,np.nan, 1, 1],  # s51上下左
                    [1 ,np.nan, np.nan, np.nan],  #s52 上
                    [np.nan, 1, 1, np.nan],  # s53 右下
                    [1 ,np.nan, 1, 1],  # s54上下左
                    [1 ,np.nan, 1, np.nan],  # s55 上下
                    [np.nan, 1, np.nan,  np.nan],  # s56右
                    [np.nan, 1, np.nan, 1],  # s57左右
                    [np.nan, 1, 1, 1],  # s58左右下
                    [1 ,np.nan, 1, 1],  # s59上下左
                    
                    [1 ,np.nan, 1, np.nan],  # s60 上下
                    [1, 1, np.nan, np.nan],  # s61上右
                    [np.nan,np.nan, np.nan,1],  # s62 左
                    [1 ,np.nan, 1, np.nan],  # s63 上下
                    [1 ,np.nan, 1, np.nan],  # s64 上下
                    [1 ,np.nan, 1, np.nan],  # s65 上下
                    [np.nan ,np.nan, 1, np.nan],  # s66 下
                    [np.nan, 1, 1, np.nan],  # s67右下
                    [1,1, np.nan,1],  # s68 上左右
                    [1 ,np.nan, 1, 1],  # s69上下左
                    
                    [1 ,np.nan, 1, np.nan],  # s70 上下
                    [np.nan, 1, 1, np.nan],  # s71右下
                    [np.nan, 1, np.nan, 1],  # s72左右
                    [1 ,np.nan, 1, np.nan],  # s73 上下
                    [1, 1, 1, np.nan],  # s74 上下右
                    [1 ,np.nan, 1, 1],  # s75上下左
                    [1 ,np.nan, 1, np.nan],  # s76 上下
                    [1, 1, 1, np.nan],  # s77 上下右
                    [np.nan, np.nan, 1, 1],  # s78下左
                    [1 ,np.nan, 1, np.nan],  # s79 上下
                    
                    [1, 1, 1, np.nan],  # s80 上下右
                    [1,1, np.nan,1],  # s81 上左右
                    [np.nan, np.nan, 1, 1],  # s82下左
                    [1 ,np.nan, 1, np.nan],  # s83 上下
                    [1 ,np.nan, 1, np.nan],  # s84 上下
                    [1, 1, 1, np.nan],  # s85 上下右
                    [1,1, np.nan,1],  # s86 上左右
                    [1,1, np.nan,1],  # s87 上左右
                    [1 ,np.nan, 1, 1],  # s88上下左
                    [1 ,np.nan, 1, np.nan],  # s89 上下
                    
                    [1, 1, np.nan, np.nan],  # s90上右
                    [np.nan, 1, np.nan, 1],  # s91左右
                    [1, np.nan, np.nan, 1],  # s92上左
                    [1, np.nan, np.nan, np.nan],  # s93上
                    [1, 1, np.nan, np.nan],  # s94上右
                    [1,1, np.nan,1],  # s95 上左右
                    [np.nan, 1, np.nan, 1],  # s96左右
                    [np.nan, 1, np.nan, 1],  # s97左右
                    [1, np.nan, np.nan, 1],  # s98上左
                    
                    
                    ])





# 方策パラメータtheta_0をランダム方策piに変換する関数の定義


def simple_convert_into_pi_from_theta(theta):
    '''単純に割合を計算する'''

    [m, n] = theta.shape  # thetaの行列サイズを取得
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])  # 割合の計算

    pi = np.nan_to_num(pi)  # nanを0に変換

    return pi

# ランダム行動方策pi_0を求める
pi_0 = simple_convert_into_pi_from_theta(theta_0)





# 初期の行動価値関数Qを設定

[a, b] = theta_0.shape  # 行と列の数をa, bに格納
Q = np.random.rand(a, b) * theta_0 * 0.1
# *theta0をすることで要素ごとに掛け算をし、Qの壁方向の値がnanになる





# ε-greedy法を実装


def get_action(s, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]

    # 行動を決める
    if np.random.rand() < epsilon:
        # εの確率でランダムに動く
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:
        # Qの最大値の行動を採用する
        next_direction = direction[np.nanargmax(Q[s, :])]

    # 行動をindexに
    if next_direction == "up":
        action = 0
    elif next_direction == "right":
        action = 1
    elif next_direction == "down":
        action = 2
    elif next_direction == "left":
        action = 3

    return action


def get_s_next(s, a, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]
    next_direction = direction[a]  # 行動aの方向

    # 行動から次の状態を決める
    if next_direction == "up":
        s_next = s - 10  # 上に移動するときは状態の数字が10小さくなる
    elif next_direction == "right":
        s_next = s + 1  # 右に移動するときは状態の数字が1大きくなる
    elif next_direction == "down":
        s_next = s + 10  # 下に移動するときは状態の数字が10大きくなる
    elif next_direction == "left":
        s_next = s - 1  # 左に移動するときは状態の数字が1小さくなる

    return s_next



# Q学習による行動価値関数Qの更新


def Q_learning(s, a, r, s_next, Q, eta, gamma):

    if s_next == 99:  # ゴールした場合
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])

    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next,: ]) - Q[s, a])

    return Q





# Q学習で迷路を解く関数の定義、状態と行動の履歴および更新したQを出力


def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    s = 0  # スタート地点
    a = a_next = get_action(s, Q, epsilon, pi)  # 初期の行動
    s_a_history = [[0, np.nan]]  # エージェントの移動を記録するリスト

    while (1):  # ゴールするまでループ
        a = a_next  # 行動更新

        s_a_history[-1][1] = a
        # 現在の状態（つまり一番最後なのでindex=-1）に行動を代入

        s_next = get_s_next(s, a, Q, epsilon, pi)
        # 次の状態を格納

        s_a_history.append([s_next, np.nan])
        # 次の状態を代入。行動はまだ分からないのでnanにしておく

        # 報酬を与え,　次の行動を求めます
        if s_next == 99:
            r = 1  # ゴールにたどり着いたなら報酬を与える
            a_next = np.nan
        else:
            r = 0
            a_next = get_action(s_next, Q, epsilon, pi)
            # 次の行動a_nextを求めます。

        # 価値関数を更新
        Q = Q_learning(s, a, r, s_next, Q, eta, gamma)

        # 終了判定
        if s_next == 99:  # ゴール地点なら終了
            break
        else:
            s = s_next

    return [s_a_history, Q]





# Q学習で迷路を解く

eta = 0.1  # 学習率
gamma = 0.9  # 時間割引率
epsilon = 0.2  # ε-greedy法の初期値
v = np.nanmax(Q, axis=1)  # 状態ごとに価値の最大値を求める
is_continue = True
episode = 1

V = []  # エピソードごとの状態価値を格納する
V.append(np.nanmax(Q, axis=1))  # 状態ごとに行動価値の最大値を求める
###
A=[]
B=[]
z=1
for i in range(496):#(E)
    A.append(0)
    B.append(0)
rewards_q_learning=[]
###
#rewards_q_learning.append(0)
for i in range(100):#(X)
    
    while is_continue:  # is_continueがFalseになるまで繰り返す
        ###print("エピソード:" + str(episode))

        # ε-greedyの値を少しずつ小さくする
        #epsilon = epsilon * (1/math.sqrt(episode))
        #epsilon = epsilon /1.1
        # Q学習で迷路を解き、移動した履歴と更新したQを求める
        [s_a_history, Q] = goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi_0)

        # 状態価値の変化
        #new_v = np.nanmax(Q, axis=1)  # 状態ごとに行動価値の最大値を求める
        #print(np.sum(np.abs(new_v - v)))  # 状態価値関数の変化を出力
        #v = new_v
        #V.append(v)  # このエピソード終了時の状態価値関数を追加

        print("迷路を解くのにかかったステップ数は" + str(len(s_a_history) - 1) + "です")
    
    
        #####
        rewards_q_learning.append(len(s_a_history) - 1)
         #print(rewards_q_learning)
        #####
        # 100エピソード繰り返す
        episode = episode + 1
        if episode > 500:
            break

    for k in range(496):#(E/5)
        A[k]=rewards_q_learning[k+i*500]+rewards_q_learning[k+1+i*500]+rewards_q_learning[k+2+i*500]+rewards_q_learning[k+3+i*500]+rewards_q_learning[k+4+i*500]+A[k]           
        print(A[k])
        
    ###Q値の初期化、数を数える変数の初期化    
    [a, b] = theta_0.shape  # 行と列の数をa, bに格納
    Q = np.random.rand(a, b) * theta_0 * 0.1
    pi_0 = simple_convert_into_pi_from_theta(theta_0)
    #if
    epsilon = 0.2
    episode = 1
    is_continue = True
    j=0
    k=0

#print(rewards_q_learning)
#print(A)    
for i in range(496):#(E/5)
        A[i]=A[i]/500#(X)
print(A)
plt.plot(A)
plt.show()







    
