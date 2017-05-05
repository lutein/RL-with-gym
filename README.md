# deep deterministic policy gradient

## gym

- constant set:
```python
    RENDER_ENV = True
    # Use Gym Monitor
    GYM_MONITOR_EN = True
    # Gym environment
    ENV_NAME = 'Pendulum-v0'
    # Directory for storing gym results
    MONITOR_DIR = './results/gym_ddpg'
    # File for saving reward and qmax
    RESULTS_FILE = './results/rewards.npz'
    RANDOM_SEED = 1234
    # Size of replay buffer
    BUFFER_SIZE = 10000
    MINIBATCH_SIZE = 128
```
- functions:

这里的动作和状态维数应该是常数。

初始动作的产生带有exploration噪声（使用随机的动作）：
```python
    a = actor.predict(np.reshape(s, (1, s_dim))) + (1. / (1. + i))
    #reshape(a, newshape, order='C')
```
而在定义actor.predict时为：
```python
    def predict(self, state):
        return self.sess.run(self.out, feed_dict={self.inputs: state})
    #self.out = network.get_actor_out(is_target=False)=self.actor_target_y = self._create_actors(self.state_feature_target)
```
然后执行动作，获得下一步的状态和奖励：
```python
    s2, r, terminal, info = env.step(a[0])
```
具体代码：
```python
    #1. 打开模拟环境
    env = gym.make(ENV_NAME)#ENV_NAME指的是例如："Pendulum-v0","CartPole-v0"之类的给定环境名称
    #2. 初始化
    s = env.reset() #initialize, before every episode begin, state needs initialize
    env.render()
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    action_bound = env.action_space.high#env.action_space.high = -env.action_space.low
    #3. 产生动作，并根据动作产生下一步状态和奖励
    next_state,reward,done,_ = env.step(action)
    #随机
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    #Monitor
    if GYM_MONITOR_EN:
       if not RENDER_ENV:
          env = wrappers.Monitor(env, MONITOR_DIR, video_callable=False, force=True)
       else:
          env = wrappers.Monitor(env, MONITOR_DIR, force=True)
    if GYM_MONITOR_EN:
       env.monitor.close()
```
## experience replay

- 伪代码

![preview](https://pic1.zhimg.com/c24454f472843ef5caef2733d50aba00_r.png)

- 理解：

在每次训练时，先初始化状态，然后在训练的每次迭代时，要么随机选取一个动作(exploration)，要么选择使得Ｑ值最大的动作。在模拟器中执行动作并观察奖励和状态，将状态转移矩阵存储在经验池中。

如果迭代结束，那么最后y_j就是反馈值，如果没有结束，那么![equation2](http://latex.codecogs.com/gif.latex?y_j)就是
![equation1](http://latex.codecogs.com/gif.latex?r_j+\gamma\max_{a'}Q(\phi_{j+1},a';\theta))，接着用梯度下降来最小化![equation3](http://latex.codecogs.com/gif.latex?(y_j-Q(\phi_j,a_j;\theta))^2)

相当于课程PPT中讲的那个图：这里的Ｑ可以理解为轨迹？目的是产生使得轨迹尽可能接近的动作。



所以说就是存储轨迹？



- 代码

一共四个子函数：

`add`:输入当前s,a,r,t和下一步s2，写成矩阵(s,a,r,t,s2),如果此时计数小于缓冲区大小，就将经验矩阵加到缓冲区右侧，并且计数加一；如果计数已经超过缓冲区大小，左出栈，缓冲区附加经验矩阵。

`size`:返回计数值（初始为０）

`sample_batch`:输入batch_size，如果计数值小于批尺寸，batch就随机选择一组缓冲区和计数值；如果大于批尺寸，就随机选缓冲区和批尺寸。其中的第０，１，２，３，４列分别代表`s_batch,a_batch,r_batch,t_batch,s2_batch`，最终返回的也是这些值。

> 关于批尺寸：batch的选择决定的是下降方向。大概指的就是每次训练的数据集大小。<br />
> <br />如果数据集较小，可以采用全数据集(Full Batch Learning)，有两个好处：由全数据集确定的方向能更好地代表样本总体，准确地朝向极值所在方向；不同权重梯度值差别大导致的选取全局学习率困难，如果用全数据集的话就可以使用Rprop，只基于梯度符号，针对性单独更新权值。<br />
> <br />全数据集的对立面是在线学习(Online Learning)，每次只训练一个样本，`batch_size=1`

`clear`:　队列和计数值均清零

调用：
```python
    from replay_buffer import ReplayBuffer
    #in function train(), inputs are constant defined in part 1, 10000, 1234
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
    replay_buffer.add(np.reshape(s, (s_dim,)), np.reshape(a, (a_dim,)), r,
                                  terminal, np.reshape(s2, (s_dim,)))
    # Keep adding experience to the memory until
    # there are at least minibatch size samples
    if replay_buffer.size() > MINIBATCH_SIZE:
        s_batch, a_batch, r_batch, t_batch, s2_batch = \
            replay_buffer.sample_batch(MINIBATCH_SIZE)
        #经验池满了之后，可以计算目标Ｑ值了
```
计算reward, Q_{max}

如何获得q：其中函数critic_target.predict的输入是下一状态，和以此为输入的下一目标动作，输出目标value
```python
    target_q = critic_target.predict(s2_batch, actor_target.predict(s2_batch))
```
然后计算y_i，按照２中的公式：
```python
    y_i = []
    for k in range(MINIBATCH_SIZE):
         if t_batch[k]:
             y_i.append(r_batch[k])
         else:
             y_i.append(r_batch[k] + GAMMA * target_q[k])#GAMMA＝0.99, dicsount factor
```
为了得到最大Ｑ，需要将其放入网络中训练
```python
    predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
```
每个episode中最大Ｑ值的平均值：
```python
    #ep_ave_max_q += np.amax(predicted_q_value)
    ep_ave_max_q += np.mean(predicted_q_value)
```
用随机梯度法更新四个网络
```python
    # Update the actor policy using the sampled gradient
    a_outs = actor.predict(s_batch)
     grads = critic.action_gradients(s_batch, a_outs)
    actor.train(s_batch, grads[0])
    
    # Update target networks
    actor_target.train()
    critic_target.train()
```
## 策略梯度

- 理论

注意：critic给出的分数是从当前到游戏结束的总reward，而系统返回的是当前获得的reward

反向传播：调整actor网络参数使得critic的打分尽量高

- ![equation1](http://latex.codecogs.com/gif.latex?\mu(s;\theta))是actor函数。此处s是输入的state，![equation1](http://latex.codecogs.com/gif.latex?\theta)是网络自己的参数。
- Q (s, a ; w)是critic函数。此处s和a是输入的state、action，w是网络自己的参数。
- 优化目标是让Q (s , a ; w) 尽量高
- 梯度（被称为policy gradient）是
![equation](http://latex.codecogs.com/gif.latex?\frac{\\partial\\mu}{\\partial\\theta}\\frac{\\partial Q}{\\partial a})
，其实很像是反向传播的chain rule：你可以把![equation1](http://latex.codecogs.com/gif.latex?\mu)看作是a，因为![equation1](http://latex.codecogs.com/gif.latex?\mu)的输出是action。（这个是deterministic policy gradient，跟传统的不太一样。）
- 然后用梯度上升更新![equation1](http://latex.codecogs.com/gif.latex?\theta)。（不是梯度下降，因为我们想让Q尽量大）。

理解：先随机产生一个状态，并选取一个动作，然后系统返回下一个状态的动作，状态，reward

## 训练整体流程：

- 算出actor网络输出关于所有参数的gradient
- 算出critic网络输出关于a的梯度
- 把两个梯度相乘得到policy gradient
- 最后把policy gradient加到actor网络所有参数上

