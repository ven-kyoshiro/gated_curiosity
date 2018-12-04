from hard_cartpole import CartPoleEnv
from curio_dqn import DoubleDQN
from chainerrl.recurrent import state_kept
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import pickle 
from notify import notify

class QFunction(chainer.Chain):

    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))


class ForwardPredictor(chainer.Chain):
    
    def __init__(self, obs_size, n_actions, n_hidden_channels=128):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size+n_actions, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l3 = L.Linear(n_hidden_channels, obs_size)
            self.bn1 = L.BatchNormalization(n_hidden_channels)
            self.bn2 = L.BatchNormalization(n_hidden_channels)
            self.bn3 = L.BatchNormalization(n_hidden_channels)

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        h = F.relu(self.bn1(self.l0(x)))
        h = F.relu(self.bn2(self.l1(h)))
        h = F.relu(self.bn3(self.l2(h)))
        return self.l3(h)




def main():
    env = CartPoleEnv()
    env.reset()

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q_func = QFunction(obs_size, n_actions)

   
    _q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
    obs_size, n_actions,
    n_hidden_layers=2, n_hidden_channels=50)
    f_pred = ForwardPredictor(obs_size,n_actions)

    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    optimizer_f = chainer.optimizers.Adam(eps=1e-7)
    optimizer_f.setup(f_pred)
    # Set the discount factor that discounts future rewards.
    gamma = 0.95

    # Use epsilon-greedy for exploration
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=env.action_space.sample)

    # DQN uses Experience Replay.
    # Specify a replay buffer and its capacity.
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    # Since observations from CartPole-v0 is numpy.float64 while
    # Chainer only accepts numpy.float32 by default, specify
    # a converter as a feature extractor function phi.
    phi = lambda x: x.astype(np.float32, copy=False)

    # Now create an agent that will interact with the environment.
    agent = DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_interval=1,
        target_update_interval=100, phi=phi,
        f_pred=f_pred,
        optimizer_f=optimizer_f,
        std=100,
        loop=20)

    n_episodes = 2000
    max_episode_len = 200
    record=[]
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        reward = 0
        done = False
        R = 0  # return (sum of rewards)
        t = 0  # time step
        while not done and t < max_episode_len:
            # Uncomment to watch the behaviour
            # env.render()
            action = agent.act_and_train(obs, reward)
            obs, reward, done, _ = env.step(action)
            R += reward
            t += 1
        agent.record['reward'].append(R)
        with open('test_rec05_std'+str(agent.std)+'_loop'+str(agent.loop)+'.pickle',mode='wb') as f:
            pickle.dump(agent.record,f)
        if i % 100 == 0:
            notify('test5\n : episode:'+str(i)+' R:'+str(R))
        agent.stop_episode_and_train(obs, reward, done)
    print('Finished.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        notify('!!!! Error !!!!\n'+str(e)+'\n')
