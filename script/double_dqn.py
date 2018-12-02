import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F

import chainer.computational_graph as c
from curio_dqn import DQN 
from chainerrl.recurrent import state_kept


class DoubleDQN(DQN):
    """Double DQN.

    See: http://arxiv.org/abs/1509.06461.
    """

    def _compute_target_values(self, exp_batch, gamma):

        batch_next_state = exp_batch['next_state']

        # 内部報酬を追加，ついでに学習
        ## バッチ内の次状態の予測モデルをつくる
        state_action = np.array([list(s)+list(a) for s,a in zip(exp_batch['state'],
            [np.eye(2)[int(a)] for a in exp_batch['action']])],dtype=np.float32)
        for i in range(10000):
            # loss = F.mean_squared_error(self.f_pred(state_action), exp_batch['next_state'])
            y = self.f_pred(state_action)
            t = exp_batch['state']
            loss = F.mean_squared_error(y, t)
            g = c.build_computational_graph([y,y])
            with open('graph2.dot', 'w') as o:
                o.write(g.dump())

            self.f_pred.cleargrads()
            loss.backward()
            self.optimizer_f.update()
            print('loss',loss)
            raise
        print('break')
        raise

        with chainer.using_config('train', False), state_kept(self.q_function):
            next_qout = self.q_function(batch_next_state)

        target_next_qout = self.target_q_function(batch_next_state)

        next_q_max = target_next_qout.evaluate_actions(
            next_qout.greedy_actions)

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max
