import torch.optim as optim

from dqn_model import DQN_RAM
from dqn_learn_no_randomselect import OptimizerSpec, dqn_learing
from utils.schedule import LinearSchedule

BATCH_SIZE = 64
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 3000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 32
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.0001
ALPHA = 0.95
EPS = 0.01

def main():


    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS, weight_decay=10**-5),
    )

    exploration_schedule = LinearSchedule(300000, 0.1)

    dqn_learing(
        q_func=DQN_RAM,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
    )

if __name__ == '__main__':

    main()
