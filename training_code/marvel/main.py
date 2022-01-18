import argparse
import torch.optim as optim

from dqn_model import DQN_RAM
from dqn_learn import OptimizerSpec, dqn_learing
from utils.schedule import LinearSchedule

BATCH_SIZE = 64
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 300
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 1
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 1
ALPHA = 0.95
EPS = 0.01

def main():
    parser = argparse.ArgumentParser(description='python Implementation')
    parser.add_argument('--time', type =int, default =1)
    args = parser.parse_args()
    times = args.time
    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(300, 0.1)

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
        times=times
    )

if __name__ == '__main__':

    main()
