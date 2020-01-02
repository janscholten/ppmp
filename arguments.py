import argparse

def get_ppmp_argparser():
    parser = argparse.ArgumentParser(
        description='Provide arguments for DDPG agent (no sanity checks).')

    # agent parameters
    parser.add_argument(
        '--algorithm',
        help='Either ppmp, ppmp_human or pmp',
        default='ppmp')
    parser.add_argument(
        '--actor-lr',
        help='actor network learning rate',
        default=0.0001)
    parser.add_argument(
        '--critic-lr',
        help='critic network learning rate',
        default=0.002)
    parser.add_argument(
        '--gamma',
        help='discount factor for critic updates',
        default=0.99)
    parser.add_argument(
        '--tau',
        help='soft target update parameter',
        default=0.003)
    parser.add_argument(
        '--buffer-size',
        help='max size of the replay buffer',
        default=1000000)
    parser.add_argument(
        '--minibatch-size',
        help='size of minibatch for training the actor',
        default=64)
    parser.add_argument(
        '--heads',
        help='number of heads for multihead',
        default=10)
    parser.add_argument(
        '--initial-variance',
        help='Initial variance of actor network ouput layer (between heads)',
        default=0.001)

    # predictor/selector
    parser.add_argument(
        '--buffer2-size',
        help='max size of the replay buffer',
        default=1600)
    parser.add_argument(
        '--cold-samples',
        help='The amount of samples collected without prediction',
        default=1500)
    parser.add_argument(
        '--filter-samples',
        help='The amount of samples before Q-filter reliability',
        default=4000)
    parser.add_argument(
        '--prediction-noise',
        help='Variance as fraction of action space',
        default=0.025)
    parser.add_argument(
        '--human-variance',
        help='Human feedback variance (action domain)',
        default=1e-8)
    parser.add_argument(
        '--resolution',
        help='Hysteresis in the correction, fraction of action space',
        default=0.125)
    parser.add_argument(
        '--scale',
        help='Fraction of action space',
        default=0.5)

    # oracle
    parser.add_argument(
        '--error',
        help='Erroneous feedback ERROR fraction of time',
        default=0)
    parser.add_argument(
        '--fb-amount',
        help='Scale amount of feedback',
        default=0.3)

    # run parameters
    parser.add_argument(
        '--env',
        help='choose the gym env- tested on {Pendulum-v0}',
        default='Pendulum-v0')
    parser.add_argument(
        '--random-seed',
        help='random seed for repeatability',
        default=35)
    parser.add_argument(
        '--max-episodes',
        help='max num of episodes to do while training',
        default=200)
    parser.add_argument(
        '--max-episode-len',
        help='max length of 1 episode',
        default=1000)
    parser.add_argument(
        '--render-env',
        help='render the gym env',
        action='store_true')
    parser.add_argument(
        '--save',
        help='save the actor network',
        action='store_true')
    parser.add_argument(
        '--header',
        help='prepend with a csv header',
        action='store_true')
    parser.add_argument(
        '--header-only',
        help='output a csv header and exit',
        action='store_true')

    parser.set_defaults(use_gym_monitor=False)
    parser.set_defaults(render_env=False)
    parser.set_defaults(header=False)
    parser.set_defaults(save=False)
    return parser
