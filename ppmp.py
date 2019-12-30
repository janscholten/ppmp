# Jan Scholten, 2019
import tensorflow as tf
import numpy as np
import pprint as pp

import gym
import sys
import time
import os
import gym.spaces

from parts_from_ddpg import Critic, ReplayBuffer, OUNoise
from parts_from_ppmp import KHeadActor, Predictor, Selector, ActionBuffer
from oracles import Oracle
from utils import get_ppmp_argparser

# Suppress tf and gym datatype warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
gym.logger.set_level(40)


def key_press(key, mod):
    global correction, human_is_done, human_sets_pause
    if key == 0xff0d:
        human_is_done = True
    if key == 32:
        human_sets_pause = not human_sets_pause
    if key == 65361:
        a = -1  # left
    elif key == 65363:
        a = 1  # right
    else:
        a = 0
        return
    correction = np.array([[a]])


def key_release(key, mod):
    global correction
    if key == 65361:
        a = -1  # left
    elif key == 65363:
        a = 1  # right
    else:
        return
    if correction == a:
        correction = np.array([[0]])


def train(
    sess, env, args, actor, critic, oracle, ou_noise, selector, predictor):
    # Create and initialise objects
    seed = int(
        ''.join([char for char in str(args['random_seed']) if char.isdigit()]))
    sess.run(tf.global_variables_initializer())
    actor.update_target_network()
    critic.update_target_network()
    replay_buffer = ReplayBuffer(int(args['buffer_size']), seed)
    predictor_buffer = ActionBuffer(int(args['buffer2_size']), seed)

    # Create and initialise variables
    global correction, human_sets_pause
    correction = np.zeros((1, actor.a_dim))
    human_sets_pause = False
    human_is_done = False
    correction_cov = np.diag(
        np.tile(float(args['human_variance']), actor.a_dim))
    timestamp = time.time()

    # Schedule feedback here
    fb_dimrate = -1 / \
        90 if (args['env'] == 'MountainCarContinuous-v0') else - \
        1/40  # Pendulum
    fb_end = 100 if (
        args['env'] == 'MountainCarContinuous-v0') else 50  # Pendulum

    predictor_noise = 2*env.action_space.high*float(args['prediction_noise'])
    start_predictor_training = float(args['cold_samples'])
    start_q_filter = float(args['filter_samples'])

    if args['algorithm'] == 'ppmp_human':
        env.reset()
        env.render()
        env.unwrapped.viewer.window.on_key_press = key_press
        env.unwrapped.viewer.window.on_key_release = key_release
        human_sets_pause = True

    for i in range(int(args['max_episodes'])):
        state = env.reset()
        actor.change_current_head()
        ep_reward = 0
        gainbuffer = np.empty(
            (int(args['max_episode_len']), actor.a_dim))*np.nan
        fb_buffer = np.zeros((int(args['max_episode_len']), actor.a_dim))

        for j in range(int(args['max_episode_len'])):
            if args['render_env'] and not human_is_done:
                env.render()
                while human_sets_pause:
                    env.render()
                    time.sleep(0.1)

            ap, policy_var = actor.policy(np.reshape(state, (1, actor.s_dim)))
            ap += ou_noise()

            # Scheduling actions:
            
            # Predictor in standby
            if replay_buffer.size() < start_predictor_training+predictor_buffer.size():
                policy = ap
            elif args['algorithm'] == 'pmp':
                policy = ap

            # Predictor active
            elif (replay_buffer.size() < start_q_filter):
                policy = predictor.predict(np.reshape(state, (1, actor.s_dim))) + \
                    predictor_noise*(np.random.random((1, actor.a_dim))-0.5)
            else:  # Q-filter is active
                ac_hat = predictor.predict(np.reshape(state, (1, actor.s_dim))) + \
                    predictor_noise*(np.random.random((1, actor.a_dim))-0.5)
                predicted_better = critic.predict(np.reshape(state, (1, actor.s_dim)), ac_hat) > \
                    critic.predict(np.reshape(state, (1, actor.s_dim)), ap)
                if predicted_better:
                    policy = ac_hat
                else:
                    policy = ap

            if 'human' in args['algorithm']:
                if args['env'] == 'Pendulum-v0':  # Flipping controls 
                    time.sleep(max(0, 0.05 - time.time() + timestamp))
                    timestamp = time.time()
                    correction = -correction

                a, gain = selector.select(
                    policy, policy_var, correction, correction_cov)
                predictor_buffer.add(np.reshape(
                    state, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)))
            else:  # use oracle
                fb_rate = float(args['fb_amount'])*(min(1, max(0, fb_dimrate*(i-fb_end))))
                if np.random.random() < fb_rate:
                    correction = oracle.correct(
                        np.reshape(state, (1, actor.s_dim)), policy)
                    if np.random.random() < float(args['error']):
                        correction *= -1
                    a, gain = selector.select(
                        policy, policy_var, correction, correction_cov)
                    predictor_buffer.add(np.reshape(
                        state, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)))
                else:  # Feedback is not given
                    correction *= 0
                    a = policy
                    _, gain = selector.select(
                        policy, policy_var, correction, correction_cov)

            # The oracle : oracle.predict(
            #   np.reshape(state, (1, env.observation_space.shape[0])))
            a = np.clip(a, env.action_space.low, env.action_space.high)

            gainbuffer[j] = np.diag(gain)
            fb_buffer[j] = correction

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(state, (actor.s_dim,)),
                              np.reshape(a, (actor.a_dim,)),
                              r, terminal, np.reshape(
                                  s2, (actor.s_dim,)), actor.get_current_head())

            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch, j_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.policy_target(s2_batch, j_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                # Update actor
                a_outs = actor.predict(s_batch)
                grads = np.array(critic.action_gradients(
                    s_batch.repeat(actor.k, axis=0), a_outs.reshape(-1, actor.a_dim))).squeeze()

                actor.train(s_batch, np.reshape(
                    grads, (-1, actor.a_dim*actor.k)))
                actor.update_target_network()
                critic.update_target_network()

            if predictor_buffer.size() > int(args['minibatch_size']):
                s_batch2, a_batch2 = \
                    predictor_buffer.sample_batch(int(args['minibatch_size']))
                predictor.train(s_batch2, a_batch2 -
                                predictor.predict(s_batch2))

            s = s2
            ep_reward += r
            if terminal:
                break
        fbtot = np.count_nonzero(fb_buffer)/fb_buffer.shape[1]/(j+1)
        print(csvformatter.format(args['env'], args['random_seed'],
                                  'PPMP',
                                  args['error'],
                                  i,
                                  str(int(ep_reward)),
                                  '{:1.4f}'.format(np.nanmean(gainbuffer)),
                                  '{:1.4f}'.format(fbtot)))


def main(args):
    global csvformatter
    csvformatter = '{:<25}{:<11}{:<10}{:<8}{:<8}{:<8}{:<8}{}'

    if args['header']:
        print(csvformatter.format('Environment', 'Seed', 'Algorithm',
                                  'Error', 'Episode', 'Reward', 'Gain', 'Feedback'))
        if args['header_only']:
            sys.exit(0)

    with tf.Session() as sess:
        env = gym.make(args['env'])
        seed = int(
            ''.join([char for char in str(args['random_seed']) if char.isdigit()]))
        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.seed(seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = np.array([env.action_space.low, env.action_space.high])
        resolution = 2*float(args['resolution'])*action_bound.max(0)

        oracle = Oracle(args['env'], action_dim,
                        state_dim, resolution=resolution)

        actor = KHeadActor(sess, state_dim, action_dim, action_bound,
                           float(args['actor_lr']), float(args['tau']),
                           int(args['minibatch_size']), k=int(args['heads']),
                           initial_variance=float(args['initial_variance']))

        predictor = Predictor(state_dim, action_dim, env.action_space.high,
                              2*float(args['actor_lr']),
                              int(args['minibatch_size']))

        critic = Critic(sess, state_dim, action_dim,
                        float(args['critic_lr']), float(args['tau']),
                        float(args['gamma']),
                        actor.get_num_trainable_vars())

        ou_noise = OUNoise(mu=np.zeros(action_dim))
        selector = Selector(scale=float(
            args['scale'])*2*action_bound.max(0), offset=2*resolution)

        train(sess, env, args, actor, critic, oracle,
              ou_noise, selector, predictor)

        del oracle  # parent of a tf.session()
        del predictor

        if args['save']:
            try:
                actor.save(args['env'])
                print('Oracle saved: oracles/', args['env'], file=sys.stderr)
            except FileExistsError:
                actor.save(args['env']+time.asctime())
                print('Oracle existed: this one is saved as oracles/',
                      args['env']+time.asctime(), file=sys.stderr)


if __name__ == '__main__':
    args = vars(get_ppmp_argparser().parse_args())
    if args['algorithm'] == 'ppmp_human':
        args['render_env'] = True
    args['header'] = True if args['header_only'] else args['header']

    if not args['header_only']:
        pp.pprint(args, stream=sys.stderr)
        print('\n', file=sys.stderr)

    main(args)
