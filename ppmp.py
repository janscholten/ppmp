# Jan Scholten, 2019
import tensorflow as tf
import numpy as np
import pprint as pp

import gym, sys, time, os
import tflearn
import argparse
import gym.spaces
from pathlib import Path

from parts_from_ddpg import Critic, ReplayBuffer, OUNoise
from parts_from_ppmp import KHeadActor, Predictor, Selector, ActionBuffer
from oracles import Oracle, LunarOracle

# Suppress tf and gym datatype warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
gym.logger.set_level(40)

def key_press(key, mod):
    global correction, human_is_done, human_sets_pause
    if key==0xff0d: human_is_done = True
    if key==32: human_sets_pause = not human_sets_pause
    if key==65361: a=-1 # left
    elif key==65363: a=1 # right
    else: 
        a = 0
        return
    correction = np.array([[a]])

def key_release(key, mod):
    global correction
    if key==65361: a=-1 # left
    elif key==65363: a=1 # right
    else: return
    if correction == a:
        correction = np.array([[0]])


def train(sess, env, args, actor, critic, oracle, actor_noise, action_selector, H):
    # Create and initialise objects
    seed = int(''.join([char for char in str(args['random_seed']) if char.isdigit()]))
    sess.run(tf.global_variables_initializer())
    actor.update_target_network()
    critic.update_target_network()
    replay_buffer = ReplayBuffer(int(args['buffer_size']), seed)
    replay_buffer2 = ActionBuffer(int(args['buffer2_size']), seed)

    # Create and initialise variables
    global correction, human_sets_pause
    correction = np.zeros((1, actor.a_dim))
    human_sets_pause = False
    human_is_done = False
    correction_cov = np.diag(np.tile(float(args['human_variance']), actor.a_dim))
    h_hat = 0
    ep_ave_max_q = 0
    timestamp = time.time()

    # Schedule feedback here
    fb_dimrate = -1/90 if (args['env']=='MountainCarContinuous-v0') else -1/40 # Pendulum
    fb_end = 100 if (args['env']=='MountainCarContinuous-v0') else 50 # Pendulum
    
    ac_hat_noise = 2*env.action_space.high*float(args['prediction_noise'])
    start_P = float(args['cold_samples'])
    start_F = float(args['filter_samples'])

    if args['algorithm']=='ppmp_human':
        env.reset(); env.render()
        env.unwrapped.viewer.window.on_key_press = key_press
        env.unwrapped.viewer.window.on_key_release = key_release
        human_sets_pause = True

    for i in range(int(args['max_episodes'])):
        s = env.reset()
        actor.change_current_head()
        ep_reward = 0
        gainbuffer = np.empty((int(args['max_episode_len']),actor.a_dim))*np.nan
        fb_buffer = np.zeros((int(args['max_episode_len']),actor.a_dim))

        for j in range(int(args['max_episode_len'])):
            if args['render_env'] and not human_is_done:  
                env.render()
                while human_sets_pause:
                    env.render(); time.sleep(0.1)

            ap, policy_var = actor.policy(np.reshape(s, (1, actor.s_dim)))
            ap += actor_noise()

            # Q-filter and scheduling
            if (replay_buffer.size()<start_P+replay_buffer2.size() or args['algorithm']=='pmp'):  # Predictor in standby
                policy = ap
            elif (replay_buffer.size()<start_F): # Always use predictor
                policy = H.predict(np.reshape(s, (1, actor.s_dim))) + ac_hat_noise*(np.random.random((1,actor.a_dim))-0.5)
            else: # Q-filter is active
                ac_hat = H.predict(np.reshape(s, (1, actor.s_dim))) + ac_hat_noise*(np.random.random((1,actor.a_dim))-0.5)
                predicted_better = critic.predict(np.reshape(s, (1, actor.s_dim)), ac_hat) > \
                                                critic.predict(np.reshape(s, (1, actor.s_dim)), ap)
                if predicted_better:
                    policy = ac_hat
                else:
                    policy = ap

            if 'human' in args['algorithm']:
                # correction is a global var
                if args['env']=='Pendulum-v0': # To flip the controls, which is more convenient
                    time.sleep(max(0,0.05 - time.time() + timestamp)); timestamp = time.time()
                    a, gain = action_selector.select(policy, policy_var, -correction, correction_cov)
                else:
                    a, gain = action_selector.select(policy, policy_var, correction, correction_cov)
                replay_buffer2.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)))
            else: # use oracle
                if np.random.random() < float(args['fb_amount'])*(min(1, max(0, fb_dimrate*(i-fb_end)))): #feedback
                    correction = oracle.correct(np.reshape(s, (1, actor.s_dim)), policy) 
                    if np.random.random() < float(args['error']):
                        correction *= -1
                    a, gain = action_selector.select(policy, policy_var, correction, correction_cov)
                    replay_buffer2.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)))
                else: # Feedback is not given
                    correction *= 0
                    a = policy
                    _, gain = action_selector.select(policy, policy_var, correction, correction_cov)

            # a = oracle.predict(np.reshape(s, (1, env.observation_space.shape[0]))) # Test the oracle
            a = np.clip(a, env.action_space.low, env.action_space.high) # Because noise and hf

            gainbuffer[j] = np.diag(gain)
            fb_buffer[j] = correction

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), \
                np.reshape(a, (actor.a_dim,)), \
                r, terminal, np.reshape(s2, (actor.s_dim,)), actor.get_current_head())

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

                actor.train(s_batch, np.reshape(grads,(-1, actor.a_dim*actor.k)))
                actor.update_target_network()
                critic.update_target_network()


            if replay_buffer2.size() > int(args['minibatch_size']):
                s_batch2, a_batch2 = \
                    replay_buffer2.sample_batch(int(args['minibatch_size']))
                H.train(s_batch2, a_batch2 - H.predict(s_batch2))

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
        print(csvformatter.format('Environment','Seed','Algorithm','Error','Episode','Reward','Gain','Feedback'))
        if args['header_only']:
            sys.exit(0)

    with tf.Session() as sess:
        env = gym.make(args['env'])
        seed = int(''.join([char for char in str(args['random_seed']) if char.isdigit()]))
        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.seed(seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = np.array([env.action_space.low, env.action_space.high])
        resolution = 2*float(args['resolution'])*action_bound.max(0)

        # if args['env']=='LunarLanderContinuous-v2': # If you like, use a PID controller
        #     oracle = LunarOracle(resolution = resolution)
        # else:
        oracle = Oracle(args['env'], action_dim, state_dim, resolution = resolution)

        actor = KHeadActor(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']), k=int(args['heads']), 
                             initial_variance=float(args['initial_variance']))

        H = Predictor(state_dim, action_dim, env.action_space.high,
                             2*float(args['actor_lr']),
                             int(args['minibatch_size']))

        critic = Critic(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        actor_noise = OUNoise(mu=np.zeros(action_dim))
        action_selector = Selector(scale=float(args['scale'])*2*action_bound.max(0), offset=2*resolution)

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, args, actor, critic, oracle, actor_noise, action_selector, H)

        if args['use_gym_monitor']:
            env.monitor.close()

        del oracle  # parent of a tf.session()
        del H

        if args['save']:
            try:
                actor.save(args['env'])
                print('Oracle saved: oracles/',args['env'], file=sys.stderr)
            except:
                actor.save(args['env']+time.asctime())
                print('Oracle existed: this one is saved as oracles/',args['env']+time.asctime(), file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Provide arguments for DDPG agent. Arguments are not checked for mutual exclusiveness.')

    # agent parameters
    parser.add_argument('--algorithm', help='Either ppmp, ppmp_human or pmp', default='ppmp')
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.002)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.003)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for training the actor', default=64)
    parser.add_argument('--heads', help='number of heads for multihead', default=10)
    parser.add_argument('--initial-variance', help='Initial variance of actor network ouput layer (between heads)', default=0.001)

    # predictor/selector
    parser.add_argument('--buffer2-size', help='max size of the replay buffer', default=1600)
    parser.add_argument('--cold-samples', help='The amount of samples collected without prediction', default=1500)
    parser.add_argument('--filter-samples', help='The amount of samples before Q-filter reliability', default=4000)
    parser.add_argument('--prediction-noise', help='Variance as fraction of action space', default=0.025)
    parser.add_argument('--human-variance', help='Human feedback variance (action domain)', default=1e-8)
    parser.add_argument('--resolution', help='Hysteresis in the correction, fraction of action space', default=0.125)
    parser.add_argument('--scale', help='Fraction of action space', default=0.5)

    # oracle
    parser.add_argument('--error', help='Erroneous feedback ERROR fraction of time', default=0)
    parser.add_argument('--fb-amount', help='Scale amount of feedback', default=0.3)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=200)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./gym_results/gym_ddpg')
    parser.add_argument('--save', help='save the actor network', action='store_true')
    parser.add_argument('--header', help='prepend with a csv header', action='store_true')
    parser.add_argument('--header-only', help='output a csv header and exit', action='store_true')

    parser.set_defaults(use_gym_monitor=False)
    parser.set_defaults(render_env=False)
    parser.set_defaults(header=False)
    parser.set_defaults(save=False)

    args = vars(parser.parse_args())
    args['render_env'] = True if args['algorithm']=='ppmp_human' else args['render_env']
    args['header'] = True if args['header_only']==True else args['header']
    
    if not args['header_only']:
        pp.pprint(args, stream=sys.stderr)
        print('\n', file=sys.stderr)

    main(args)
