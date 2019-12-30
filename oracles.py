import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants
from pathlib import Path

# Shared code for both oracle classes


def correct(should, action, resolution):
    """ Binary signal thus in {-1, 0, 1}"""
    diff = should - action
    correction = np.sign(diff)
    correction *= np.ceil(np.abs(diff) - resolution)
    return np.sign(correction)


class Oracle(object):
    """Will restore a tf graph from disk, 
    and provides a policy() method which returns the optimal action"""

    def __init__(self, env, action_dim, state_dim, resolution):
        dir = Path('oracles/' + env)
        assert dir.is_dir(), 'There was no oracle found'
        self.graph = tf.Graph()
        # with tf.Session(graph=self.graph) as self.sess: # cumbersome
        # but now we need 'del oracle'
        self.sess = tf.Session(graph=self.graph)
        tf.saved_model.loader.load(
            self.sess, [tag_constants.SERVING], "./oracles/"+env)
        self.scaled_out_ph = self.graph.get_tensor_by_name('clip_by_value:0')
        self.inputs_ph = self.graph.get_tensor_by_name('InputData/X:0')
        self.resolution = resolution
        self.k = self.sess.run(self.scaled_out_ph,
                               feed_dict={self.inputs_ph: np.random.random((1, state_dim))}).size/action_dim

    def predict(self, states):
        return np.array(np.hsplit(self.sess.run(
            self.scaled_out_ph, feed_dict={self.inputs_ph: states}), self.k)).mean(0)

    def correct(self, state, action):
        should = self.predict(state)
        return correct(should, action, self.resolution)

    def __del__(self):
        self.sess.close()


class LunarOracle(object):
    """This one is a bit crippled"""

    def __init__(self, resolution):
        self.resolution = resolution

    def predict_good(self, s):
        s = s.squeeze()  # there are no batches here anyway
        # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
        angle_targ = s[0]*0.5 + s[2]*1.0
        if angle_targ > 0.4:
            angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4:
            angle_targ = -0.4
        # target y should be proporional to horizontal offset
        hover_targ = 0.55*np.abs(s[0])

        # PID controller: s[4] angle, s[5] angularSpeed
        angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
        #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

        # PID controller: s[1] vertical coordinate s[3] vertical speed
        hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
        #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

        if s[6] or s[7]:  # legs have contact
            angle_todo = 0
            # override to reduce fall speed, that's all we need after contact
            hover_todo = -(s[3])*0.5

        a = np.array([hover_todo*20 - 1, -angle_todo*20])
        a = np.clip(a, -1, +1)
        return np.array([a])

    def predict(self, s):
        s = s.squeeze()  # there are no batches here anyway
        # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
        angle_targ = s[0]*0.5 + s[2]*1.0
        if angle_targ > 0.4:
            angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4:
            angle_targ = -0.4
        # target y should be proporional to horizontal offset
        hover_targ = 0.55*np.abs(s[0])

        # PID controller: s[4] angle, s[5] angularSpeed
        angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
        #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

        # PID controller: s[1] vertical coordinate s[3] vertical speed
        hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
        #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

        if s[6] or s[7]:  # legs have contact
            angle_todo = 0
            # override to reduce fall speed, that's all we need after contact
            hover_todo = -(s[3])*0.5

        a = np.array([np.fmax(hover_todo*20 - 1, .2), -angle_todo*20])
        a = np.clip(a, -1, +1)
        return np.array([a])

    def predict2(self, s):
        s = s.squeeze()  # there are no batches here anyway
        # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
        angle_targ = s[0]*0 + s[2]*1.0
        if angle_targ > 0.4:
            angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4:
            angle_targ = -0.4
        # target y should be proporional to horizontal offset
        hover_targ = 0.*np.abs(s[0])

        # PID controller: s[4] angle, s[5] angularSpeed
        angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
        #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

        # PID controller: s[1] vertical coordinate s[3] vertical speed
        hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
        #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

        if s[6] or s[7]:  # legs have contact
            angle_todo = 0
            # override to reduce fall speed, that's all we need after contact
            hover_todo = -(s[3])*0.5

        a = np.array([hover_todo*20 - 1, -angle_todo*20])
        a = np.clip(a, -1, +1)
        return np.array([a])

    def correct(self, state, action):
        should = self.predict(state)
        return correct(should, action, self.resolution)
