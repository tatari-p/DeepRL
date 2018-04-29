from ops import replay_DQN
import random
from model import DuelingDQN
from utils import AirSimEnv
from AirSimClient import *
import tensorflow as tf


client = MultirotorClient()
env = AirSimEnv(client, 1.)

model_name = "DuelingDQN"
save_path = "./Models/{}.ckpt".format(model_name)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

buffer_size = 10000
train_size = 30
num_train_per_epoch = 20
save_period = 10

training = True
write_meta_graph = True

with tf.Session(config=config) as sess:
    main = DuelingDQN("main")
    target = DuelingDQN("target")

    main.build_model((128, 72))
    target.build_model((128, 72))
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    max_epochs = 5000

    env.set_to_drive()

    train_buffer = []
    loss = 0

    for epoch in range(max_epochs):
        l_rate = 0.0001*0.9**(epoch/50.)

        while True:
            env.target = np.array([np.random.uniform(-150, 150),
                                   np.random.uniform(-150, 150),
                                   -10.], dtype="float")

            # temporary hard coding solution as AirSim has infinite rotation bug when target angle is in specific range
            if 0 <= np.arctan2(env.target[1], env.target[0]) <= 6*np.pi/8\
                    or -6*np.pi/8 <= np.arctan2(env.target[1], env.target[0]) <= 0:
                break

        env.goal_range = 5.
        env.head_to_target()

        state, reward, done = env.get_state()
        e = 0.3 if training else 0

        episode_step = 0
        episode_reward = 0

        while True:
            action = np.argmax(main.predict(np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0), np.expand_dims(state[2], axis=0))) if np.random.uniform(0, 1) > e else np.random.randint(7)
            env.drive(action)
            next_state, reward, done = env.get_state()

            if episode_step % 5 == 0:
                p = env.get_position()

            episode_step += 1
            episode_reward += reward

            # prevent Drone from infinite straying
            if episode_step > 200:
                done = True

            train_buffer.append((state, action, reward, done, next_state))

            if len(train_buffer) > buffer_size:
                train_buffer.pop(0)

            if done:
                break

            state = next_state

        if len(train_buffer) > buffer_size/20:

            for i in range(num_train_per_epoch):
                batch = random.sample(train_buffer, train_size)
                loss = replay_DQN(main, target, batch, l_rate=l_rate)

            target.update("main")

            if epoch % save_period == 0:
                saver.save(sess, save_path, write_meta_graph=write_meta_graph)

                if write_meta_graph:
                    write_meta_graph = False

                break
