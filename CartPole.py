import os
import gym
import numpy as np
import tensorflow as tf

#############################
# Create the Neural Network #
#############################

learning_rate = 0.01

n_inputs = 4    # == env.obersvation_space.shape[0]
n_hidden = 4
n_outputs = 1

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.sigmoid(logits)

p_left_and_right = tf.concat(axis=1, values=[outputs, 1-outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

# Policy gradients for optimization
y = 1. - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

###############################
# Create the reward functions #
###############################

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

##########################
# Time to play the game! #
##########################

checkpoint_path = "./policy_net.ckpt"
checkpoint_iteration_path = checkpoint_path + ".iteration"
model_path = "./policy_net"

env = gym.make('CartPole-v0') 
n_games_per_update = 10
n_max_steps = 10000
n_iterations = 250*80
save_iteractions = 10
discount_rate = 0.95

with tf.Session() as sess:
    if os.path.isfile(checkpoint_iteration_path):
        # if the checkpoint file exists, restore the model and load the iteration number
        with open(checkpoint_iteration_path, "rb") as f:
            start_iteration = int(f.read())
        print("Restoring previous model")
        saver.restore(sess, checkpoint_path)
    else:
        start_iteration = 0
        sess.run(init)

    init.run()
    for iteration in range(start_iteration, start_iteration + n_iterations):
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                if game == 0:
                    env.render()
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            if game == 0:
                print("Iteration {}, reward {}".format(iteration, len(current_rewards)))
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
        feed_dict = {}
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients

        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iteractions == 0:
            print("Checkpoint saved")
            saver.save(sess, checkpoint_path)
            with open(checkpoint_iteration_path, "wb") as f:
                f.write(b"%d" % (iteration + 1))

    saver.save(sess, model_path)

env.close()