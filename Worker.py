import tensorflow as tf
#import tensorflow.contrib.slim as slim
import numpy as np
from helpers import *
from AC_Network import AC_Network 


class Worker():
    def __init__(self,name,s_size,a_size,trainer,model_path,s_shape,session):
        self.name = "worker_" + str(name)       
        self.trainer = trainer

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer,s_shape)
        self.update_local_ops = update_target_graph('global',self.name)       
        self.episode_count = 0
        self.summary_writer = tf.summary.FileWriter("tensor_flow/train_"+str(name), session.graph) 
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]

        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:self.batch_rnn_state[0],
            self.local_AC.state_in[1]:self.batch_rnn_state[1]}

        v_l,p_l,e_l,g_n,v_n, self.batch_rnn_state,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.state_out,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
            
    
    def work(self, gamma, session, prv_observation, prv_action, reward, observation, game_over, do_train, first_run):

        # print ("Starting worker " + str(self.number))
        with session.as_default(), session.graph.as_default():                 
            if first_run:
                session.run(self.update_local_ops)
                self.episode_buffer = []

            prv_observation = process_frame(prv_observation)
            observation = process_frame(observation)

            
            rnn_state = self.local_AC.state_init
            self.batch_rnn_state = rnn_state

            
            #Take an action using probabilities from policy network output.
            action_distribution,value,rnn_state = session.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                feed_dict={self.local_AC.inputs:[observation],
                self.local_AC.state_in[0]:rnn_state[0],
                self.local_AC.state_in[1]:rnn_state[1]})
            action = np.random.choice(action_distribution[0],p=action_distribution[0])
            action = np.argmax(action_distribution == action)
                
            self.episode_buffer.append([prv_observation, prv_action, reward, observation, game_over, value[0,0]])
            
            # If the episode hasn't ended, but the experience buffer is full, then we
            # make an update step using that experience rollout.
            if do_train:
                # Since we don't know what the true final return is, we "bootstrap" from our current
                # value estimation.
                # v1 = sess.run(self.local_AC.value, 
                #     feed_dict={self.local_AC.inputs:[s],
                #     self.local_AC.state_in[0]:rnn_state[0],
                #     self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                
                #v_l,p_l,e_l,g_n,v_n = self.train(self.episode_buffer,sess,gamma,v)
                v_l,p_l,e_l,g_n,v_n = self.train(self.episode_buffer, session, gamma, 0.0)
                self.episode_buffer = []
                session.run(self.update_local_ops)

                
                self.episode_count += 1
                # summary = tf.Summary()
                # summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                # summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                # summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                # summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                # summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n)) 
                # self.summary_writer.add_summary(summary, self.episode_count)
                self.summary_writer.flush()                              
            
            return action