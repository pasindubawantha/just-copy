import tensorflow as tf
#import tensorflow.contrib.slim as slim
import numpy as np
from helpers import *
from AC_Network import AC_Network 


class Worker():
    def __init__(self,name,s_size,a_size,trainer,model_path,s_shape):
        self.name = "worker_" + str(name)
        self.number = name        
        self.trainer = trainer

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer,s_shape)
        self.update_local_ops = update_target_graph('global',self.name)        
        
        self.actions = self.actions = np.identity(a_size,dtype=bool).tolist()

        
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
        
    def visualize(self):

        plt.ylabel(str(conv1.shape))
        plt.plot(conv1)
        
    
    def work(self,gamma,sess,coord,saver, sso,reward, prv_observation,prv_action,elapsed_timer,do_train,game_over, first_run):

        # print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            if first_run:
                sess.run(self.update_local_ops)
                self.episode_buffer = []

            s = process_frame(prv_observation)
            a = prv_action
            r = reward

            s1 = process_frame(sso.observation)

            
            rnn_state = self.local_AC.state_init
            self.batch_rnn_state = rnn_state

            
            #Take an action using probabilities from policy network output.
            a_dist,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                feed_dict={self.local_AC.inputs:[s1],
                self.local_AC.state_in[0]:rnn_state[0],
                self.local_AC.state_in[1]:rnn_state[1]})
            a1 = np.random.choice(a_dist[0],p=a_dist[0])
            a1 = np.argmax(a_dist == a1)

            d = game_over

                
            self.episode_buffer.append([s,a,r,s1,d,v[0,0]])
            
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
                v_l,p_l,e_l,g_n,v_n = self.train(self.episode_buffer,sess,gamma,0.0)
                #self.train(self.episode_buffer,sess,gamma,0.0)
                self.episode_buffer = []
                sess.run(self.update_local_ops)
            
            return a1