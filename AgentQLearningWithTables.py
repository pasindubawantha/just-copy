import tensorflow as tf
#import tensorflow.contrib.slim as slim
import numpy as np
from helpers import *
from AC_Network import AC_Network 
from Worker import Worker
import os

class AgentQLearningWithTables :
    def __init__(self):
        self.ran_first_time = False
        self.train_steps = 5
        tf.reset_default_graph()
        
   
    
    def init(self, sso, elapsed_timer):
        if not self.ran_first_time :
            #STAT OF FIRST INIT
            self.prv_observation = None
            self.prv_score = 0
            self.prv_action = 0
            self.ran_first_time = True

            self.gamma = .99 # discount rate for advantage estimation and reward discounting

            self.s_size = sso.observation.shape[0] * sso.observation.shape[1] * sso.observation.shape[2] # Observations are greyscale frames of 84 * 84 * 1
            self.a_size = len(sso.availableActions) # Agent can move Left, Right, or Fire
            self.s_shape = sso.observation.shape

            self.model_path = '.tensor_flow/model'
        
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
                
            # #Create a directory to save episode playback gifs to
            # if not os.path.exists('./frames'):
            #     os.makedirs('./frames')
        
            tf.reset_default_graph()

            with tf.device("/cpu:0"): 
                self.global_episodes = 0
                self.trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
                self.master_network = AC_Network(self.s_size, self.a_size, 'global', None, self.s_shape) # Generate global network
                
                self.saver = tf.train.Saver(max_to_keep=5)
                self.coord = tf.train.Coordinator()
                self.session = tf.Session()
                self.session.run(tf.global_variables_initializer())

            game_lvl = 1
            self.Worker = Worker(game_lvl, self.s_size, self.a_size, self.trainer, self.model_path, self.s_shape)
            self.session.run(tf.global_variables_initializer())
            #END OF FIRST INIT
        
        

        self.prv_observation = sso.observation
        print("Agent Init ran")


    def act(self, sso, elapsed_timer):
        reward = sso.gameScore - self.prv_score

        if sso.gameTick == 1:
            action = self.Worker.work(self.gamma, self.session, self.coord, self.saver, sso, reward, self.prv_observation , self.prv_action, elapsed_timer, False, False, True)
        else:
            if sso.gameTick % self.train_steps == 0: #train
                action = self.Worker.work(self.gamma,self.session, self.coord,self.saver, sso,reward, self.prv_observation, self.prv_action ,elapsed_timer, True, False, False)
                self.global_episodes += 1
            else: #collect experiance
                action = self.Worker.work(self.gamma,self.session, self.coord,self.saver, sso,reward, self.prv_observation, self.prv_action ,elapsed_timer, False, False, False)
        

        self.prv_observation = sso.observation
        self.prv_score = sso.gameScore
        self.prv_action = action
        
        return sso.availableActions[action]

 
    def result(self, sso, elapsed_timer):
        reward = sso.gameScore - self.prv_score
        self.Worker.work(self.gamma,self.session, self.coord,self.saver, sso,reward, self.prv_observation ,self.prv_action ,elapsed_timer, True, True, False)
        print("Agent Result ran")
        return 0