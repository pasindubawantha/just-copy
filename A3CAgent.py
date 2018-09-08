import tensorflow as tf
import numpy as np
from helpers import *
from AC_Network import AC_Network 
from Worker import Worker
import os

class A3CAgent:
    def __init__(self):
        self.first_init = True
        self.roll_out_steps = 5
        self.this_level = 0
        self.worker_number = 0
        tf.reset_default_graph()
    
    def init(self, sso, elapsed_timer):
        if self.first_init :
            #STAT OF FIRST INIT
            self.first_init = False
            
            self.roll_out_steps =len(sso.availableActions)*3
            self.gamma = .99 # discount rate for advantage estimation and reward discounting

            self.prv_observation = None
            self.prv_score = 0
            self.prv_action = 0

            self.s_size = sso.observation.shape[0] * sso.observation.shape[1] * sso.observation.shape[2] # Observations are greyscale frames of 84 * 84 * 1
            self.a_size = len(sso.availableActions) # Agent can move Left, Right, or Fire
            self.s_shape = sso.observation.shape

            self.model_path = 'tensor_flow/model'
        
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)

            with tf.device("/cpu:0"): 
                self.trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
                self.master_network = AC_Network(self.s_size, self.a_size, 'global', None, self.s_shape) # Generate global network
                
                self.session = tf.Session()
                self.session.run(tf.global_variables_initializer())

                
                
            print("A3CAgent Init ran for first time")
            #END OF FIRST INIT

        #Start new worker for level
        self.worker_number += 1
        print("Worker starting : "+str(self.worker_number))
        self.Worker = Worker(self.worker_number, self.s_size, self.a_size, self.trainer, self.model_path, self.s_shape,self.session)
        
        #Initialize Workers local variables
        self.session.run(tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "worker_"+str(self.worker_number))
        ))
        self.prv_observation = sso.observation
        print("A3CAgent Init ran")


    def act(self, sso, elapsed_timer):
        reward = sso.gameScore - self.prv_score

        if sso.gameTick == 1:# First action of the level
            action = self.Worker.work(self.gamma, self.session, self.prv_observation , self.prv_action, reward, sso.observation, game_over=False, do_train=False, first_run=True)
        else:
            if sso.gameTick % self.roll_out_steps == 0: #Role back onexperiance gained and Train action
                action = self.Worker.work(self.gamma, self.session, self.prv_observation , self.prv_action, reward, sso.observation, game_over=False, do_train=True, first_run=False)
            else: #Collect experiance action
                action = self.Worker.work(self.gamma, self.session, self.prv_observation , self.prv_action, reward, sso.observation, game_over=False, do_train=False, first_run=False)
        

        self.prv_observation = sso.observation
        self.prv_score = sso.gameScore
        self.prv_action = action
        
        return sso.availableActions[action]

 
    def result(self, sso, elapsed_timer):
        reward = sso.gameScore - self.prv_score
        self.Worker.work(self.gamma, self.session, self.prv_observation , self.prv_action, reward, sso.observation, game_over=True, do_train=True, first_run=False)
        
        self.this_level += 1
        if self.this_level > 2:
            self.this_level = 0 
        print("A3CAgent Result ran")
        return self.this_level