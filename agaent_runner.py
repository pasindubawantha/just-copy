import gym
import gym_gvgai
import time
import sso as sso_class
from pprint import pprint

# import your YourAgent and assigne it to Agent Agent
import AgentQLearningWithTables
Agent = AgentQLearningWithTables.AgentQLearningWithTables()

# games = ["gvgai-aliens"]
games = ["gvgai-aliens","gvgai-angelsdemons","gvgai-assemblyline","gvgai-avoidgeorge","gvgai-bait","gvgai-beltmanager","gvgai-blacksmoke","gvgai-boloadventures","gvgai-bomber","gvgai-bomberman"] # add 10 games

for game in games:
    print("Playing game "+game)
    #run level 0,1,2 for training for 5 minutes
    training_time_limit = 5*60
    training_start_time = time.time()
    elapsed_timer = 0

    #run levels 0,1,2 in sequence
    for i in range(1):
        print("Playing "+game+" level "+str(i))

        env = gym.make(game + "-lvl" + str(i) + "-v0")
        observation = env.reset()

        sso = sso_class.sso()
        sso.availableActions = [x for x in range(env.action_space.n)]#just send a number
        sso.observation = observation
        Agent.init(sso, elapsed_timer)

        for i in range (3):
            
            print(env.observation_space.shape)
            
            env.render()

            sso.gameTick += 1
            observation, reward, done, info = env.step(Agent.act(sso, elapsed_timer))
            #reward is -1.0 when player loses
            #observation is visual screen
            sso.gameScore += reward
            sso.observation = observation

            

            elapsed_timer = time.time() - training_start_time

            if i == 999:
                next_level = Agent.result(sso, elapsed_timer)
                elapsed_timer = time.time() - training_start_time
            if done:
                if reward != -1:
                    sso.gameWinner = "Player"
                next_level = Agent.result(sso, elapsed_timer)
                elapsed_timer = time.time() - training_start_time
                break
            if elapsed_timer > training_time_limit:
                break


        elapsed_timer = time.time() - training_start_time
        #close render window
        env.close()
        if elapsed_timer > training_time_limit:
            break

    #if 5 mins is not over run level of choice from 0,1,2
    # elapsed_timer = time.time() - training_start_time
    # while elapsed_timer < training_time_limit:
    #     env = gym.make(game + "-lvl" + str(next_level) + "-v0")


    #     if done:
    #         next_level = Agent.result(sso, elapsed_timer)
    #         elapsed_timer = time.time() - training_start_time
    #         break
    #     if elapsed_timer > training_time_limit:
    #         break

            
            
            
    
    #run level 3,4 for validation