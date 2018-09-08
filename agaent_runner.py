import gym
import gym_gvgai
import time
import sso as sso_class
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# import your YourAgent and assigne it to Agent Agent

#from YourAgent import YourAgent as Agent
from AgentQLearningWithTables import AgentQLearningWithTables

games = ["gvgai-aliens","gvgai-angelsdemons","gvgai-assemblyline","gvgai-avoidgeorge","gvgai-bait","gvgai-beltmanager","gvgai-blacksmoke","gvgai-boloadventures","gvgai-bomber","gvgai-bomberman"] # add 10 games

fig, ax = plt.subplots()
xs = [0]
ys = [0]

ax.set_xlabel("Game Ticks")
ax.set_ylabel("Score")
def animate(i):
    line, = ax.plot(xs, ys)
    return line,

ani = animation.FuncAnimation(fig, animate, interval=2)
plt.show(block=False)

for game in games:
    ax.clear()
    print("Playing game "+game)
    #run level 0,1,2 for training for 5 minutes
    training_time_limit = 5*60
    training_start_time = time.time()
    elapsed_timer = 0
    Agent = AgentQLearningWithTables()

    #run levels 0,1,2 in sequence
    for i in range(3):
        print("Playing "+game+" level "+str(i))
        ax.set_title("Game :"+game+" Level :"+str(i))

        env = gym.make(game + "-lvl" + str(i) + "-v0")
        observation = env.reset()

        sso = sso_class.sso()
        sso.availableActions = [x for x in range(env.action_space.n)]#just send a number
        sso.observation = observation
        Agent.init(sso, elapsed_timer)
        xs = [0]
        ys = [0]
        
        for j in range (1000):
            
            env.render()

            sso.gameTick += 1
            print("Running "+game+" lvl :"+str(i)+" Tick :"+str(sso.gameTick)+" Score :"+str(sso.gameScore))
            observation, reward, done, info = env.step(Agent.act(sso, elapsed_timer))
            #reward is -1.0 when player loses
            #observation is visual screen
            sso.gameScore += reward
            sso.observation = observation
            
            xs.append(sso.gameTick)
            ys.append(sso.gameScore)

            elapsed_timer = time.time() - training_start_time

            if j == 999:
                next_level = Agent.result(sso, elapsed_timer)
                elapsed_timer = time.time() - training_start_time
            if done:
                if reward != -1:
                    sso.gameWinner = "Player"
                next_level = Agent.result(sso, elapsed_timer)
                elapsed_timer = time.time() - training_start_time
                break
            if elapsed_timer > training_time_limit:
                next_level = Agent.result(sso, elapsed_timer)
                break


        elapsed_timer = time.time() - training_start_time
        #close render window
        env.close()
        if elapsed_timer > training_time_limit:
            break

    # elapsed_timer = time.time() - training_start_time
    # while elapsed_timer < training_time_limit:

    
    #run level 3,4 for validation