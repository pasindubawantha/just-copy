import gym
import gym_gvgai
import time
import sso as sso_class
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from A3CAgent import A3CAgent
plt.style.use('dark_background')

games = ["gvgai-aliens","gvgai-angelsdemons","gvgai-assemblyline","gvgai-avoidgeorge","gvgai-bait","gvgai-beltmanager","gvgai-blacksmoke","gvgai-boloadventures","gvgai-bomber","gvgai-bomberman"] # add 10 games
# games = ["gvgai-aliens","gvgai-angelsdemons"] # add 10 games


results_file = open("results.txt", "w")
results_file.write("") # empty file
results_file.close()

results_file = open("results.txt", "a")

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
    results_file.write("Playing game "+game+"\n")
    #run level 0,1,2 for training for 5 minutes
    training_time_limit = 5*60
    training_start_time = time.time()
    elapsed_timer = 0
    Agent = A3CAgent()
    next_level = 0
    #run levels 0,1,2 in sequence
    for i in range(3):
        print("Training : Playing "+game+" level "+str(i))
        ax.set_title("Training : Game :"+game+" Level :"+str(i))
        results_file.write("Training : level "+str(i)+" Score : ")

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
            # print("Training : Running "+game+" lvl :"+str(i)+" Tick :"+str(sso.gameTick)+" Score :"+str(sso.gameScore))
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
                results_file.write(str(sso.gameScore)+"\n")
                print("Training : level "+str(i)+" Score : "+str(sso.gameScore))
                #close render window
                env.close()
            if done:
                if reward != -1:
                    sso.gameWinner = "Player"
                next_level = Agent.result(sso, elapsed_timer)
                elapsed_timer = time.time() - training_start_time
                results_file.write(str(sso.gameScore)+"\n")
                print("Training : level "+str(i)+" Score : "+str(sso.gameScore))
                #close render window
                env.close()
                break
            if elapsed_timer > training_time_limit:
                next_level = Agent.result(sso, elapsed_timer)
                results_file.write(str(sso.gameScore)+"\n")
                print("Training : level "+str(i)+" Score : "+str(sso.gameScore))
                #close render window
                env.close()
                break


        elapsed_timer = time.time() - training_start_time
        if elapsed_timer > training_time_limit:
            break

    elapsed_timer = time.time() - training_start_time
    while elapsed_timer < training_time_limit and False:
        if next_level > 2 and next_level < 0:
            next_level = 0
        print("Training : Playing "+game+" level "+str(next_level))
        ax.set_title("Training : Game :"+game+" Level :"+str(next_level))
        results_file.write("Training : level "+str(next_level)+" Score : ")

        env = gym.make(game + "-lvl" + str(next_level) + "-v0")
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
            #print("Training : Running "+game+" lvl :"+str(next_level)+" Tick :"+str(sso.gameTick)+" Score :"+str(sso.gameScore))
            observation, reward, done, info = env.step(Agent.act(sso, elapsed_timer))
            #reward is -1.0 when player loses
            #observation is visual screen
            sso.gameScore += reward
            sso.observation = observation
            
            xs.append(sso.gameTick)
            ys.append(sso.gameScore)

            elapsed_timer = time.time() - training_start_time

            if j == 999:
                print("Training : level "+str(next_level)+" Score : "+str(sso.gameScore))
                next_level = Agent.result(sso, elapsed_timer)
                elapsed_timer = time.time() - training_start_time
                results_file.write(str(sso.gameScore)+"\n")
                #close render window
                env.close()
            if done:
                if reward != -1:
                    sso.gameWinner = "Player"
                print("Training : level "+str(next_level)+" Score : "+str(sso.gameScore))
                next_level = Agent.result(sso, elapsed_timer)
                elapsed_timer = time.time() - training_start_time
                results_file.write(str(sso.gameScore)+"\n")
                #close render window
                env.close()
                break
            if elapsed_timer > training_time_limit:
                print("Training : level "+str(next_level)+" Score : "+str(sso.gameScore))
                next_level = Agent.result(sso, elapsed_timer)
                results_file.write(str(sso.gameScore)+"\n")
                #close render window
                env.close()
                break

    #run level 3,4 for validation
    evaluvating_time_limit = 2*60
    evaluvating_start_time = time.time()
    elapsed_timer = 0
    for i in range(3,5):
        print("Evaluvation : Playing "+game+" level "+str(i))
        ax.set_title("Evaluation : Game :"+game+" Level :"+str(i))
        results_file.write("Evaluation : level "+str(i)+" Score : ")

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
            #print("Evaluvation : Running "+game+" lvl :"+str(i)+" Tick :"+str(sso.gameTick)+" Score :"+str(sso.gameScore))
            observation, reward, done, info = env.step(Agent.act(sso, elapsed_timer))
            #reward is -1.0 when player loses
            #observation is visual screen
            sso.gameScore += reward
            sso.observation = observation
            
            xs.append(sso.gameTick)
            ys.append(sso.gameScore)

            elapsed_timer = time.time() - evaluvating_start_time

            if j == 999:
                print("Training : level "+str(i)+" Score : "+str(sso.gameScore))
                next_level = Agent.result(sso, elapsed_timer)
                results_file.write(str(sso.gameScore)+"\n")
                #close render window
                env.close()
            if done:
                if reward != -1:
                    sso.gameWinner = "Player"
                print("Training : level "+str(i)+" Score : "+str(sso.gameScore))
                next_level = Agent.result(sso, elapsed_timer)
                results_file.write(str(sso.gameScore)+"\n")
                #close render window
                env.close()
                break
            if elapsed_timer > evaluvating_time_limit:
                print("Training : level "+str(i)+" Score : "+str(sso.gameScore))
                next_level = Agent.result(sso, elapsed_timer)
                results_file.write(str(sso.gameScore)+"\n")
                #close render window
                env.close()
                break

    
results_file.close()