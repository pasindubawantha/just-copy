class YourAgent:
    """
    Method to be called at the start of an evaluation process (training & testing) for a particular game.
    """
    def __init__(self):
        self.yourvar = 0
    
    
    """
    * Public method to be called at the start of every level of a game.
    * Perform any level-entry initialization here.
    * @param sso Phase Observation of the current game.
    * @param elapsedTimer Timer (1s)
    """
    def init(self, sso, elapsed_timer):
        print("init ran")


    """
    Method used to determine the next move to be performed by the agent.

    @param sso: observation of the current state of the game
    @param elapsed_timer: the timer
    @return index of the action to be taken
    """
    def act(self, sso, elapsed_timer):
        print("act ran")

        return sso.availableActions[1]

    """
    * Method used to perform actions in case of a game end.
    * This is the last thing called when a level is played (the game is already in a terminal state).
    * Use this for actions such as teardown or process data.
    *
    * @param sso The current state observation of the game.
    * @param elapsedTimer Timer (up to CompetitionParameters.TOTAL_LEARNING_TIME
    * or CompetitionParameters.EXTRA_LEARNING_TIME if current global time is beyond TOTAL_LEARNING_TIME)
    * @return The next level of the current game to be played.
    * The level is bound in the range of [0,2]. If the input is any different, then the level
    * chosen will be ignored, and the game will play a random one instead.
    """
    def result(self, sso, elapsed_timer):
        print("result ran")
        return 0