# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def foodGridAsList(foodGrid) :
    foodList = []
    if foodGrid :
        i = 0
        for foodRow in foodGrid:
            j = 0
            for foodBoolean in foodRow :
                if (foodBoolean) :
                    foodList.append((i,j))
                j+=1
            i += 1

    return foodList

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        # return legalMoves[chosenIndex]
        chosenMove = legalMoves[chosenIndex]
        # print ("\n\n=======================================", "chosenMove:", chosenMove, " legalMoves:", legalMoves, " scores:", scores)
        return chosenMove

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPacmanPos = successorGameState.getPacmanPosition()
        newFoodGrid = successorGameState.getFood()
        newGhostStatesList = successorGameState.getGhostStates()
        newScaredTimesList = [ghostState.scaredTimer for ghostState in newGhostStatesList]
        # print ("successorGameState:" , successorGameState, "newPacmanPos:" , newPacmanPos, "newFoodGrid:\n" , newFoodGrid ,"newGhostStatesList:" , newGhostStatesList, "newScaredTimesList:" , newScaredTimesList,)

        "*** YOUR CODE HERE Q1***"
        # Calculations

        #find distance to closest ghost
        if newGhostStatesList :
            minGhostDistance = 99999
            for ghost in newGhostStatesList:
                ghostDistance = manhattanDistance(newPacmanPos, ghost.configuration.pos)
                if (ghostDistance < minGhostDistance) :
                    minGhostDistance = ghostDistance
        else :
            minGhostDistance = 0

        if newFoodGrid :
            foodList = foodGridAsList(newFoodGrid)
            remainingFoodNum = len(foodList)
            # print ("newFoodGrid:", newFoodGrid, "foodList", foodList)
            minFoodDistance = 99999
            for foodPos in foodList :
                foodDistance = manhattanDistance(newPacmanPos, foodPos)
                # print ("----------",newPos, foodPos, foodDistance)
                if (foodDistance < minFoodDistance) :
                    minFoodDistance = foodDistance
        else :
            minFoodDistance = 0
            remainingFoodNum = 0

        ## Establishing rules from raw sensor input

        #Generally, higher distance from ghost is better, especially when ghosts are nearby. 
        #This is unless the capsule has been eaten which means Ghosts can be pursued
        if min(newScaredTimesList) == 0 :
            ghostDistRule = - 4.0 / (minGhostDistance + 1)
        else :
            ghostDistRule = 1.0 / (minGhostDistance + 1)

        #generally, lower distance from food is preferred
        foodDistRule =  1.0 / (minFoodDistance + 1)

        #generally, lower remaining food is better
        remainingFoodRule = -remainingFoodNum

        evaluationResult = ghostDistRule + foodDistRule + remainingFoodRule
        # print ("evaluationResult", evaluationResult, "newPos", newPos, "minGhostDistance", minGhostDistance, "minFoodDistance", minFoodDistance, "remainingFoodNum", remainingFoodNum)
        return evaluationResult

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE Q2***"
        return self.mini_max_search(gameState, depth=0, agent_index=self.index)

    def mini_max_search(self, game_state, depth, agent_index):

        # If all the agents have played their turn, go to next depth and start with pacman again
        if agent_index >= game_state.getNumAgents():
            agent_index = 0
            depth += 1

        # End of game
        if depth >= self.depth or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)

        # Max Player/Pacman
        elif agent_index == self.index:
            return self.max_value(game_state, depth, agent_index)
        # Min Player/Ghosts
        else:
            return self.min_value(game_state, depth, agent_index)

    def max_value(self, game_state, depth, agent_index):
        max_value = float("-Inf")
        legal_actions = game_state.getLegalActions(agent_index)
        best_action = None

        for action in legal_actions:
            successor = game_state.generateSuccessor(agent_index, action)
            value = self.mini_max_search(successor, depth, agent_index + 1)

            if value > max_value:
                best_action = action
                max_value = value
        # Root
        if depth == 0:
            return best_action
        else:
            return max_value

    def min_value(self, game_state, depth, agent_index):
        min_value = []
        legal_actions = game_state.getLegalActions(agent_index)

        for action in legal_actions:
            successor = game_state.generateSuccessor(agent_index, action)
            value = self.mini_max_search(successor, depth, agent_index + 1)
            min_value.append(value)

        return min(min_value)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    # a boolean termination function when the game is win or lose or the depth provided is reached
    def isTerminal(self, gameState, depth):
        return gameState.isLose() or gameState.isWin() or depth == self.depth


    def prune(self, agent, gameState, alpha, beta, depth):
        if self.isTerminal(gameState, depth):  
            return self.evaluationFunction(gameState)
        
        #if it is agent 0 (pacman) we maximaze, else then they're ghosts and we perform minimizing
        if agent == 0:
            return self.max_value(agent, gameState, alpha, beta, depth)
        return self.min_value(agent, gameState, alpha, beta, depth)


    #the maximization function
    def max_value(self, agent, gameState, alpha, beta, depth):

        v = float("-inf")
        legal_actions = gameState.getLegalActions(agent)
        for state in legal_actions:
            successor = gameState.generateSuccessor(agent, state)
            v = max(v, self.prune(1, successor, alpha, beta, depth))

            if v > beta:
                return v
            alpha = max(alpha, v)
           
        return v


    #the minimization function
    def min_value(self, agent, gameState, alpha, beta, depth):

        v = float("inf")

        #go to the next agent, and if it is the last aget, reset to pacman (agent 0) and increase the depth
        next_agent = agent + 1 
        if gameState.getNumAgents() == next_agent:
            next_agent = 0
            depth += 1

        legal_actions = gameState.getLegalActions(agent)
        for state in legal_actions:
            successor = gameState.generateSuccessor(agent, state)
            v = min(v, self.prune(next_agent, successor, alpha, beta, depth))

            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        utility = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        action = Directions.WEST

        #perform max_val (maximazing) to the root 
        first_legal_actions = gameState.getLegalActions(0)
        for state in first_legal_actions:
            successor = gameState.generateSuccessor(0, state)
            val = self.prune(1, successor, alpha, beta, 0)
        
            if val > utility:
                utility = val
                action = state

            if utility > beta:
                return utility

            alpha = max(alpha, utility)
       
        return action
        util.raiseNotDefined()



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, agent, depth, gameState):

           
            # END CONDITION
            if gameState.isLose() or gameState.isWin() or depth >= self.depth:  
                return self.evaluationFunction(gameState)

            arr1 = []
            #maximazion for pacman (agent 0)
            if agent == 0:
                legal_actions = gameState.getLegalActions(agent)
                for next_state in legal_actions:
                    successor = gameState.generateSuccessor(agent, next_state)
                    pacman_expect = self.expectimax(1, depth, successor)
                    arr1.append(pacman_expect)
    
                return max(arr1)


            #else increment and do expectimax for other agents (ghosts >= 0)
            next_agent = agent + 1

           #back to pacman, and increment the depth
            agents_count = gameState.getNumAgents()
            if agents_count == next_agent:
                next_agent = 0
                depth += 1
   

            actions_number = 0
            average = 0.0

            # perform the calculations for the average
            for next_state in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, next_state)
                expect = self.expectimax(next_agent, depth, successor)
                agent_actions = gameState.getLegalActions(agent)
                actions_number = float(len(agent_actions))
                average = float(average + (expect / actions_number))

            return average

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        max_val = float("-inf")
        action = Directions.WEST

        # start from root or pacman
        root_actions = gameState.getLegalActions(0)

        for next_state in root_actions:
            successor =  gameState.generateSuccessor(0, next_state)
            util_value = self.expectimax(1, 0, successor)
            
            if util_value > max_val:
                max_val = util_value
                action = next_state

        return action
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE Q5***"
    # Useful information you can extract from a GameState (pacman.py)
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStatesList = currentGameState.getGhostStates()
    scaredTimesList = [ghostState.scaredTimer for ghostState in ghostStatesList]
    gameScore = currentGameState.getScore()
    # print ("currentGameState:" , currentGameState, "pacmanPos:" , pacmanPos, "foodGrid:\n" , foodGrid ,"ghostStatesList:" , ghostStatesList, "scaredTimesList:" , scaredTimesList,)

    # Calculations

    #find distance to closest ghost
    if ghostStatesList :
        minGhostDistance = 99999
        for ghostState in ghostStatesList:
            ghostDistance = manhattanDistance(pacmanPos, ghostState.configuration.pos)
            if (ghostDistance < minGhostDistance) :
                minGhostDistance = ghostDistance
    else :
        minGhostDistance = 0

    if foodGrid :
        foodList = foodGridAsList(foodGrid)
        remainingFoodNum = len(foodList)
        # print ("foodGrid:", foodGrid, "foodList", foodList)
        minFoodDistance = 99999
        for foodPos in foodList :
            foodDistance = manhattanDistance(pacmanPos, foodPos)
            # print ("----------",pacmanPos, foodPos, foodDistance)
            if (foodDistance < minFoodDistance) :
                minFoodDistance = foodDistance
    else :
        minFoodDistance = 0
        remainingFoodNum = 0

    ## Establishing rules from raw sensor input

    #Generally, higher distance from ghost is better, especially when ghosts are nearby.
    #This is unless the capsule has been eaten which means Ghosts can be pursued
    if min(scaredTimesList) == 0 :
        ghostDistRule = - 4.0 / (minGhostDistance + 1)
    else :
        ghostDistRule = 1.0 / (minGhostDistance + 1)

    #Generally, lower distance from food is preferred
    foodDistRule =  1.0 / (minFoodDistance + 1)

    #Generally, lower remaining food is better
    remainingFoodRule = -remainingFoodNum

    #Generally, higher is better
    gameScoreRule = gameScore

    evaluationResult = ghostDistRule + foodDistRule + remainingFoodRule + gameScoreRule
    # print ("evaluationResult", evaluationResult, "newPos", newPos, "ghostDistRule", ghostDistRule, "foodDistRule", foodDistRule)
    return evaluationResult

# Abbreviation
better = betterEvaluationFunction
