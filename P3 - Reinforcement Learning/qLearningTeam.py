# qLearningTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import sys
from util import nearestPoint
from game import Actions

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DefensiveApproximateQAgent', second='DefensiveApproximateQAgent', **args):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  print ("createTeam()firstIndex, secondIndex, isRed,first,second ",firstIndex, secondIndex, isRed,first,second)
  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex, **args), eval(second)(secondIndex, **args)]


# Global counters
# Flag for reporting debug information during testing
debugReportFlag = False
# This is used to save the values for preCalculated maze distances to avoid recalculation
globalDistanceCache = {}

class ValueEstimationAgent(CaptureAgent):
    """
      Abstract agent which assigns values to (state,action)
      Q-Values for an environment. As well as a value to a
      state and a policy given respectively by,

      V(s) = max_{a in actions} Q(s,a)
      policy(s) = arg_max_{a in actions} Q(s,a)

      Both ValueIterationAgent and QLearningAgent inherit
      from this agent. While a ValueIterationAgent has
      a model of the environment via a MarkovDecisionProcess
      (see mdp.py) that is used to estimate Q-Values before
      ever actually acting, the QLearningAgent estimates
      Q-Values while acting in the environment.
    """
    def distance_to_ghost(self, gameState):
        """
        return a list list[0] is distance to nearest ghost list[1] is the state of ghost
        """
        global ghost_state
        my_position = gameState.getAgentState(self.index).getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

        if len(ghosts) > 0:
            distance = 999
            for ghosts in ghosts:
                temp = self.getMazeDistanceFromCache(my_position, ghosts.getPosition())
                if temp < distance:
                    distance = temp
                    ghost_state = ghosts
            return [distance, ghost_state]
        else:
            return None

    def findClosestCapsuleDistance(self, position, gameState):
      ''''
      calculate the distance to the closest capsule
      '''
      capsules = self.getCapsules(gameState)
      if capsules and len(capsules) > 0 :
        minDist = 99999
        for capsule in self.getCapsules(gameState):
          distance = self.getMazeDistanceFromCache(position, capsule)
          if distance < minDist:
            minDist = distance
      else :
        minDist = 0

      return minDist
    def findClosestDangerDistance(self, position):
      ''''
      Locate the ghost that's closest to the agent and can pose a threat. If no ghosts are found/known, then return None
      '''
      # check if there are any ghosts
      ghosts = [oponentAgent for oponentAgent in self.oponentAgentStates
      if not oponentAgent.isPacman and oponentAgent.getPosition() != None and oponentAgent.scaredTimer < 3]

      if len(ghosts) > 0 :
        ghostDists = [self.getMazeDistanceFromCache(position, ghost.getPosition()) for ghost in ghosts]
        return min(ghostDists)
      else:
        return None  # no ghosts found

    def findClosestFoodDistance(self, position, gameState):
      # closest food distance
      foodGridList = self.getFood(gameState).asList()
      if foodGridList and len(foodGridList) > 0:
        closesetDistance = 999999
        i = 0
        for foodPosition in foodGridList :
            distance = self.getMazeDistanceFromCache(position, foodPosition)
            if (distance < closesetDistance):
                closesetDistance = distance
      else:
        closesetDistance = 0

      return closesetDistance

    def populateDistanceCache(self, gameState):
      '''
      Distance cache is used to calculate the maze distance of all the legitimate cells (not walls) and
      store them once in the beginning in of the game to speed up things. The storage requirement is N^2 so
      not suitable for a very very large layout but the normal layouts are nowhere near problematic.
      (1000*1000) grid or bigger can be a concern
      '''
      global globalDistanceCache

      if (globalDistanceCache): # making sure this is done only once
        return

      # print ("---------------------- ******************** ------------------------")
      # print ("---------------------- populateDistanceCache ------------------------")
      # print ("---------------------- ******************** ------------------------")
      counter = 0
      for x1 in range (self.width):
        for y1 in range (self.height):
          for x2 in range (x1+1, self.width):
            for y2 in range (y1+1, self.height):
              counter = counter + 1
              position1 = (x1,y1)
              position2 = (x2,y2)

              #TODO can reduce the cache size to half to save memory

              # distance of a cell to itself is 0
              if x1 == x2 and y1 == y2:
                globalDistanceCache[(position1, position2)] = 0

              # no need to account for walls s they're unreachable
              elif not gameState.hasWall(x1, y1) and not gameState.hasWall(x2, y2):
                distance = self.getMazeDistance(position1, position2)
                globalDistanceCache[(position1, position2)] = distance

      if debugReportFlag: print ("globalDistanceCache len:", len(globalDistanceCache), "width:", self.width, "height:",self.height,
      "totalDistCalculated:", counter, "TotalDist:", pow(self.width * self.height, 2))

    def __init__(self, index, timeForComputing, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 0):
        """
        Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        CaptureAgent.__init__(self, index, timeForComputing)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)

    def calculateAgentPosition(self, gameState):
      '''helper method to calculate agent position'''
      agentPosition = gameState.getAgentState(self.index).getPosition()
      return agentPosition

    def LastConsumedFoodPosition(self, gameState):
        """
        return the location of the last eaten food
        """
        if len(self.observationHistory) > 1:
            prev_state = self.getPreviousObservation()
            prev_food_list = self.getFoodYouAreDefending(prev_state).asList()
            current_food_list = self.getFoodYouAreDefending(gameState).asList()
            if len(prev_food_list) != len(current_food_list):
                for food in prev_food_list:
                    if food not in current_food_list:
                        self.lastEatenFoodPosition = food

    def findClosestTurfBoundary(self):
      ''''
      calculate the distance to the closest boundary location
      '''
      position = self.agentPosition
      minDist = 99999
      for cell in self.homeBoundary:
        distance = self.getMazeDistanceFromCache(position, cell)
        if distance < minDist:
          minDist = distance

      return minDist

    def calculateAgentHomeBoundaries(self,gameState):
      ''''
      create a list of boundaries and returns it for our agent
      '''
      if self.red:
        boundaryX = int(self.midWidth - 1)
      else:
        boundaryX = int(self.midWidth + 1)
      boudaries = [(boundaryX, boundaryY) for boundaryY in  range(self.height)]
      validBoundaries = [] #boundaries that are not wall
      for cellXY in boudaries:
        if not gameState.hasWall(cellXY[0], cellXY[1]):
          validBoundaries.append(cellXY)
      return validBoundaries

    def refreshDynamicValues(self, gameState):
      '''
      A method to keep dynamic values which are dependent on the gameState updated and used when required.
      This is clean up code and to avoid frequently recalculating these values
      '''
      self.oponentAgentStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      self.oponentIndexes = self.getOpponents(gameState)
      self.agentState = gameState.getAgentState(self.index)
      self.agentPosition = self.calculateAgentPosition(gameState)
      self.LastConsumedFoodPosition(gameState)

    def getMazeDistanceFromCache(self, position1, position2):
      '''simply lookup distance of two positions from the cache or calculates in case it's not in the cache'''
      key = (position1, position2)
      keyReverse = (position2, position1)

      if position1 == position2:
        return 0
      elif key in globalDistanceCache:
        return globalDistanceCache[key]
      elif keyReverse in globalDistanceCache:
        return globalDistanceCache[keyReverse]
      else:
        if debugReportFlag: print (" Uncached values, position1, position2", position1, position2)
        globalDistanceCache[key] = self.getMazeDistance(position1, position2)
        return globalDistanceCache[key]

    def registerInitialState(self, state):
      CaptureAgent.registerInitialState(self, state)

      self.start = state.getAgentPosition(self.index)

      self.height = state.data.layout.height
      self.width = state.data.layout.width

      self.midWidth = state.data.layout.width / 2
      self.midHeight = state.data.layout.height / 2

      self.foodEaten = 0

      self.initialnumberOfFood = len(self.getFood(state).asList())
      self.initialnumberOfCapsule = len(self.getCapsules(state))

      self.lastEatenFoodPosition = None
      self.blueRebornHeight = self.height - 1
      self.blueRebornWidth = self.width - 1

      self.refreshDynamicValues(state)
      self.homeBoundary = self.calculateAgentHomeBoundaries(state)
      self.startEpisode()
      self.populateDistanceCache(state)

    ####################################
    #    Override These Functions      #
    ####################################
    def getQValue(self, state, action):
        """
        Should return Q(state,action)
        """
        util.raiseNotDefined()

    def getValue(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s,a)
        """
        util.raiseNotDefined()

    def getPolicy(self, state):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with getAction
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        util.raiseNotDefined()

    def getAction(self, state):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        util.raiseNotDefined()


class ReinforcementAgent(ValueEstimationAgent):
    """
      Abstract Reinforcemnt Agent: A ValueEstimationAgent
            which estimates Q-Values (as well as policies) from experience
            rather than a model

        What you need to know:
                    - The environment will call
                      observeTransition(state,action,nextState,deltaReward),
                      which will call update(state, action, nextState, deltaReward)
                      which you should override.
        - Use self.getLegalActions(state) to know which actions
                      are available in a state
    """
    ####################################
    #    Override These Functions      #
    ####################################

    def update(self, state, action, nextState, reward):
        """
                This class will call this function, which you write, after
                observing a transition and reward
        """
        util.raiseNotDefined()

    ####################################
    #    Read These Functions          #
    ####################################

    def getLegalActions(self, state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return state.getLegalActions(self.index)

    def observeTransition(self, state, action, nextState, deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward)

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def __init__(self, index, timeForComputing, **args):
        """
        actionFn: Function which takes a state and returns the list of legal actions

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        ValueEstimationAgent.__init__(self, index, timeForComputing, **args)
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0


    ################################
    # Controls needed for Crawler  #
    ################################
    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setLearningRate(self, alpha):
        self.alpha = alpha

    def setDiscount(self, discount):
        self.discount = discount

    def doAction(self,state,action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action

    ###################
    # Pacman Specific #
    ###################
    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

    def registerInitialState(self, state):
        ValueEstimationAgent.registerInitialState(self, state)
        self.startEpisode()
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining), "for:", self.index)

    def final(self, state):
        """
          Called by Pacman game at the terminal state
        """
        if debugReportFlag: print ("final() index", self.index)
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print('\tCompleted %d out of %d training episodes' % (
                       self.episodesSoFar,self.numTraining))
                print('\tAverage Rewards over all training: %.2f' % (
                        trainAvg))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
                print('\tAverage Rewards over testing: %.2f' % testAvg)
            print('\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE,windowAvg))
            print('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))


class DefensiveApproximateQAgent(ReinforcementAgent):
    """
    QLearningAgent:
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
# class DefensiveApproximateQAgent(QLearningAgent):
    "PacmanQAgent: Exactly the same as QLearningAgent, but with different default parameters"
    """
    PacmanQAgent class
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, index, timeForComputing = .1, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self,index, timeForComputing,  **args)

        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()

# def __init__(self, index, timeForComputing = .1, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """

        self.weights = {
            'distToFoodFeature': -10.0, #not encouraging the defensive agent to search food
            'distToCapsuleFeature': -50.0, #not encouraging the defensive agent to get capsule
            'distToLastEatenFoodFeature': 20.0, #encourages going closer to where last food lost
            'stopModeFeature': -50.0,  # not encouraging stopping
            'defensiveModeFeature': 50.0,  #encourages defensive agent to be in defensive mode
            'reverseActionFeature': -5.0,  #reversing is not encouraged as showes poor decision making
            'InvadersCountFeature': -1000.0, # eliminating invaders yeild lots of points
            'closenessToInvaderFeature': 400.0, #trying to get closer to invaders
            'distToBoundaryFeature': 1.0 #encourages being closer to the boundary area
            }

    def registerInitialState(self, gameState):
      ReinforcementAgent.registerInitialState(self, gameState)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state, action)]
        # util.raiseNotDefined()

        # """
        #   from super class:
        #   Returns Q(state,action)
        #   Should return 0.0 if we have never seen a state
        #   or the Q node value otherwise
        # """
        # "*** YOUR CODE HERE ***"

    def getSuccessor(self, gameState,  action):
      """Finds the next successor which is a grid position (location tuple)."""
      successor = gameState.generateSuccessor(self.index, action)
      return successor


    def calculateFeatures(self, gameState, action):
      features = util.Counter()
      successor = self.getSuccessor(gameState, action)

      myState = successor.getAgentState(self.index)
      agentPosition = gameState.getAgentState(self.index).getPosition()

      distToFood = self.findClosestFoodDistance(agentPosition, gameState)
      features['distToFoodFeature'] = distToFood if (distToFood) else 0.0

      distToCapsule = self.findClosestCapsuleDistance(agentPosition, gameState)
      features['distToCapsuleFeature'] = distToCapsule if (distToCapsule) else 0.0


      if (self.lastEatenFoodPosition):
        features['distToLastEatenFoodFeature'] = self.getMazeDistanceFromCache(agentPosition, self.lastEatenFoodPosition)
      else:
        features['distToLastEatenFoodFeature'] = 0.0

      features['stopModeFeature'] = 1.0 if action == Directions.STOP else 0.0

      features['defensiveModeFeature'] = 0.0 if myState.isPacman else 1.0

      rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      features['reverseActionFeature'] = 1.0 if action == rev else 0.0

      # Computes distance to invaders we can see
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      nearbyPacmen = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]
      features['InvadersCountFeature'] = float(len(nearbyPacmen))

      features['closenessToInvaderFeature'] = 0.0
      if len(nearbyPacmen) > 0:
        dists = [self.getMazeDistanceFromCache(self.agentPosition, invader.getPosition()) for invader in nearbyPacmen]
        features['closenessToInvaderFeature'] = float(1/min(dists))
        if gameState.getAgentState(self.index).scaredTimer > 2:
          features['closenessToInvaderFeature'] = float(-1/min(dists))

      features['distToBoundaryFeature'] = 0.0 if self.lastEatenFoodPosition else float(- self.findClosestTurfBoundary())

      return features


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)

        # terminal state has value of 0.0
        if len(actions) != 0:
          # return the max Value of available values
          return max([self.getQValue(state, action) for action in actions])
        else:
          return 0.0

        # util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if len(actions) == 0:
          return None
        else:
          stateMaxQValue = self.computeValueFromQValues(state)
          topActions = [action for action in actions if self.getQValue(state, action) == stateMaxQValue]
          if (not topActions):
            return Directions.STOP
          else:
            chosenPolicy = random.choice(topActions)
            return chosenPolicy
        # util.raiseNotDefined()

    def getAction(self, state):
      """
      Simply calls the getAction method of QLearningAgent and then
      informs parent of action for Pacman.  Do not change or remove this
      method.
      """

      """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
      """
      # Pick Action
      # state.getLegalActions()
      legalActions = self.getLegalActions(state)
      if debugReportFlag: print ("  ----- getAction() self.index, legalActions", self.index, legalActions)
      action = None
      "*** YOUR CODE HERE ***"
      probability = self.epsilon
      if util.flipCoin(probability) == True:
        action = random.choice(legalActions)
      else:
        action = self.computeActionFromQValues(state)

      self.doAction(state, action)
      return action

    ####################################
    #    Override These Functions      #
    ####################################

    def getQValue(self, state, action):
      """ the sum of all the features by their weith will be the qValue"""
      qValue = 0
      featureDict = self.calculateFeatures(state, action)
      weightDict = self.getWeights()

      for key in featureDict.keys():
        qValue = qValue + (weightDict[key] * featureDict[key])

      return qValue

    def calcNewReward(self, state, nextState, reward):
      # TODO correct later, temp solution
      newReward = reward + 1
      return newReward

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        """
        super class:
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        """
          super super class:
                This class will call this function, which you write, after
                observing a transition and reward
        """

        "*** YOUR CODE HERE ***"
        featureDict = self.calculateFeatures(state, action)
        qValueCurrentState = self.getQValue(state,action)
        if debugReportFlag: print(" **** new features:", featureDict)

        qValueNextState = self.computeValueFromQValues(nextState)
        difference = (reward + self.discount*(qValueNextState))- qValueCurrentState
        for key in featureDict.keys():
          self.weights[key] =  self.weights[key] + (featureDict[key] * self.alpha * difference)
        if debugReportFlag: print(" **** new weights:", self.weights)




    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        ReinforcementAgent.final(self, state)

        # did we finish training?
        if debugReportFlag: print (" *** final() index, self.episodesSoFar, self.numTraining",self.index, self.episodesSoFar , self.numTraining)
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(" *****        Finalised Weights:")
            print(self.getWeights())
            pass
