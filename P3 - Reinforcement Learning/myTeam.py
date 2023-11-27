# myTeam.py
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
from util import nearestPoint
from game import Actions

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent', **args):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex, **args)]

# Global counters
offence_action_counter = 0
defence_action_counter = 0
generic_action_counter = 0
ghost_state = 0
# Strings keeping debug information between agent calls
current_offensive_decision = ""
previous_offensive_decision = ""
# Flag for reporting debug information during testing
debugReportFlag = True

# Flag for clearing previous debugDraw
debugDrawClearFlag = True
# This is used to save the values for preCalculated maze distances to avoid recalculation
globalDistanceCache = {}

agentIntentionDraw = False

##########
# Agents #
##########

class BaseAgent(CaptureAgent):
  """
  A base class for the Defensive and Offensive agents to inherit with common functionality
  """
  def refreshDynamicValues(self, gameState):
    '''
    A method to keep dynamic values which are dependent on the gameState updated and used when required.
    This is clean up code and to avoid frequently recalculating these values
    '''
    self.oponentAgentStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    self.oponentIndexes = self.getOpponents(gameState)
    self.agentState = gameState.getAgentState(self.index)
    self.agentPosition = self.calculateAgentPosition(gameState)

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


  def registerInitialState(self, gameState):
    '''
    Your initialization code goes here, if you need any.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    self.start = gameState.getAgentPosition(self.index)

    self.height = gameState.data.layout.height
    self.width = gameState.data.layout.width

    self.midWidth = gameState.data.layout.width / 2
    self.midHeight = gameState.data.layout.height / 2

    self.foodEaten = 0
    self.lastEatenFoodPosition = None

    self.initialnumberOfFood = len(self.getFood(gameState).asList())
    self.initialnumberOfCapsule = len(self.getCapsules(gameState))

    self.blueRebornHeight = self.height - 1
    self.blueRebornWidth = self.width - 1

    self.refreshDynamicValues(gameState)
    self.homeBoundary = self.calculateAgentHomeBoundaries(gameState)
    self.populateDistanceCache(gameState)

  def debugReport(self, offensiveAgentDecisionStr, offensiveAgentActions):
    """
    report used for debug to track decision making of the Agents
    """
    global debugReportFlag

    if (not debugReportFlag):
      return

    print ("OfficeAgent", offence_action_counter, " ** ",offensiveAgentDecisionStr, " ** " , offensiveAgentActions)

    position = self.agentState.getPosition()
    intendedPositions = []
    if offensiveAgentActions and len(offensiveAgentActions) > 0 :
      for action in offensiveAgentActions:
        if (action == Directions.NORTH):
          position = (position[0], position[1] + 1)
        elif (action == Directions.SOUTH):
          position = (position[0], position[1] - 1)
        elif (action == Directions.EAST):
          position = (position[0] + 1, position[1])
        elif (action == Directions.WEST):
          position = (position[0] - 1, position[1])

        intendedPositions.append(position)

    if (not agentIntentionDraw):
      return

    if (debugDrawClearFlag) : self.debugClear()

    self.debugDraw(intendedPositions, [random.uniform(.5,1.0),random.uniform(0,0.2),random.uniform(0,0.2)])

  def isInHomeTurf(self):
    '''
    Simply check if agent is in home turf or the opponent turf
    '''
    positionX, positionY = self.agentState.getPosition()
    if self.red :
      return positionX < self.midWidth
    else :
      return positionX > self.midWidth

  def isPositionInHomeTurf(self, position):
    '''
    Simply check if agent is in home turf or the opponent turf
    '''
    positionX, positionY = position
    if self.red :
      return positionX < self.midWidth
    else :
      return positionX > self.midWidth

  def isInHomeTurfCautious(self):
    '''
    Makes sure at least one deap in home turf to avoid border issues
    '''
    positionX, positionY = self.agentState.getPosition()
    if self.red :
      return positionX < self.midWidth -1
    else :
      return positionX > self.midWidth + 1

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

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    global generic_action_counter
    generic_action_counter = generic_action_counter + 1
    if debugReportFlag: print ("BaseAgent.chooseAction", generic_action_counter)

    self.refreshDynamicValues(gameState)

    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistanceFromCache(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    # print ("BaseAgent.getSuccessor")
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
        # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    # print ("BaseAgent.evaluate")

    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    if debugReportFlag: print ("*** evaluate() evaluate:", features * weights)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    # print ("BaseAgent.getFeatures")

    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gameState. They can be either
    a counter or a dictionary.
    """
    # print ("BaseAgent.getWeights")

    return {'successorScore': 1.0}

  def getAgentPosition(self):
    '''helper method to return agent position'''
    return self.agentPosition

  def calculateAgentPosition(self, gameState):
    '''helper method to calculate agent position'''
    agentPosition = gameState.getAgentState(self.index).getPosition()
    return agentPosition

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
      globalDistanceCache[key] = self.getMazeDistance(position1, position2)
      return globalDistanceCache[key]

  def getMyMazeDistanceFromCache(self, position):
    '''
    convenience function to calculate the distance from the agent to another position
    '''
    return self.getMazeDistanceFromCache(self.agentState.getPosition(), position)

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

  def findClosestDangerDistanceWithState(self, position, gameState):
    ''''
    Locate the ghost that's closest to the agent and can pose a threat. If no ghosts are found/known, then return None
    '''
    # check if there are any ghosts
    oponentAgentStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

    ghosts = [oponentAgent for oponentAgent in oponentAgentStates
    if not oponentAgent.isPacman and oponentAgent.getPosition() != None and oponentAgent.scaredTimer < 3]

    if len(ghosts) > 0 :
      ghostDists = [self.getMazeDistanceFromCache(position, ghost.getPosition()) for ghost in ghosts]
      return min(ghostDists)
    else:
      return None  # no ghosts found

  def findClosestOponentPacman(self, gameState):
    ''''
    Locate the pacman with capsule that's closest to the agent and can pose a threat. If none is found, then return None
    '''
    agentPosition = self.calculateAgentPosition(gameState)
    oponentAgentStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

    OponentPacmans = [oponentAgent for oponentAgent in oponentAgentStates
    if oponentAgent.isPacman and oponentAgent.getPosition() != None and oponentAgent.scaredTimer < 3]

    if len(OponentPacmans) > 0:
      dists = [self.getMazeDistanceFromCache(agentPosition, OponentPacman.getPosition()) for OponentPacman in OponentPacmans]
      return min(dists)
    else:
      return None

  def pacmanAvoidanceWhenScaredHeuristic(self, position, gameState):
      """
      This heuristic is used for to avoid pacman when they eaten the capsule and the agent is scared,
      positions closer to a angry pacman are given higher heuristic value when the agent is scared
      """
      # print ("pacmanAvoidanceWhenScaredHeuristic", position)

      heuristic = 0
      ghostDist = self.findClosestOponentPacman(gameState)
      # print ("pacmanAvoidanceWhenScaredHeuristic, ghostDist", ghostDist,"type(ghostDist)",type(ghostDist))
      if ghostDist != None and ghostDist < 5:
        heuristic = pow((4 - ghostDist),4)

      heuristic = max(heuristic, 0)  # ensure we don't go lower than 0
      return heuristic

  def aStarHeuristic(self, position, problem):

    """
    So far the heuristic in mind is more like generalized, I don't know how good or how bad it will be
    but we can start something and try it, but the idea of the heuristic is to avoid the ghosts when when offending,
    maybe this way we can also remove the function for capsuledInvader, and use the heuristic to defend

    first we need a check for any near ghosts, then we get the enemies and ghosts, then we get the distance of the ghosts,(manhattan? maze?)
    and then calc an appropriate heuristic.
    """

    heuristic = 0
    enemies = []
    ghosts = []
    all_ghost_pos = []
    distances = []

    # get the position of the agent
    pac_pos = problem.gameState.getAgentState(self.index).getPosition()

    nearest_ghost = self.findClosestGhostDistance(problem.gameState)

    if nearest_ghost != None:
    # get enemies first
      opponents = self.getOpponents(problem.gameState)
      for opp in opponents:
        curr_opp = problem.gameState.getAgentState(opp)
        enemies.append(curr_opp)

      for a in enemies:
        if not a.isPacman and a.getPosition() != None:
          ghosts.append(a)

      if len(ghosts) > 0 and ghosts != None:
        # min_ghost_dist = 0
        for i in ghosts:
          # maze_distance = self.getMazeDistanceFromCache(pac_pos, i.getPosition())
          # print("Maze dist:  ",maze_distance)
          all_ghost_pos.append(i.getPosition())

        for i in all_ghost_pos:
          distances.append(self.getMazeDistanceFromCache(position, i))

        min_ghost_dist = min(distances)
        if min_ghost_dist != None and min_ghost_dist < 3:
          heuristic = pow((4 - min_ghost_dist),4)

        heuristic = max(heuristic, 0)  # ensure we don't go lower than 0
    return heuristic

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

  def foodHuntAndDangerAvoidanceHeuristic(self, position, problem):
    """
    This heuristic is used for to be generic and to look for food while avoiding ghosts,
    positions closer to a ghost are given higher heuristic value
    """

    # print ("ghostAvoidanceHeuristic", position)

    closesetFoodDistance = self.findClosestFoodDistance(position, problem.gameState)

    # closest capsule
    capsuleDistance = self.findClosestCapsuleDistance(position, problem.gameState)

    ghostDist = self.findClosestDangerDistance(position)
    # print ("ghostAvoidanceHeuristic, position:",position, "ghostDist:", ghostDist,"type(ghostDist)",type(ghostDist))
    if ghostDist != None and ghostDist < 5:
      ghostHeuristicValue = pow((5 - ghostDist), 5)
    else:
      ghostHeuristicValue = 0

    # add 1 to heuristic if the position is in oponent turf due to additional risk
    oponentTurfRisk =  0 if self.isPositionInHomeTurf(position) else 1

    heuristic = ghostHeuristicValue * 5 + closesetFoodDistance * 2 + capsuleDistance * 3 + oponentTurfRisk

    heuristic = max(heuristic, 0)  # ensure we don't go lower than 0

    return heuristic

  def ghostAvoidanceHeuristic(self, position, problem):
    """
    This heuristic is used for to avoid ghosts,
    positions closer to a ghost are given higher heuristic value
    """

    # print ("ghostAvoidanceHeuristic", position)

    heuristic = 0
    ghostDist = self.findClosestDangerDistance(position)
    # print ("ghostAvoidanceHeuristic, position:",position, "ghostDist:", ghostDist,"type(ghostDist)",type(ghostDist))
    if ghostDist != None and ghostDist < 5:
      heuristic = pow((5 - ghostDist),5)

    heuristic = max(heuristic, 0)  # ensure we don't go lower than 0
    return heuristic

  def finaliseActionSelection(self, titleStr, actions):
      self.debugReport(titleStr, actions)
      if actions and len(actions) != 0:
        return actions[0]
      else :
        return Directions.STOP


class DefensiveAgent(BaseAgent):

    def chooseAction(self, gameState):
        global agentIntentionDraw
        agentIntentionDraw = False

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()


        self.LastConsumedFoodPosition(gameState)

        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]

        # get important data from opponent, i.e. enemies invaders etc.
        enemies = []
        for i in self.getOpponents(gameState):
            enemies.append(gameState.getAgentState(i))

        # list of invaders
        invaders = []
        for a in enemies:
            if a.isPacman:
                invaders.append(a)

        # list of invaders within the range
        known_invaders = []
        for a in enemies:
            if a.getPosition() != None and a.isPacman:
                known_invaders.append(a)

        global defence_action_counter
        defence_action_counter = defence_action_counter + 1

        # if there aren't ane invaders (attackers) then our defender can attack and get some points as well.
        if len(invaders) < 1:
            # eat maximum 3 food
            if gameState.getAgentState(self.index).numCarrying < 3 and len(
                    self.getFood(gameState).asList()) != 0 and not (
                    self.distance_to_ghost(gameState) is not None and self.distance_to_ghost(gameState)[0] < 4 and
                    self.distance_to_ghost(gameState)[1].scaredTimer < 2):
                # Look for all the food
                problem = FoodSearchProblem(gameState, self)
                actions = aStarSearch(problem, self.foodHuntAndDangerAvoidanceHeuristic)
                self.debugReport("FoodSearchProblem", actions)

                if actions and len(actions) != 0:
                    return actions[0]
            else:
                # if any of the conditions above breaks, our defender (who'd now attacking), should retreat
                problem = RetreatToTurfSearchProblem(gameState, self)
                actions = aStarSearch(problem, self.ghostAvoidanceHeuristic)
                if actions and len(actions) != 0:
                    return actions[0]
                else:
                  bestAction = super().chooseAction(gameState)
                  self.debugReport("ReflextAgent", [bestAction])
                  return bestAction

        # but if there are invaders, the defender will go and chase the last consumed food on our turf
        # this way, when the defender gets to the position of that last eaten food, he should be able to know -
        # that the attacker is around him if the attacker was within the range of 5 squares and thus chase him. we cannot directly chase
        # an attacker wherever he is, since we're limited to the fact that attackers are observable within 5 squares.
        if len(invaders) > 0:
            if len(known_invaders) == 0 and self.lastEatenFoodPosition is not None \
                    and gameState.getAgentState(self.index).scaredTimer == 0:

                problem = ChaseLastDotConsumed(gameState, self)
                actions = aStarSearch(problem, self.foodHuntAndDangerAvoidanceHeuristic)
                return self.finaliseActionSelection("RetreatToTurfSearchProblem (final Run)", actions)

            # otherwise, defender will take some random actions.
            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]
            return random.choice(best_actions)

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
            for a in ghosts:
                temp = self.getMazeDistanceFromCache(my_position, a.getPosition())
                if temp < distance:
                    distance = temp
                    ghost_state = a
            return [distance, ghost_state]
        else:
            return None

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

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


class OffensiveAgent(BaseAgent):

  def chooseAction(self, gameState):
    """
    the key function of the agent, making decision about the next action to be carried out depending on a
    number of observatory details
    """
    global agentIntentionDraw
    agentIntentionDraw = True

    global offence_action_counter
    offence_action_counter = offence_action_counter + 1
    if debugReportFlag: print ("OffensiveAgent.chooseAction", offence_action_counter)

    self.refreshDynamicValues(gameState)

    # depending on the status, a specific problem will be searched for
    closestDangerDistance = self.findClosestDangerDistance(self.agentState.getPosition())
    # print ("OffensiveAgent.chooseAction closestGhostDistance:", closestGhostDistance)

    # if no more food needed to collect and carrying food, then go back to turf
    if not self.isInHomeTurf() and (len(self.getFood(gameState).asList()) <= 2 and self.agentState.numCarrying > 0):
      problem = RetreatToTurfSearchProblem(gameState, self)
      actions = aStarSearch(problem, self.ghostAvoidanceHeuristic)
      return self.finaliseActionSelection("RetreatToTurfSearchProblem (final Run)", actions)

    # if carrying food and reaching time limit at the end of the game, then go back to turf
    if self.agentState.numCarrying > 0 and gameState.data.timeleft < self.findClosestTurfBoundary() + 10 :
      problem = RetreatToTurfSearchProblem(gameState, self)
      actions = aStarSearch(problem, self.ghostAvoidanceHeuristic)
      return self.finaliseActionSelection("RetreatToTurfSearchProblem (final Run)", actions)


    # if self.agentState.numCarrying > 2 and gameState.data.timeleft < self.findClosestTurfBoundary() + 10 and self.getScore() <= 3:
    #   problem = RetreatToTurfSearchProblem(gameState, self)
    #   actions = aStarSearch(problem, self.ghostAvoidanceHeuristic)
    #   return self.finaliseActionSelection("RetreatToTurfSearchProblem (final Run)", actions)

    # the capsule problem, the attacker aims for the capsule when he is in danger, and if the capsule is close,
    # rather than run home or any other action.
    if not self.isInHomeTurf() and closestDangerDistance != None and closestDangerDistance < 4 and \
      self.findClosestCapsuleDistance(self.agentPosition ,gameState) <=4 and len(self.getCapsules(gameState)) != 0:

      problem = CapsuleSearchProblem(gameState, self)
      actions = aStarSearch(problem, self.ghostAvoidanceHeuristic)
      return self.finaliseActionSelection("CapsuleSearchProblem (final Run)", actions)


    # if no more food needed to collect and not carrying food, go hunt oponents
    # TODO uncomment and correct below
    # if not self.isInHomeTurf() and (len(self.getFood(gameState).asList()) <= 2 and self.agentState.numCarrying == 0):
      # problem = SearchHuntInvaders(gameState, self)
      # actions = aStarSearch(problem, self.foodHuntAndDangerAvoidanceHeuristic)  #TODO this heuristic needs to be updated
      # return self.finaliseActionSelection("SearchHuntInvaders", actions)

    # TODO add a condition when the oponents are pacman (eaten capsule)
    if not self.isInHomeTurf() and closestDangerDistance != None and closestDangerDistance < 4:
      problem = AvoidOponentSearchProblem(gameState, self)
      actions = aStarSearch(problem, self.ghostAvoidanceHeuristic)
      return self.finaliseActionSelection("AvoidOponentSearchProblem", actions)

    # a case when for our attacker to know when to retreat back to the turf or home,
    if not self.isInHomeTurfCautious() and (len(self.getFood(gameState).asList()) <= 2 or self.agentState.numCarrying > 7):
      problem = RetreatToTurfSearchProblem(gameState, self)
      actions = aStarSearch(problem, self.foodHuntAndDangerAvoidanceHeuristic)
      return self.finaliseActionSelection("RetreatToTurfSearchProblem", actions)

    # a case for when the agent is carrying a specific number of food, and the next food is quite far, this case ensures
    # that the agent goes back home and drops the food to secure it rather than risk it all for an extra far food.
    # findClosestFoodDistance vs findClosestTurfBoundary

    if not self.isInHomeTurf() and self.agentState.numCarrying > 3 and self.findClosestFoodDistance(self.agentPosition,gameState) > self.findClosestTurfBoundary():
      problem = RetreatToTurfSearchProblem(gameState, self)
      actions = aStarSearch(problem, self.foodHuntAndDangerAvoidanceHeuristic)
      return self.finaliseActionSelection("RetreatToTurfSearchProblem", actions)

    # if self.agentState.numCarrying < 1:

    # if none of the above conditions are satisfied, then the general way
    # of the attacker's actions is to search for food without conditions as follows
    problem = FoodSearchProblem(gameState, self)
    actions = aStarSearch (problem, self.foodHuntAndDangerAvoidanceHeuristic)
    self.debugReport("FoodSearchProblem", actions)
    if actions and len(actions) != 0:
      return actions[0]

    # fall back to default reflex agent if no actions were selected this far
    bestAction = super().chooseAction(gameState)
    self.debugReport("ReflextAgent", [bestAction])
    return bestAction


  def getFeatures(self, gameState, action):
    # self.refreshDynamicValues(gameState)

    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    position = self.calculateAgentPosition(successor)
    if debugReportFlag: print ("*** offensiveAgent.getFeature() action, position, origPosition", action, position, self.agentPosition)
    foodList = self.getFood(successor).asList()
    numberOfFood = len(foodList)
    if (numberOfFood > 2):
      features['numberOfFood'] = numberOfFood
      features['distToFood'] = self.findClosestFoodDistance(position, successor)
    else :
      features['numberOfFood'] = 0
      features['distToFood'] = 0

    capsules = self.getCapsules(successor)
    features['numberOfCapsules'] = len(capsules)

    features['distToCapsules'] = self.findClosestCapsuleDistance(position ,successor)

    distToDanger = self.findClosestDangerDistanceWithState(position, successor)
    distToPacman = self.findClosestOponentPacman(successor)

    if (not distToDanger):
      distToDanger = 9999

    if (not distToPacman) :
      distToPacman = 9999

    distToDanger = min(distToDanger, distToPacman)
    if (distToDanger > 5): # ignore if far away
      features['distToDanger'] = 0
    elif (distToDanger <= 5 and distToDanger> 3):
      features['distToDanger'] = -100
    else :
      features['distToDanger'] = -1000


    return features

  def getWeights(self, gameState, action):
    return {'numberOfFood': -100, 'distToFood': -1, 'numberOfCapsules': -150, 'distToCapsules': -2, 'distToDanger': 1}


# ------------------------------- from Assignment 1 solution,  A* Search code

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    """ using a similar approach to UCS search but as defined for A*, using the huristic + nodeCost to prioritise
    the node for expansion in the frontier"""

    # using stack as LIFO queue to track frontier
    rootNode = problem.getStartState()
    frontierPQueue = util.PriorityQueue()
    rootActionList = []
    cost = 0

    # tracking the nodes to the goal and the relevant actions leading to the node.
    frontierPQueue.push((rootNode, rootActionList, cost), cost + heuristic(rootNode, problem))

    # tracking the set of all explored nodes to avoid repetition
    exploredSet = set()

    while not frontierPQueue.isEmpty():
        currentNode, currentNodeActions, currentNodeCost = frontierPQueue.pop()

        if currentNode not in exploredSet:

            if problem.isGoalState(currentNode):
                return currentNodeActions

            exploredSet.add(currentNode)

            # add the children/successors to the frontier
            for successor, action, stepCost in problem.getSuccessors(currentNode):
                if successor not in exploredSet :
                    successorAction = currentNodeActions + [action]
                    successorCost = currentNodeCost + stepCost
                    frontierPQueue.push((successor, successorAction, successorCost), successorCost + heuristic(successor, problem))


#######################################################
# Creating Search Problems                            #
#######################################################

class BaseSearchProblem:
  '''
  this is the base class for all problems, taken from assignment 1 search
  '''
  def __init__(self, gameState, agent, costFn = lambda x: 1):
    self.gameState = gameState
    self.walls = gameState.getWalls()
    self.startState = gameState.getAgentState(agent.index).getPosition()
    self.carry = gameState.getAgentState(agent.index).numCarrying
    self.costFn = costFn
    # store a number of relevant details to be used in the future by the problem
    self.food = agent.getFood(gameState)
    self.capsules = agent.getCapsules(gameState)
    self.walls = gameState.getWalls()
    self.agent = agent

    # For display purposes
    self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE


  def getStartState(self):
      return self.startState

  def isGoalState(self, state):
    ''' goals need to be defined by subcalsses so it will be an error if the child doesn't override this function'''
    if (True):
      util.raiseNotDefined()

    isGoal = state == self.goal
    return isGoal

  def getSuccessors(self, state):
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
        x,y = state
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        if not self.walls[nextx][nexty]:
            nextState = (nextx, nexty)
            cost = self.costFn(nextState)
            successors.append( ( nextState, action, cost) )

    # Bookkeeping for display purposes
    self._expanded += 1 # DO NOT CHANGE
    if state not in self._visited:
        self._visited[state] = True
        self._visitedlist.append(state)

    return successors

  def getCostOfActions(self, actions):
    """
    Returns the cost of a particular sequence of actions. If those actions
    include an illegal move, return 999999.
    """
    if actions == None: return 999999
    x,y= self.getStartState()
    cost = 0
    for action in actions:
        # Check figure out the next state and see whether its' legal
        dx, dy = Actions.directionToVector(action)
        x, y = int(x + dx), int(y + dy)
        if self.walls[x][y]: return 999999
        cost += self.costFn((x,y))
    return cost


class FoodSearchProblem(BaseSearchProblem):
  """
   This problem looks for all the food
  """

  def __init__(self, gameState, agent):
    BaseSearchProblem.__init__(self, gameState, agent)
    "Stores information from the gameState.  You don't need to change this."
    self.oponentAgentStates = [gameState.getAgentState(i) for i in self.agent.getOpponents(gameState)]
    self.foodLeft = len(self.food.asList())


  def isGoalState(self, state):
    oponents = [oponentAgent for oponentAgent in self.oponentAgentStates
      if not oponentAgent.isPacman and oponentAgent.scaredTimer < 3 ]

    #only check for capsule if the oponents are not scared otherwise don't waste them
    return state in self.food.asList() or (oponents and len(oponents)> 0 and state in self.capsules)


class AvoidOponentSearchProblem(BaseSearchProblem):
  """
   running away from opponent, either as ghost or angry pacman who's eaten the capsule
  """

  def __init__(self, gameState, agent):
    BaseSearchProblem.__init__(self, gameState, agent)
    "Stores information from the gameState.  You don't need to change this."
    self.foodLeft = len(self.food.asList())
    self.homeBoundary = agent.homeBoundary


  def isGoalState(self, state): #safe when returned to home area or has the capsule
    return state in self.capsules or state in self.homeBoundary

class CapsuleSearchProblem(BaseSearchProblem):
    ''' This problem is used to search for valid capsules'''
    def __init__(self,  gameState, agent):
        BaseSearchProblem.__init__(self, gameState, agent)
        self._expanded = 0 # DO NOT CHANGE
        self.walls = gameState.getWalls()
        self.startState = ()
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        self.gameState = gameState
        self.walls = gameState.getWalls()
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.carry = gameState.getAgentState(agent.index).numCarrying

        self.agent = agent

    def getStartState(self):
      return self.startState

    def isGoalState(self, state):
      return state in self.capsule



class SearchHuntInvaders(BaseSearchProblem):
  """
   Actively hunt invaders where possible
  """

  def __init__(self, gameState, agent):
    BaseSearchProblem.__init__(self, gameState, agent)
    "Stores information from the gameState.  You don't need to change this."
    self.foodLeft = len(self.food.asList())
    self.homeBoundary = agent.homeBoundary
    self._expanded = 0 # DO NOT CHANGE
    self.walls = gameState.getWalls()
    self.startState = ()
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    self.gameState = gameState

  # the goal state for this problem is going to be the position of the invaders (if any).
    self.oponentIndexes = agent.getOpponents(gameState)
    self.invaders = []
    self.enemies = []
    self.ghosts = []

    for opp in self.oponentIndexes:
      curr_opp = gameState.getAgentState(opp)
      self.enemies.append(curr_opp)

    for a in self.enemies:
      if a.isPacman and a.getPosition() != None:
        self.ghosts.append(a)

    # when we have invaders, we get their positions

    if len(self.invaders) > 0:

      self.invaders_pos = []
      for i in self.invaders:
        self.invaders_pos.append(i.getPosition())
    self.invaders_pos = None

  def isGoalState(self, state):
    return state in self.invaders_pos


class ChaseLastDotConsumed(BaseSearchProblem):
  '''
  This problem finds the last location of the eaten food to track the opponent agent
  that consumed it
  '''
  def __init__(self, gameState, agent):
    BaseSearchProblem.__init__(self, gameState, agent)
    self.foodLeft = len(self.food.asList())
    self.homeBoundary = agent.homeBoundary
    self._expanded = 0 # DO NOT CHANGE
    self.walls = gameState.getWalls()
    self.startState = ()
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    self.gameState = gameState
    self.lastEatenFood = agent.lastEatenFoodPosition
    self.startState = gameState.getAgentState(agent.index).getPosition()
    self.walls = gameState.getWalls()

  def isGoalState(self, state):
    return state == self.lastEatenFood

class RetreatToTurfSearchProblem(BaseSearchProblem):
  """
   Returning Home when only 2 foods left on the map and the agent is carrying some
  """

  def __init__(self, gameState, agent):
    BaseSearchProblem.__init__(self, gameState, agent)
    "Stores information from the gameState.  You don't need to change this."
    self.foodLeft = len(self.food.asList())
    self.homeBoundary = agent.homeBoundary


  def isGoalState(self, state): # safe when returned to home area or has the capsule
    return state in self.homeBoundary


# ------------------------------- from assignment 2 reflex agent, might be useful for heuristic calculation

# can replace the manhattanDistance with the provided real distance function
from util import manhattanDistance

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



# taken from Assignment 2 Q5,
# split this into multiple features and weights and adopt to this issue as heuristic

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    """
    # Useful information you can extract from a GameState (pacman.py)
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStatesList = currentGameState.getGhostStates()
    scaredTimesList = [ghostState.scaredTimer for ghostState in ghostStatesList]
    gameScore = currentGameState.getScore()
    # print ("currentGameState:" , currentGameState, "pacmanPos:" , pacmanPos, "foodGrid:\n" , foodGrid ,"ghostStatesList:" , ghostStatesList, "scaredTimesList:" , scaredTimesList,)

    # Calculations

    # find distance to closest ghost
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

    # Establishing rules from raw sensor input

    # Generally, higher distance from ghost is better, especially when ghosts are nearby.
    # This is unless the capsule has been eaten which means Ghosts can be pursued
    if min(scaredTimesList) == 0 :
        ghostDistRule = - 4.0 / (minGhostDistance + 1)
    else :
        ghostDistRule = 1.0 / (minGhostDistance + 1)

    # Generally, lower distance from food is preferred
    foodDistRule = 1.0 / (minFoodDistance + 1)

    # Generally, lower remaining food is better
    remainingFoodRule = -remainingFoodNum

    # Generally, higher is better
    gameScoreRule = gameScore

    evaluationResult = ghostDistRule + foodDistRule + remainingFoodRule + gameScoreRule
    # print ("evaluationResult", evaluationResult, "newPos", newPos, "ghostDistRule", ghostDistRule, "foodDistRule", foodDistRule)
    return evaluationResult




##------------------------------ Qlearning Agent CLASSES based FROM PROJECT 3 (REINFORCEMENT LEARNING)-------------------------------------------##

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
