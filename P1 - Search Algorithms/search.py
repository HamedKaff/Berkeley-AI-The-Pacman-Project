# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    frontier = util.Stack()
    visited = []
    start_state = problem.getStartState()
    frontier.push((start_state, []))
    path = []

    while not frontier.isEmpty():

        node, node_actions = frontier.pop()
        visited.append(node)
        is_goal = problem.isGoalState(node)

        if is_goal:
            return node_actions

        successors = problem.getSuccessors(node)

        for successor, direction, cost in successors:
            if not successor in visited:
                action = node_actions + [direction]
                frontier.push((successor, action))
                path = action

    return path
    util.raiseNotDefined()

def breadthFirstSearch(problem):

    frontier = util.Queue()
    visited = []
    path = []
    start_state = problem.getStartState()
    frontier.push((start_state, []))

    while not frontier.isEmpty():

        node, node_actions = frontier.pop()
        is_goal = problem.isGoalState(node)

        if is_goal:
            return node_actions

        if not node in visited:
            visited.append(node)

            successors = problem.getSuccessors(node)
            for successor, direction, cost in successors:
                action = node_actions + [direction]
                frontier.push((successor, action))
                path = action

    return path
    util.raiseNotDefined()

def uniformCostSearch(problem):

    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    visited = []
    path = []
    start_state = problem.getStartState()
    priority = 0
    frontier.push((start_state, []), priority)

    while not frontier.isEmpty():

        node, node_actions = frontier.pop()
        is_goal = problem.isGoalState(node)

        if is_goal:
            return node_actions

        if not node in visited:
            visited.append(node)

            successors = problem.getSuccessors(node)
            for successor, direction, cost in successors:
                action = node_actions + [direction]
                priority_cost = problem.getCostOfActions(action)
                frontier.push((successor, action), priority_cost)
                path = action

    return path
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    visited = []
    path = []
    start_state = problem.getStartState()
    initial_h_of_n = heuristic(start_state, problem)
    frontier.push((start_state, path,0), initial_h_of_n)

    while not frontier.isEmpty():

        node, node_actions, node_cost = frontier.pop()
        is_goal = problem.isGoalState(node)

        if is_goal:
            return node_actions

        if not node in visited:
            visited.append(node)

            successors = problem.getSuccessors(node)
            for successor, direction, cost in successors:

                action = node_actions + [direction]
                g_of_n = cost + node_cost
                h_of_n = heuristic(successor, problem)
                f_of_n = g_of_n + h_of_n

                frontier.push((successor, action, g_of_n), f_of_n)
                path = action

    return path
    util.raiseNotDefined()

#####################################################
# EXTENSIONS TO BASE PROJECT
#####################################################

def depthLimitiedSearch(problem, l):
    frontier = util.Stack()
    visited = []
    failure = []
    result = failure
    start_state = problem.getStartState()
    frontier.push((start_state, [], 0))

    while not frontier.isEmpty():
        node, actions, cost = frontier.pop()

        if problem.isGoalState(node):
            return actions

        if len(actions) > l:
            result = 'cutoff'

        elif node not in visited:
            visited.append(node)

            for successor, direction, child_cost in problem.getSuccessors(node):
                action = actions + [direction]
                new_cost = cost + child_cost
                frontier.push((successor, action, new_cost))
    return result


# Extension Q1e
def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE ***"
    depth = 0
    while True:
        result = depthLimitiedSearch(problem, depth)
        if result != 'cutoff':
            return result

        depth += 1

    util.raiseNotDefined()

# Extension Q2e
def enforcedHillClimbing(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

#####################################################
# Abbreviations
#####################################################
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
ehc = enforcedHillClimbing
