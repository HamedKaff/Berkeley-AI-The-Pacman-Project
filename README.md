# Artificial Intelligence  - The Pacman Project AI21@RMIT COSC1127

## Project 1 - Search

### Uninformed Search Algorithms:

Breadth-First Search (BFS) <br />
Depth-First Search (DFS)<br />
Uniform Cost Search (UCS)<br />
Depth Limited Search (DLS)<br />
Iterative Deepening Search (IDS)<br />

### Informed Search Algorithms:
- **A* Search:** Used the Manhattan distance heuristic to find the optimal solution.

### Other:

- CornersProblem: Search problem and a heuristic function for pacman to reach all active corner dots on board.

- FoodSearchProblem: Search problem and heuristic for pacman to eat all active dots on board.


## Project 2 - MultiAgents

- **ReflexAgent:** A reflex agent uses an evaluation function (a heuristic function), to estimate the value of an action using the current game state.

- **MinimaxAgent:** A minimax agent is implemented using a minimax tree with multiple min layers for every max layer. The agent uses a heuristic function which evaluates the states.

- **AlphaBetaAgent:** An alpha beta agent uses alpha-beta pruning to explore the minimax tree.

- **Expectimax:** The expectimax pacman makes decisions using the expected value.


## Project 3 - Contest -> Reinforcement Learning

- **Q-Learning:** a reinforcement learning agent starts off not performing good, but instead learns by trial and error from interactions with the environment through its update(state, action, nextState, reward) method, and evetually it gets better.

- **Decision Tree:** Created a decision tree along with Q learning to optimize the agent's performance in the contest.
