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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    import math
    inf = math.inf

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

        return legalMoves[chosenIndex]

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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        # Average Score: 1240.2
        food = currentGameState.getFood()
        min_dist = self.inf

        for ghost in newGhostStates:
            ghost_pos = ghost.getPosition()
            ghost_dist = manhattanDistance(newPos, ghost_pos)
            if ghost.scaredTimer > 0 and ghost_dist == 0:
                return self.inf
            elif ghost.scaredTimer > ghost_dist:
                min_dist = min(min_dist, ghost_dist)
            elif ghost_dist < 2:
                return -self.inf

        if food[newPos[0]][newPos[1]]:
            return self.inf

        min_dist = min([min(min_dist, manhattanDistance(newPos, food)) \
                        for food in newFood.asList()])
            
        return 1 / float(min_dist)


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
        import math
        self.inf = math.inf

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
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState)


    def max_value(self, gameState, agent= 0, depth = 0):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent)
        v, my_action = -self.inf, None
        for action in actions:
            successor = gameState.generateSuccessor(agent, action)
            new_v = self.min_value(successor, agent + 1, depth)
            if new_v > v:
                v, my_action = new_v, action
        return my_action if depth == 0 else v


    def min_value(self, gameState, agent, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent)
        v = self.inf
        for action in actions:
            successor = gameState.generateSuccessor(agent, action)
            v = min(v, self.get_value(gameState, successor, agent, depth))
        return v


    def get_value(self, gameState, successor, agent, depth):
        return self.max_value(successor, 0, depth + 1) \
        if agent + 1 == gameState.getNumAgents() \
        else self.min_value(successor, agent + 1, depth)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, -self.inf, self.inf)


    def max_value(self, gameState, alpha, beta, agent=0, depth=0):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent)
        v, my_action = -self.inf, None
        for action in actions:
            successor = gameState.generateSuccessor(agent, action)
            new_v = self.min_value(successor, alpha, beta, agent + 1, depth)
            if new_v > v:
                v, my_action = new_v, action
            if v > beta:
                return v
            alpha = max(alpha, v)
        return my_action if depth == 0 else v


    def min_value(self, gameState, alpha, beta, agent, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent)
        v = self.inf
        for action in actions:
            successor = gameState.generateSuccessor(agent, action)
            v = min(v, self.get_value(gameState, successor, alpha, beta, agent, depth))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v


    def get_value(self, gameState, successor, alpha, beta, agent, depth):
        return self.max_value(successor, alpha, beta, 0, depth + 1) \
        if agent + 1 == gameState.getNumAgents() \
        else self.min_value(successor, alpha, beta, agent + 1, depth)
        


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState)


    def max_value(self, gameState, agent= 0, depth = 0):
        actions = gameState.getLegalActions(agent)
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        v, my_action = -self.inf, None
        for action in actions:
            successor = gameState.generateSuccessor(agent, action)
            new_v = self.avg_value(successor, agent + 1, depth)
            if new_v > v:
                v, my_action = new_v, action
        return my_action if depth == 0 else v


    def avg_value(self, gameState, agent, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent)
        return sum([self.get_value(gameState, 
        gameState.generateSuccessor(agent, action),
         agent, depth) for action in actions]) / len(actions)


    def get_value(self, gameState, successor, agent, depth):
        return self.max_value(successor, 0, depth + 1) \
        if agent + 1 == gameState.getNumAgents() \
        else self.avg_value(successor, agent + 1, depth)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Average Score: 1047.4
    score = currentGameState.getScore()
    position = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    for ghost in ghostStates:
        ghost_pos = ghost.getPosition()
        ghost_dist = manhattanDistance(position, ghost_pos)
        if ghost_dist:
            score +=  1 / float(ghost_dist)

    score += sum([(1 / float(manhattanDistance(position, food))) \
                for food in currentGameState.getFood().asList()]) 
    return score

    # score += sum([(1 / float(manhattanDistance(position, 
    #             ghost.getPosition()))) for ghost in ghostStates
    #             if float(manhattanDistance(position, ghost.getPosition()))])



# Abbreviation
better = betterEvaluationFunction
