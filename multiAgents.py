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
        # Position of the pacman
        newPos = successorGameState.getPacmanPosition()
        # Grid of food
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        # State of each ghosts
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        score = successorGameState.getScore()
        distance_min_ghost = float("+inf")

        for ghost_position in successorGameState.getGhostPositions():
            distance_min_ghost = min(
                                        distance_min_ghost,
                                        util.manhattanDistance(newPos, ghost_position)
                                    )

        # First get the current state
        if distance_min_ghost == 0:
            return float("-inf")
        if successorGameState.isWin():
            return float("+inf")

        # go far away from the ghost
        score += 2*distance_min_ghost
        
        distance_min_food = float("+inf")
        for foodPosition in newFood:
            distance_min_food = min(distance_min_food, util.manhattanDistance(foodPosition,newPos))
        
        # go to the food
        score -= 2*distance_min_food
        
        # If the action eat a food dont add point to the score
        if(successorGameState.getNumFood() < currentGameState.getNumFood()):
            score += 5
        
        # Penaliza las acciones de detención
        if action == Directions.STOP:
            score -= 10

        return score

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
        "*** YOUR CODE HERE ***"
        # AgentIndex = 0 is Pacman, AgentIndex = 1 is Ghost_1 and AgentIndex = 2 is Ghost_2
        # Format of result = [score, action]
        result = self.get_value(gameState, 0, 0)

        # Return the action from result
        return result[1]

    def get_value(self, gameState, index, depth):
        # Terminal states are declared as if we dont have movement to make or we achieve our goal depth:
        
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return gameState.getScore(), ""
            # return self.evaluationFunction(gameState)

        # maximaze the value for index = 0, because It is the pacman
        if index == 0:
            return self.max_value(gameState, index, depth)

        # minimaze the value for index > 0, because It is the 
        else:
            return self.min_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
    
        max_value = float("-inf")
        max_action = ""

        successor_index_ghost = index + 1
        currently_depth = depth
        
        # Actions that the pacman can make in this gameState
        # Generate the succesor for each actions
        for action in gameState.getLegalActions(index):
            successor = gameState.generateSuccessor(index, action)

            # Evaluate if the succesor index is equal to the amount of ghosts and if its a pacman update the index and depth.
            if successor_index_ghost == gameState.getNumAgents():
                successor_index_ghost = 0
                currently_depth += 1

            # We get the value of the ghost or the pacman
            recursive_value = self.get_value(successor, successor_index_ghost, currently_depth)[0]

            # We update the action and value if we get a max value than the last one
            if recursive_value > max_value:
                max_value = recursive_value
                max_action = action

        return max_value, max_action

    def min_value(self, gameState, index, depth):
    
        min_value = float("inf")
        min_action = ""

        # Increase the successor index the result could be the pacman or another ghost 
        successor_index = index + 1
        # We set the depth that we are
        currently_depth = depth

        # We get the actions which ghosts can do
        for action in gameState.getLegalActions(index):
            # We generate the next scenario that we create with taking a action 
            successor = gameState.generateSuccessor(index, action)

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                currently_depth += 1

            current_value = self.get_value(successor, successor_index, currently_depth)[0]

            if current_value < min_value:
                min_value = current_value
                min_action = action

        return min_value, min_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
