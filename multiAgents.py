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
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth, alpha, beta):
            actions = gameState.getLegalActions(0)
            if len(actions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)

            value, goAction = -(float("inf")), None

            for action in actions:
                successorValue = min_value(gameState.generateSuccessor(0, action), 1, depth, alpha, beta)[0]
                if value < successorValue:
                    value, goAction = successorValue, action

                if value > beta:
                    return (value, goAction)

                alpha = max(alpha, value)

            return (value, goAction)

        def min_value(gameState, agentID, depth, alpha, beta):
            actions = gameState.getLegalActions(agentID)
            if len(actions) == 0:
                return (self.evaluationFunction(gameState), None)

            value, goAction = float("inf"), None

            for action in actions:
                if agentID == gameState.getNumAgents() - 1:
                    successorValue = max_value(gameState.generateSuccessor(agentID, action), depth + 1, alpha, beta)[0]
                else:
                    successorValue = min_value(gameState.generateSuccessor(agentID, action), agentID + 1, depth, alpha, beta)[0]
                if successorValue < value:
                    value, goAction = successorValue, action

                if value < alpha:
                    return (value, goAction)

                beta = min(beta, value)

            return (value, goAction)

        alpha, beta = -(float("inf")), float("inf")
        return max_value(gameState, 0, alpha, beta)[1]



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

        def max_value(game_state, depth):
            available_actions = game_state.getLegalActions(0)
            if len(available_actions) == 0 or game_state.isWin() or game_state.isLose() or depth == self.depth:
                return (self.evaluationFunction(game_state), None)
            
            max_value, selected_action = -(float("inf")), None

            for action in available_actions:
                successor_value = exp_value(game_state.generateSuccessor(0, action), 1, depth)[0]
                if max_value < successor_value:
                    max_value, selected_action = successor_value, action

            return (max_value, selected_action)

        def exp_value(game_state, agent_id, depth):
            available_actions = game_state.getLegalActions(agent_id)
            if len(available_actions) == 0:
                return (self.evaluationFunction(game_state), None)

            expected_value, selected_action = 0, None

            for action in available_actions:
                if agent_id == game_state.getNumAgents() - 1:
                    successor_value = max_value(game_state.generateSuccessor(agent_id, action), depth + 1)[0]
                else:
                    successor_value = exp_value(game_state.generateSuccessor(agent_id, action), agent_id + 1, depth)[0]

                probability = successor_value / len(available_actions)
                expected_value += probability

            return (expected_value, selected_action)

        return max_value(gameState, 0)[1]

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: First, we return positive or negative infinity, depending on whether the game state indicates a victory or a defeat. 
    Next, we figure out how far it is to the nearest food pellet and penalise Pac-Man by taking the inverse of this distance. 
    We also take into consideration, with commensurate penalties, the distance to the nearest scared and non-scared ghost. 
    We also penalise the overall amount of capsules and food pellets left, hoping to incentivize Pac-Man to finish them. 
    To fine-tune Pac-Man's behaviour in various game conditions, you can change the weights attached to these penalties.
    Pac-Man is encouraged to prioritise nearby food, steer clear of non-scared ghosts, 
    and specifically target scared ghosts due to negative weights assigned to distances to food, non-scared ghosts, and scared ghosts. 
    We modified the weights by trying diferent executations in the map provided to find the configuration that suits better to it.
    """
    "*** YOUR CODE HERE ***"

    pacmanPos = currentGameState.getPacmanPosition()
    ghostList = currentGameState.getGhostStates()
    foods = currentGameState.getFood()
    capsules = currentGameState.getCapsules()

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    foodDistList = [util.manhattanDistance(each, pacmanPos) for each in foods.asList()]
    minFoodDist = min(foodDistList) if foodDistList else 0
    ghostDistList = [util.manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostList if ghost.scaredTimer == 0]
    scaredGhostDistList = [util.manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostList if ghost.scaredTimer > 0]
    minGhostDist = min(ghostDistList) if ghostDistList else float("inf")
    minScaredGhostDist = min(scaredGhostDistList) if scaredGhostDistList else float("inf")

    score = scoreEvaluationFunction(currentGameState)

    # Distance to closest food
    score += (-1.5 * minFoodDist)

    # Distance to closest ghost
    if minGhostDist < float("inf"):
        score += (-2 * (1.0 / minGhostDist))

    # Distance to closest scared ghost
    if minScaredGhostDist < float("inf"):
        score += (-2 * minScaredGhostDist)

    # Number of capsules
    score += (-20 * len(capsules))

    # Number of food
    score += (-4 * len(foods.asList()))

    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
