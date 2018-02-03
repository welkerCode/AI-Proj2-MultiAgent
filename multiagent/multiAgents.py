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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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



        foodList = currentGameState.getFood()
        foodList = foodList.asList()

        eatReward = 0
        minPenalty = 0
        ghostPenalty = 10
        corneredPenalty = 0
        stationaryPenalty = 0
        remainingFoodPenalty = 0
        pathPenalty = 0

        if newPos in foodList:
            eatReward = 25

        score = successorGameState.getScore()
        closest = (0,0)
        distance = 100
        for food in foodList:
            if util.manhattanDistance(newPos, food) < distance:
                distance = util.manhattanDistance(newPos, food)
                closest = food
            remainingFoodPenalty += distance

        if distance == 0:
            minPenalty = 0
        elif distance < 6:
            "*******************************"
            queue = util.Queue()  # DFS requires a queue
            initNode = (successorGameState, newPos, 0)  # Get the initial state and save it in a Search Node
            queue.push(initNode)  # Put the initial node onto the queue
            visited = []
            while not queue.isEmpty():  # Until the queue is empty (and we fail to find the goal)
                nextStateNode = queue.pop()  # Get the next node off of the queue
                nextState = nextStateNode[1]  # Save its state in a local variable
                if nextState not in visited:  # If the state has not been visited previously
                    visited.append(nextState)  # Add it to the visited list
                    if nextState in foodList:  # If the state is a goal state
                        minPenalty = nextStateNode[2]
                        queue.list = []
                    # If we have reached this part of the DFS, then the current state is not a goal state
                    else:
                        for action in nextStateNode[0].getLegalActions():  # For every possible action and potential new state

                            successor = nextStateNode[0].generatePacmanSuccessor(action)
                            position = successor.getPacmanPosition()

                            successorNode = (successor, position, nextStateNode[2] + 1)  # Convert to Search Node
                            queue.push(successorNode)  # Push onto queue
            "************************************"
        else:
            minPenalty = distance


        for ghost in xrange(len(newGhostStates)):
            penalty = util.manhattanDistance(newPos, newGhostStates[ghost].getPosition())
            if (penalty + 1) < newScaredTimes[ghost]:
                ghostPenalty = -(penalty+1) / newScaredTimes[ghost]
            elif penalty == 0:
                ghostPenalty = .14
            elif ghostPenalty > penalty:
                ghostPenalty = penalty


        if distance > 1:
            corneredPenalty = len(successorGameState.getLegalActions())

        if currentGameState.getPacmanPosition() == newPos:
            stationaryPenalty = 150

        score = score + eatReward - 10* minPenalty - 50 / (ghostPenalty ** 2) - distance*5/(corneredPenalty**2+1) - stationaryPenalty - remainingFoodPenalty - pathPenalty
        return (score)





        return successorGameState.getScore()

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
        """
        "*** YOUR CODE HERE ***"

        (value, action) = self.value(0,gameState,0)
        print 'Value is:' + str(value) + '        Depth is:' + str(self.depth)
        return(action)
        util.raiseNotDefined()



    def value(self, agent, gameState, depth):
        if agent == gameState.getNumAgents():
            depth +=1
            agent = 0
        if agent == 0:
            result = self.maxValue(agent,gameState,depth)
        else:
            result = self.minValue(agent,gameState,depth)
        return result

    def maxValue(self, agent, gameState, depth):
        value = -9999999
        choice = 'Center'
        if depth == self.depth:
            return (self.evaluationFunction(gameState), choice)
        if len(gameState.getLegalActions(0)) == 0:
            return (self.evaluationFunction(gameState), choice)
        for action in gameState.getLegalActions(0):
            next = gameState.generateSuccessor(0, action)

            newValue =  max(value, self.value(agent+1, next, depth))
            if newValue[0] > value:
                choice = action
            value = max(value, newValue[0])
        return (value, choice)

    def minValue(self, agent, gameState, depth):
        value = 999999
        if len(gameState.getLegalActions(agent)) == 0:
            return (self.evaluationFunction(gameState),None)
        for action in gameState.getLegalActions(agent):
            ghost = gameState.generateSuccessor(agent, action)
            result = self.value(agent+1, ghost, depth)
            value = min(value,result[0])
        return (value, None)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        (value, action) = self.value(0,gameState,0)
        print 'Value is:' + str(value) + '        Depth is:' + str(self.depth)
        return(action)


        util.raiseNotDefined()


    def value(self, agent, gameState, depth, alpha = -999999,beta = 9999999 ):
        if agent == gameState.getNumAgents():
            depth += 1
            agent = 0
        if agent == 0:
            result = self.maxValue(agent, gameState, depth, alpha, beta)
        else:
            result = self.minValue(agent, gameState, depth, alpha, beta)
        return result

    def maxValue(self, agent, gameState, depth, alpha, beta):
        value = -9999999
        choice = 'Center'
        if depth == self.depth:
            return (self.evaluationFunction(gameState), choice)
        if len(gameState.getLegalActions(0)) == 0:
            return (self.evaluationFunction(gameState), choice)
        for action in gameState.getLegalActions(0):
            next = gameState.generateSuccessor(0, action)
            newValue = max(value, self.value(agent + 1, next, depth, alpha, beta))
            if newValue[0] > beta:
                return (newValue[0], choice)
            if newValue[0] > value:
                choice = action
            value = max(value, newValue[0])
            alpha = max(alpha, value)
        return (value, choice)

    def minValue(self, agent, gameState, depth, alpha, beta):
        value = 999999
        if len(gameState.getLegalActions(agent)) == 0:
            return (self.evaluationFunction(gameState), None)
        for action in gameState.getLegalActions(agent):
            ghost = gameState.generateSuccessor(agent, action)
            result = self.value(agent + 1, ghost, depth, alpha, beta)
            value = min(value, result[0])
            if value < alpha:
                return (value,None)
            beta = min (beta, value)
        return (value, None)



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

        (value, action) = self.value(0, gameState, 0)
        print 'Value is:' + str(value) + '        Depth is:' + str(self.depth)
        return (action)
        util.raiseNotDefined()

    def value(self, agent, gameState, depth):
        if agent == gameState.getNumAgents():
            depth += 1
            agent = 0
        if agent == 0:
            result = self.maxValue(agent, gameState, depth)
        else:
            result = self.minValue(agent, gameState, depth)
        return result

    def maxValue(self, agent, gameState, depth):
        value = -99999
        choice = 'Center'
        if depth == self.depth:
            return (self.evaluationFunction(gameState), choice)
        if len(gameState.getLegalActions(0)) == 0:
            return (self.evaluationFunction(gameState), choice)
        for action in gameState.getLegalActions(0):
            next = gameState.generateSuccessor(0, action)

            newValue = max(value, self.value(agent + 1, next, depth))
            if newValue[0] > value:
                choice = action
            value = max(value, newValue[0])
        return (value, choice)

    def minValue(self, agent, gameState, depth):
        value = 0.0
        if len(gameState.getLegalActions(agent)) == 0:
            return (self.evaluationFunction(gameState), None)
        for action in gameState.getLegalActions(agent):
            ghost = gameState.generateSuccessor(agent, action)
            result = self.value(agent + 1, ghost, depth)
            value += result[0]
        value = value/len(gameState.getLegalActions(agent))
        return (value, None)


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

