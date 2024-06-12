package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to.
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.3;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=15000;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.1;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair, you can do
	 * qTable.get(game).get(move) which return the Q(game,move) value stored. Be careful with 
	 * cases where there is currently no value. You can use the containsKey method to check if the mapping is there.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount)
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent()
	{
		this(new RandomAgent(), 0.3, 15000, 0.9);
		
	}
	
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 */
	
	public void train()
	{
		try {
			
			for (int i=0;i<=this.numEpisodes;i++) {
				
				//start from start game
				Game g = this.env.getCurrentGameState();
				
				while(g.isTerminal() == false) {
					
					//get a move according to an epsilon-greedy policy
					Move m = getMove(g);					
				    double currentQValue = this.qTable.getQValue(g, m);
				    
				    //execute move (careful: g becomes gPrime automatically)
					Outcome mOutcome = this.env.executeMove(m);
					double reward = mOutcome.localReward;
					Game gPrime = mOutcome.sPrime;
					
					//calculate maxQ'(s',a')
					List <Move> movesSprime =  gPrime.getPossibleMoves();
					double primeMaxValue = -9999;
					if(gPrime.isTerminal() == false) {
						for(Move mPrime: movesSprime) {
							double currentValueprime = this.qTable.getQValue(gPrime, mPrime);
							if(currentValueprime > primeMaxValue)
								primeMaxValue = currentValueprime;
						}
						
					} else {
						primeMaxValue = 0.0;
					} //close if is terminal
					
					//update qValue and add to qTable
					currentQValue = (1 - this.alpha)*currentQValue + this.alpha*(reward + this.discount*primeMaxValue);
					this.qTable.addQValue(mOutcome.s, m, currentQValue);
					
				} //close while loop
				
				//reset episode
				this.env.resetEpisode();
				
			} //close numEpisodes for loop
			
		
			//--------------------------------------------------------
			//you shouldn't need to delete the following lines of code.
			this.policy=extractPolicy();
			if (this.policy==null)
			{
				System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
				//System.exit(1);
			}
		
		} catch (IllegalMoveException e) {
			e.printStackTrace();
		} //close try/catch
	}
	
	// The agent plays (moves) according to an epsilon-greedy policy
	public Move getMove (Game g) 
	{
		Random rand = new Random();
	    double randomNumber = rand.nextDouble(); // generates a random number between 0 and 1
	    Move nextMove = null;
	    List <Move> moves =  g.getPossibleMoves(); //all valid moves of game g
	    
	    if (randomNumber < this.epsilon) { //random move
	    	nextMove = moves.get(rand.nextInt(moves.size()));
	    }
	    else { //optimal move
			double maxValue = -9999;
			Move maxMove = null;
			for(Move m: moves) {
				double currentValue = this.qTable.getQValue(g, m);
				if(currentValue > maxValue) {
					maxValue = currentValue;
					maxMove = m;
				}
			}
			
			nextMove = maxMove;
	    }
	    
		return nextMove;
	}
	
	
	
	
	
	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		Set <Game> allGames = this.qTable.keySet();
		HashMap<Game, Move> optimalMoves = new HashMap<Game, Move>();
		
		for(Game g: allGames) {
			
			List<Move> moves = g.getPossibleMoves();
			double maxValue = -9999;
			Move maxMove = null;
			
			if(g.isTerminal() == false) {
				for(Move m: moves) {
					double currentValue = this.qTable.getQValue(g, m);
					if(currentValue > maxValue)
						maxValue = currentValue;
						maxMove = m;
				}
			}
			
			optimalMoves.put(g, maxMove);
			
		}
		
		Policy pol = new Policy(optimalMoves);
		return pol;
		
	}
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
	
	
	


	
}
