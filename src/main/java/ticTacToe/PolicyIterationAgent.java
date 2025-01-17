package ticTacToe;


import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;
/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues=new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
		
		
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP paramters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);
		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 */
	public void initRandomPolicy()
	{
		Set <Game> allGames = this.policyValues.keySet();
		for(Game g: allGames) {
			if(g.isTerminal() == false) {
				List <Move> moves = g.getPossibleMoves(); //all valid moves of game g.
				Random rand = new Random();
				Move randomMove = moves.get(rand.nextInt(moves.size()));
				this.curPolicy.put(g, randomMove);
			}
			
		}
	}
	
	
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@code delta}, in other words
	 * until the values under the current policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta)
	{
		
		boolean allConverge = false;
		Set <Game> allGames = this.policyValues.keySet();
		
		while (allConverge == false) {
			
			allConverge = true;
			
			for(Game g: allGames) {
				
				double updateg = 9999;
				
				if(g.isTerminal() == false) {
					
					//get a valid random move of g
					Move randomMove = this.curPolicy.get(g);
					double currentValue = 0;
					List <TransitionProb> pro = this.mdp.generateTransitions(g, randomMove);
					int s = pro.size();
					
					for (int j=0; j<s; ++j) {
						
						double probability = pro.get(j).prob;
						double reward = pro.get(j).outcome.localReward;
						Game sPrime = pro.get(j).outcome.sPrime;							
						currentValue += probability*(reward + this.discount*this.policyValues.get(sPrime));
						
						//difference between policy value of g and the calculated one for g
						double difference = this.policyValues.get(g) - currentValue;
						//get the smallest difference
						if (updateg > difference)
                            updateg = difference;
						
					} //close transitions for loop
					
					this.policyValues.put(g, currentValue);
					//if the difference of one g is larger than delta, then we haven't reach convergence of all the games yet
					if (updateg > delta)
						allConverge = false;
				
				} //close if is Terminal
				
			} //close games for loop
		
		} //close while of convergence
			
	}
		
	
	
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy()
	{
		
		Set <Game> allGames = this.curPolicy.keySet();
		boolean policyChange = false;
		
			for(Game g: allGames) {
				
				if(g.isTerminal() == false) {
					
					List<Move> moves = g.getPossibleMoves();
					double maxValue = -9999;
                    Move maxMove = null;

					for(Move m: moves) {
						
						List <TransitionProb> pro = this.mdp.generateTransitions(g, m);
						int s = pro.size();
						double currentValue = 0;
						
						for (int j=0; j<s; ++j) {
							double probability = pro.get(j).prob;
							double reward = pro.get(j).outcome.localReward;
							Game sPrime = pro.get(j).outcome.sPrime;							
							currentValue += probability*(reward + this.discount*this.policyValues.get(sPrime));
						} //close transitions for loop
						
						//get max value and optimal move
						if (currentValue > maxValue) {
                            maxValue = currentValue;
                            maxMove = m;
						} //close max value if condition
		
					} //close moves for loop
					
					//if the policy value of g is smaller than the calculated maximum value, then policy changes (there is no convergence)
					if (this.policyValues.get(g) < maxValue) {
						this.curPolicy.put(g, maxMove);
                        policyChange = true;
					} //close policy change if condition
					
				} //close if is terminal
					
			} //close games for loop
		
		return policyChange;
	}
	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{
		boolean policyChange = true;
		
		//repeat steps until convergence (until policy doesn't change)
		while (policyChange == true) {
			evaluatePolicy(this.delta);
			policyChange = improvePolicy();
		}
		
		super.policy = new Policy(this.curPolicy);
	}
	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();
		
		
	}
	

}
