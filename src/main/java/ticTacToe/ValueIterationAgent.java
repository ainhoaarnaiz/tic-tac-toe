package ticTacToe;


import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction = new HashMap<Game, Double>();
	HashMap<Game, Move> optimalMoves = new HashMap<Game, Move>();
	
	/**
	 * the discount factor
	 */
	double discount=0.9;
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k=10;
	
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
		
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues();
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);
		
		
		
	}
	
	
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 
	
	/*
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 * 
	 *
	 */
	public void iterate()
	{
		
		for (int i=0;i<=k;i++) {
	
			Set <Game> allGames = this.valueFunction.keySet();
			
			for(Game g: allGames) {

				List<Move> moves = g.getPossibleMoves();
				double maxValue = -9999;
				Move maxMove = null;
				
				if (g.isTerminal() == true){
					this.valueFunction.put(g, 0.0);
				}
				
				else {
					for(Move m: moves) {

						List <TransitionProb> pro = this.mdp.generateTransitions(g, m);
						int s = pro.size();
						double currentValue = 0;
						
						for (int j=0; j<s; ++j) {
							double probability = pro.get(j).prob;
							double reward = pro.get(j).outcome.localReward;
							Game sPrime = pro.get(j).outcome.sPrime;							
							currentValue += probability*(reward + this.discount*this.valueFunction.get(sPrime));
						} //close transitions for loop
						
						if (currentValue > maxValue) {
							maxValue = currentValue;
							maxMove = m;
						} //close max value if condition

					} //close moves loop
					
					//update value and move
					this.valueFunction.put(g, maxValue);
					this.optimalMoves.put(g, maxMove);
					
				} //close if is terminal
				
			} //close games for loop
			
		}  //close k iterations for loop
	
	}
	
	/**This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy()
	{
		//create a policy from optimalMoves HashMap
		Policy pol = new Policy(this.optimalMoves);
		return pol;
	}
	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
		 *  
		 */
		
		super.policy = extractPolicy();
		
		if (this.policy == null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
		
		
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
}
