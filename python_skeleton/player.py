'''
Improved pokerbot for B4G Hold'em variant
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import eval7
import time  # For performance tracking

class QLearningAgent():
    def __init__(self, epsilon=0.05, discount=0.80, alpha=0.2, numTraining=20, weights=False):
        self.epsilon = epsilon
        self.discount = discount
        self.alpha = alpha
        self.numTraining = numTraining
        if weights == False:
            self.weights = {}
        else:
            self.weights = weights

    def getWeights(self):
        return self.weights

    def getLegalActions(self, state):
        if state == False or state[1] == False:
            return [CheckAction(), FoldAction(), CallAction()]
        thisState = list(state)
        game_state = thisState[0]
        round_state = thisState[1]
        active = thisState[2]
        legal_actions = round_state.legal_actions()
        myActions = []
        my_pip = round_state.pips[active]
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
            min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
            max_cost = max_raise - my_pip  # the cost of a maximum bet/raise
            myActions += [RaiseAction(min_raise + (max_raise - min_raise) * .1)]
            myActions += [RaiseAction(min_raise + (max_raise - min_raise) * .5)]
            myActions += [RaiseAction(min_raise + (max_raise - min_raise) * .8)]
        if CheckAction in legal_actions:  # check-call
            myActions += [CheckAction()]
        if random.random() < 0.25:
            myActions += [FoldAction()]
        myActions += [CallAction()]
        return myActions

    def evaluateHandStrength(self, my_cards, board_cards, num_simulations=10):
        """ 
        Monte Carlo simulation to estimate hand strength using eval7.
        Adapted for B4G Hold'em with 3 hole cards.
        """
        deck = [card for card in eval7.Deck() if str(card) not in my_cards + board_cards]
        my_hand = [eval7.Card(str(c)) for c in my_cards]
        board = [eval7.Card(str(c)) for c in board_cards]

        wins = 0
        for _ in range(num_simulations):
            random.shuffle(deck)
            # In B4G, opponent has 3 cards
            opp_hand = deck[:3]
            
            # In B4G, we need to calculate the remaining board cards differently
            # If no board cards yet, we'll add 4 community cards
            # If we have some board cards, we'll add enough to make 4 total
            remaining_board = deck[3:7 - len(board)]
            
            full_board = board + remaining_board
            
            # We need to find the best 5-card hand out of our 3 cards and the 4 community cards
            my_best_hand_value = self.get_best_hand_value(my_hand, full_board)
            opp_best_hand_value = self.get_best_hand_value(opp_hand, full_board)

            if my_best_hand_value > opp_best_hand_value:
                wins += 1

        return wins / num_simulations  # Returns estimated win probability
    
    def get_best_hand_value(self, hole_cards, community_cards):
        """
        Find the best 5-card hand value from hole cards and community cards.
        """
        all_cards = hole_cards + community_cards
        return eval7.evaluate(all_cards)

    def getFeatures(self, state):
        features = {}
        if state == False or state[1] == False:
            features["hand_strength"] = 0
            features["pot_odds"] = 0
            features["effective_stack"] = 0
            features["opp_aggression"] = 0
            features["position"] = 0
            features["street"] = 0
            features["bluff_potential"] = 0
            return features
        thisState = list(state)
        game_state = thisState[0]
        round_state = thisState[1]
        active = thisState[2]

        # 1. Hand Strength (calculated using hand evaluator)
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:round_state.street]  # Cards revealed so far
        features["hand_strength"] = self.evaluateHandStrength(my_cards, board_cards)

        # 2. Pot Odds
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        pot_size = sum(round_state.pips)
        continue_cost = opp_pip - my_pip
        features["pot_odds"] = continue_cost / (pot_size + continue_cost) if continue_cost > 0 else 0

        # 3. Effective Stack Size (Relative to Big Blind)
        my_stack = round_state.stacks[active]
        features["effective_stack"] = my_stack / BIG_BLIND

        # 4. Opponent's Betting Behavior (Aggression Factor)
        opp_contribution = STARTING_STACK - round_state.stacks[1 - active]
        features["opp_aggression"] = opp_contribution / (pot_size + 1)  # +1 to avoid division by zero

        # 5. Position (1 if acting last, 0 if first)
        features["position"] = 1 if active == 1 else 0

        # 6. Street (adjusted for B4G: 0 = pre-flop, 1 = flop, 2 = final)
        features["street"] = round_state.street / 4  # Normalize between 0 and 1

        # 7. Bluff Potential (Simple metric: If opponent has checked, it might be a bluff spot)
        features["bluff_potential"] = 1 if CheckAction in round_state.legal_actions() else 0

        return features

    def update(self, state, action, nextState, reward):
        if self.numTraining > 0:
            self.numTraining -= 1
        else:
            self.epsilon = 0
        difference = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        features = self.getFeatures(state)
        for feature in features:
            thisWeight = 0
            if feature in self.weights:
                thisWeight = self.weights[feature]
            self.weights[feature] = thisWeight - self.alpha * difference * features[feature]

    def getQValue(self, state, action):
        features = self.getFeatures(state)
        output = 0
        for feature in features:
            thisWeight = 0
            if feature in self.weights:
                thisWeight = self.weights[feature]
            output += thisWeight * features[feature]
        return output
    
    def computeActionFromQValues(self, state):
        legalActions = [action for action in self.getLegalActions(state)]
        if len(legalActions) == 0:
            return None
        else:
            bestAction = [legalActions[0]]
            bestActionScore = -9999999999999999999
            for action in legalActions:
                thisActionScore = self.getQValue(state, action)
                if thisActionScore == bestActionScore:
                    bestAction += [action]
                elif thisActionScore > bestActionScore:
                    bestActionScore = thisActionScore
                    bestAction = [action]
            if random.random() < self.epsilon:
                return random.choice(legalActions)
            return random.choice(bestAction)
    
    def computeValueFromQValues(self, state):
        q_values = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        if len(q_values) == 0:
            return 0.0
        else:
            return max(q_values)


class Player(Bot):
    '''
    A pokerbot for B4G Hold'em.
    '''
    
    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.previousAction = False
        self.previousState = False
        self.myAgent = QLearningAgent(numTraining=100)  # Increased training rounds
        
        # Statistics and debugging info
        self.hand_counts = {'royal_flush': 0, 'straight_flush': 0, 'four_of_a_kind': 0, 
                            'full_house': 0, 'flush': 0, 'straight': 0, 
                            'three_of_a_kind': 0, 'two_pair': 0, 'pair': 0, 'high_card': 0}
        self.total_hands = 0
        self.wins = 0
        self.losses = 0
        
        # Save start time to track performance
        self.start_time = time.time()

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        # We can use this to initialize per-round state
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        thisGameState = (game_state, False, active)
        if self.myAgent.numTraining > 0:
            self.myAgent.update(self.previousState, self.previousAction, thisGameState, terminal_state.deltas[active])
        
        # Print weights occasionally for debugging
        if game_state.round_num % 500 == 0:
            print(f"Round {game_state.round_num}, Weights: {self.myAgent.getWeights()}")

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        # Get our cards and the board cards
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:round_state.street]
        
        # Check for premium hands that deserve special handling
        hand_rank = self.evaluate_hand_rank(my_cards, board_cards)
        
        # Handle premium hands directly with optimal strategy
        if hand_rank in ['royal_flush', 'straight_flush', 'four_of_a_kind', 'full_house']:
            return self.play_premium_hand(round_state, hand_rank)
        
        # For regular hands, use our Q-learning agent
        thisGameState = tuple([game_state, round_state, active])
        
        if self.myAgent.numTraining > 0 and self.previousState:
            self.myAgent.update(self.previousState, self.previousAction, thisGameState, 0)

        action = self.myAgent.computeActionFromQValues(thisGameState)
        self.previousAction = action
        self.previousState = thisGameState
        
        if isinstance(action, FoldAction):
            return FoldAction()
        elif isinstance(action, CallAction):
            return CallAction()
        elif isinstance(action, CheckAction):
            return CheckAction()
        elif isinstance(action, RaiseAction):
            return RaiseAction(int(action.amount))
            
        # Fallback strategy (should not normally reach here)
        legal_actions = round_state.legal_actions()
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()
        
    def evaluate_hand_rank(self, my_cards, board_cards):
        '''
        Evaluates the current hand rank (royal flush, straight flush, etc.)
        
        Arguments:
        my_cards: List of my hole cards
        board_cards: List of community cards
        
        Returns:
        String describing the hand rank
        '''
        if not board_cards:
            return 'high_card'  # Pre-flop
            
        # Convert cards to eval7 format
        my_hand = [eval7.Card(str(c)) for c in my_cards]
        board = [eval7.Card(str(c)) for c in board_cards]
        all_cards = my_hand + board
        
        # Get the hand rank value
        hand_value = eval7.evaluate(all_cards)
        
        # Map eval7 hand ranks to readable categories
        # Royal flush is a special case of straight flush
        if self.is_royal_flush(all_cards):
            return 'royal_flush'
        
        # Use the hand value to determine rank
        if 0x800000 <= hand_value <= 0x8FFFFF:  # Straight flush
            return 'straight_flush'
        elif 0x700000 <= hand_value <= 0x7FFFFF:  # Four of a kind
            return 'four_of_a_kind'
        elif 0x600000 <= hand_value <= 0x6FFFFF:  # Full house
            return 'full_house'
        elif 0x500000 <= hand_value <= 0x5FFFFF:  # Flush
            return 'flush'
        elif 0x400000 <= hand_value <= 0x4FFFFF:  # Straight
            return 'straight'
        elif 0x300000 <= hand_value <= 0x3FFFFF:  # Three of a kind
            return 'three_of_a_kind'
        elif 0x200000 <= hand_value <= 0x2FFFFF:  # Two pair
            return 'two_pair'
        elif 0x100000 <= hand_value <= 0x1FFFFF:  # Pair
            return 'pair'
        else:
            return 'high_card'
            
    def is_royal_flush(self, cards):
        '''
        Check if the hand is a royal flush (A, K, Q, J, 10 of the same suit)
        
        Arguments:
        cards: List of eval7.Card objects
        
        Returns:
        Boolean indicating if the hand is a royal flush
        '''
        # Need at least 5 cards
        if len(cards) < 5:
            return False
            
        # Group cards by suit
        suits = {}
        for card in cards:
            suit = card.suit
            if suit not in suits:
                suits[suit] = []
            suits[suit].append(card.rank)
            
        # Check if any suit has A, K, Q, J, 10
        for suit, ranks in suits.items():
            if all(rank in ranks for rank in [14, 13, 12, 11, 10]):  # A=14, K=13, Q=12, J=11, 10=10
                return True
                
        return False
        
    def play_premium_hand(self, round_state, hand_rank):
        '''
        Optimal strategy for premium hands
        
        Arguments:
        round_state: the RoundState object
        hand_rank: String describing the hand rank
        
        Returns:
        Action to take
        '''
        legal_actions = round_state.legal_actions()
        
        # Strategy depends on hand strength
        if hand_rank == 'royal_flush' or hand_rank == 'straight_flush':
            # With the nuts or near nuts, maximize value
            if RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                
                # With monster hands, we want to extract maximum value
                # If early in betting, make a small-medium raise to build pot
                if round_state.street < 2:  # Pre-flop or flop
                    # Raise 40% of max to build pot
                    raise_amount = min_raise + int((max_raise - min_raise) * 0.4)
                    return RaiseAction(raise_amount)
                else:  # Final betting round
                    # Go all-in on the last street to maximize value
                    return RaiseAction(max_raise)
            
            # If we can't raise, call
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction()
            
        elif hand_rank == 'four_of_a_kind' or hand_rank == 'full_house':
            # Very strong hands, but could be beaten
            if RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                
                if round_state.street < 2:  # Pre-flop or flop
                    # Make a medium-sized raise
                    raise_amount = min_raise + int((max_raise - min_raise) * 0.6)
                    return RaiseAction(raise_amount)
                else:  # Final betting round
                    # Strong bet but not necessarily all-in
                    raise_amount = min_raise + int((max_raise - min_raise) * 0.8)
                    return RaiseAction(raise_amount)
            
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction()
            
        # This shouldn't be reached, but just in case
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
