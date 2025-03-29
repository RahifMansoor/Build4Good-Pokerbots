'''
Build4Good B4G Hold'em Pokerbot
'''
import random
import eval7 # Make sure eval7 is installed (pip install eval7)

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class Player(Bot):
    '''
    A pokerbot for B4G Hold'em.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.
        '''
        self.log_file = open("bot_log.txt", "w") # Optional: For debugging locally
        self.log_file.write("Bot initialized\n")
        self.log_file.flush()
        pass

    def __del__(self):
        '''
        Called when the bot is shutting down.
        '''
        if self.log_file:
            self.log_file.close()

    def _log(self, message):
        '''Helper function for logging'''
        if self.log_file:
            self.log_file.write(f"[Round {self.round_num}] {message}\n")
            self.log_file.flush()

    def _evaluate_preflop_strength(self, cards):
        """
        Simple heuristic for 3-card hand strength pre-flop.
        Returns a strength value (higher is better).
        """
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        is_suited = len(set(suits)) == 1
        is_pair = ranks[0] == ranks[1] or ranks[1] == ranks[2]
        is_trips = ranks[0] == ranks[1] == ranks[2]
        is_connected = (ranks[0] == ranks[1] + 1) or (ranks[1] == ranks[2] + 1) or (ranks[0] == ranks[2] + 2) # Simple check
        is_straight_draw = (ranks[0] == ranks[1] + 1 and ranks[1] == ranks[2] + 1) or \
                           (ranks[0] == 12 and ranks[1] == 1 and ranks[2] == 0) # A23
        
        strength = 0
        
        # Base strength from high cards (0-12 for 2-A)
        strength += ranks[0] * 1.5 + ranks[1] * 1.0 + ranks[2] * 0.5

        if is_trips:
            strength += 100 + ranks[0] * 3 # Trips are very strong
        elif is_pair:
            pair_rank = ranks[1] # The middle card is always part of the pair if one exists
            strength += 50 + pair_rank * 2 # Pairs are good

        if is_suited:
            strength += 20
            if is_straight_draw:
                 strength += 30 # Suited connectors/gappers are strong
            elif is_connected:
                 strength += 15

        elif is_straight_draw:
            strength += 25 # Unsuited connectors/gappers
        elif is_connected:
            strength += 10

        # Premium hands boost
        if ranks[0] >= 10: # Ace, King, Queen, Jack high
             strength += 5
        if ranks[0] >= 10 and ranks[1] >= 9: # Two high cards
             strength += 10

        # Normalize slightly (rough estimate)
        return strength / 200 # Aim for a value roughly between 0 and 1, can exceed 1 for monsters


    def _get_hand_strength(self, my_cards, board_cards):
        """
        Evaluates hand strength using eval7.
        Returns a score where higher is better. Handles different street lengths.
        """
        if not board_cards: # Pre-flop
            hole_cards = [eval7.Card(c) for c in my_cards]
            return self._evaluate_preflop_strength(hole_cards)

        all_cards_str = my_cards + board_cards
        all_cards = [eval7.Card(c) for c in all_cards_str]

        # eval7.evaluate needs 5, 6, or 7 cards to find the best 5-card hand.
        num_cards = len(all_cards)

        if num_cards < 5:
             # This shouldn't happen on flop (5) or turn (7) in B4G Hold'em
             # If it does, we need a heuristic or return 0
             self._log(f"Warning: Trying to evaluate less than 5 cards: {num_cards}")
             return 0 # Or calculate potential based on hole + board? Needs more logic. For now, return 0.
        elif num_cards > 7:
             # This also shouldn't happen.
             self._log(f"Warning: Trying to evaluate more than 7 cards: {num_cards}")
             # Trim to 7? For now, use the first 7.
             score = eval7.evaluate(all_cards[:7])
        else:
            score = eval7.evaluate(all_cards)

        # eval7 returns a numerical score. Higher is better.
        # We can normalize this score to a 0-1 range for easier thresholding.
        # Max possible score (Royal Flush) is around 7462 according to some sources,
        # but lets use a slightly higher number for normalization safety.
        # Worst score (7 high) is 0.
        # Let's use a practical scale based on observed values or poker knowledge.
        # Straight Flush: ~7462+
        # Quads: ~7452 - 7461
        # Full House: ~7437 - 7451
        # Flush: ~5863 - 7436
        # Straight: ~4904 - 5862
        # Three of a Kind: ~3426 - 4903
        # Two Pair: ~1610 - 3425
        # Pair: ~11 - 1609
        # High Card: 0 - 10
        # Using a simple linear scale up to Royal Flush might not be the best representation
        # of practical strength differences. Let's try a rough normalization based on these tiers.
        # A very rough normalization, needs refinement:
        max_theoretical_score = 7500 # Adjust as needed
        normalized_strength = score / max_theoretical_score
        
        # We could also return the rank class (e.g., eval7.HANDTYPE_NAMES[eval7.handtype(score)])
        # rank_class = eval7.handtype(score) # 0=HIGH_CARD, 1=PAIR, ..., 8=STRAIGHT_FLUSH
        
        return normalized_strength # Return normalized score (0 to ~1)

    def handle_new_round(self, game_state: GameState, round_state: RoundState, active: int):
        '''
        Called when a new round starts.
        '''
        self.round_num = game_state.round_num
        my_bankroll = game_state.bankroll
        game_clock = game_state.game_clock
        my_cards = round_state.hands[active]
        self.is_big_blind = bool(active)
        self.is_small_blind = not self.is_big_blind

        self._log(f"--- Starting Round {self.round_num} ---")
        self._log(f"My cards: {my_cards}, Clock: {game_clock:.2f}s, Bankroll: {my_bankroll}")
        self._log(f"Position: {'Big Blind' if self.is_big_blind else 'Small Blind'}")

    def handle_round_over(self, game_state: GameState, terminal_state: TerminalState, active: int):
        '''
        Called when a round ends.
        '''
        my_delta = terminal_state.deltas[active]
        previous_state = terminal_state.previous_state
        street = previous_state.street
        my_cards = previous_state.hands[active]
        opp_cards = previous_state.hands[1-active]

        self._log(f"--- Round {self.round_num} Over ---")
        self._log(f"My cards: {my_cards}")
        if opp_cards:
            self._log(f"Opponent cards: {opp_cards}")
        else:
             self._log(f"Opponent cards not revealed (Folded)")
        self._log(f"Round ended on street {street}")
        self._log(f"My bankroll change: {my_delta}")
        self._log(f"Final Bankroll: {game_state.bankroll}")
        self._log(f"Remaining Clock: {game_state.game_clock:.2f}s")


    def get_action(self, game_state: GameState, round_state: RoundState, active: int):
        '''
        This is the core function where the bot decides what action to take.
        '''
        legal_actions = round_state.legal_actions()
        street = round_state.street # 0=preflop, 2=flop, 4=turn (final)
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:street] # Board cards are revealed based on street

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]
        pot_size = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)

        continue_cost = opp_pip - my_pip
        
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            min_cost = min_raise - my_pip
            max_cost = max_raise - my_pip
            can_raise = True
        else:
            min_raise, max_raise = (0,0) # Set defaults if cannot raise
            min_cost, max_cost = (0,0)
            can_raise = False

        # --- Calculate Hand Strength ---
        strength = self._get_hand_strength(my_cards, board_cards)
        self._log(f"Street: {street}, My Hand: {my_cards}, Board: {board_cards}")
        self._log(f"My Pip: {my_pip}, Opp Pip: {opp_pip}, My Stack: {my_stack}, Opp Stack: {opp_stack}")
        self._log(f"Pot Size: {pot_size}, Continue Cost: {continue_cost}")
        self._log(f"Calculated Strength: {strength:.4f}")
        self._log(f"Legal Actions: {[a.__name__ for a in legal_actions]}")
        if can_raise:
             self._log(f"Raise Bounds: [{min_raise}, {max_raise}], Cost: [{min_cost}, {max_cost}]")


        # --- Decision Logic ---

        # Basic strategy thresholds (adjust these based on testing!)
        # Strength is roughly normalized 0-1.
        fold_threshold_preflop = 0.25
        call_threshold_preflop = 0.45
        raise_threshold_preflop = 0.70

        fold_threshold_flop = 0.20 # Fold less post-flop if pot invested
        call_threshold_flop = 0.40 # Value of Pair or better? Or good draw?
        raise_threshold_flop = 0.65 # Two pair or better?

        fold_threshold_turn = 0.20 # Final street, fold weak hands to bets
        call_threshold_turn = 0.45 # Marginal made hands
        raise_threshold_turn = 0.75 # Strong made hands (e.g., Two Pair+, sometimes strong Pair)


        # Determine current thresholds based on street
        if street == 0: # Pre-flop
            fold_threshold = fold_threshold_preflop
            call_threshold = call_threshold_preflop
            raise_threshold = raise_threshold_preflop
        elif street == 2: # Flop
            fold_threshold = fold_threshold_flop
            call_threshold = call_threshold_flop
            raise_threshold = raise_threshold_flop
        elif street == 4: # Turn (Final street)
            fold_threshold = fold_threshold_turn
            call_threshold = call_threshold_turn
            raise_threshold = raise_threshold_turn
        else: # Should not happen
            self._log("Error: Unknown street number encountered!")
            fold_threshold = 1.0 # Default to folding if something is wrong
            call_threshold = 1.1
            raise_threshold = 1.1

        # --- Action Selection ---

        # Raise Logic
        if can_raise and strength >= raise_threshold:
            # Determine raise amount
            # Simple: raise somewhere between min and pot size (or max possible)
            # Pot size raise: current pot + amount needed to call
            pot_raise_amount = pot_size + continue_cost # Amount opponent needs to call if we make pot bet *after* calling
            
            # Calculate the total bet size for a pot-sized raise
            # Total amount = current pot + 2*opponent's last bet/raise (simplified)
            # A simpler approximation: raise *by* pot size
            raise_by_pot = pot_size
            target_total_bet = my_pip + continue_cost + raise_by_pot # Amount to call + pot size
            
            # Ensure target is within bounds and affordable
            raise_amount = min(max_raise, target_total_bet) # Don't exceed max raise
            raise_amount = max(min_raise, raise_amount) # Don't go below min raise
            
            # Ensure we don't bet more than our stack allows (max_raise should handle this, but double check)
            actual_raise_cost = raise_amount - my_pip
            if actual_raise_cost > my_stack:
                 raise_amount = my_pip + my_stack # All-in

            # Sanity check amount is still valid
            if min_raise <= raise_amount <= max_raise:
                 self._log(f"ACTION: Raise to {raise_amount} (Strength: {strength:.4f} >= {raise_threshold:.4f})")
                 return RaiseAction(raise_amount)
            else: # Fallback to min raise if calculation is weird or max raise if strong
                 self._log(f"ACTION: Raise MIN {min_raise} (Strength: {strength:.4f} >= {raise_threshold:.4f}, calc issue)")
                 # Or maybe raise max if very strong?
                 if strength > raise_threshold + 0.1:
                     self._log(f"ACTION: Raise MAX {max_raise} (Very Strong: {strength:.4f})")
                     return RaiseAction(max_raise)
                 return RaiseAction(min_raise)


        # Call / Check Logic
        if CheckAction in legal_actions: # Opponent has not bet, or BB option preflop
             if strength >= call_threshold: # Check if decent hand, maybe bet later
                 self._log(f"ACTION: Check (Strength: {strength:.4f} >= {call_threshold:.4f})")
                 return CheckAction()
             else: # Check if weak hand, hope to see next card free
                 self._log(f"ACTION: Check (Strength: {strength:.4f} < {call_threshold:.4f})")
                 return CheckAction()

        if CallAction in legal_actions: # Opponent has bet
             if strength >= call_threshold:
                 # Potentially add pot odds check here later for drawing hands
                 self._log(f"ACTION: Call (Strength: {strength:.4f} >= {call_threshold:.4f}, Cost: {continue_cost})")
                 return CallAction()
             # If strength is below call but above fold, consider calling small bets?
             # Pot odds = amount to call / (pot size + amount to call)
             # Required equity = pot odds
             # if strength > fold_threshold and (continue_cost / (pot_size + continue_cost)) < strength: # Very rough pot odds check
             #    self._log(f"ACTION: Call based on pot odds (Strength: {strength:.4f}, Cost: {continue_cost})")
             #    return CallAction()


        # Fold Logic
        if FoldAction in legal_actions:
            # Fold if strength is below threshold and we can't check
            if strength < fold_threshold or continue_cost > my_stack: # Also fold if we can't afford to call
                self._log(f"ACTION: Fold (Strength: {strength:.4f} < {fold_threshold:.4f} or cannot afford call)")
                return FoldAction()
            else:
                # If we didn't meet call/raise threshold but are above fold threshold,
                # and CallAction is available, we should Call (this case handles the gap)
                if CallAction in legal_actions:
                    self._log(f"ACTION: Call (Fallback - Strength {strength:.4f} between fold/call thresholds)")
                    return CallAction()
                else:
                    # This state should ideally not be reached if logic is correct
                    # (e.g., facing an all-in we can't call/raise over, but don't want to fold)
                    # But if it does, Fold is the safest default if Check/Call aren't options.
                     self._log(f"ACTION: Fold (Fallback - Unable to Check/Call/Raise)")
                     return FoldAction()

        # Final fallback if no other action chosen (should not happen with proper logic)
        self._log("ACTION: Fold (ERROR - Default Fallback)")
        return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
# ----------------------------------------