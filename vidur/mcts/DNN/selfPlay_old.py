# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)


"""
 THis file will be responsible for self play logic using DNN model , but for now we will use MCTS from vidur/vidur/mcts/mcts.py, but in future we will implement self play logic here IA.
"""

import math 
import time 
import numpy 
import torch 
import models

# import ray  ## for distributed training in future
## we will use mcts from vidur/vidur/mcts/mcts.py for now 
import vidur.vidur.mcts.mcts as mcts


## NOTE : IN our MCTS 1 Node is essentially state on Vidur Simulator + player to play next move on that state !
## So Node = (state, player) pair
## Action is the action taken by the player on that state to reach next state.

##----------------------------------------------
# HARDCODED PARAMETERS FOR SELF PLAY
##----------------------------------------------
TRAINING_STEPS = 1000
TEMPERATURE_VALUE = 1.0
TEMPERATURE_THRESHOLD = 30
SIMULATION_ITERATIONS = 5000
INITIAL_HISTORY_STEPS = 5 # number of initial steps at random by both Request Generator and Scheduler before using the DNN model to guide the self-play in order to create a new state.
DEVICE_MODE = 'gpu' if torch.cuda.is_available() else 'cpu'
MAX_GAME_LENGTH = 50  # maximum number of steps in a single game to avoid infinite loops.

#ray.remote
class SelfPlay:

    """
    A class to handle self-play logic using a DNN model and MCTS.
    This will run in a dedicated thread to play games and save them in the replay buffer.
    """

    def __init__(self, initial_checkpoint, game, shared_storage, config, seed=42):

        """Initializes the SelfPlay with the game environment, shared storage, and configuration."""
        self.game = game

        # Fix random generator seed 
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialising the Network 
        self.model = models.AlphaZeroModel()
        self.model.set_weights(initial_checkpoint['weights'])
        self.model.to(torch.device(DEVICE_MODE))
        self.model.eval() # means model will not be trained here aka model is in inference mode

        # TODO : Initialise the VidurMcts + environment here or PASS it on the constructor of the SelfPlay class from outside.



    def continous_self_play(self, shared_storage, replay_buffer):

        """Continuously plays games and stores them in the replay buffer."""
        # --> need ray here later for distributed training
        while shared_storage.get_info('training_steps') < TRAINING_STEPS:

            self.model.set_weights(shared_storage.get_info('weights'))

            ## Training mode only :
            game_history = self.play_game(TEMPERATURE_VALUE, TEMPERATURE_THRESHOLD,"self", 0)
            replay_buffer.save_game_history(game_history, shared_storage)
        
        # TODO: Add logic for the RAY/ray distributed training here later to work with multiple self-play workers + training workers.


    def play_game(self, temperature, temperature_threshold, player, step):

        """Plays a single game using MCTS guided by the DNN model at each move."""
        game_history = GameHistory()

        # We will get the initial state from the game environment + player who will play next move on that state from the mcts.py
        # initial_state , player = mcts.generateInititalState(INITIAL_HISTORY_STEPS) # TODO: implement this function in mcts.py (core functionality is in the search function of mcts.py)
        state = self.game.get_initial_state()
        game_history.state_history.append(state)
        game_history.to_play_history.append(player)
        done = False
        new_root_state = initial_state
        with torch.no_grad():
            while not done and len(game_history.state_history) <= MAX_GAME_LENGTH:
                # TODO : Adjust the logic of search function to utilise new params and return the state as needed. Atm state will have sim_snapshot, state_value, actions for that state etc 
                # - need to make MCTS Search create state via the state passed to it from here 
                new_player , action_space = mcts.search(self.model, new_root_state, SIMULATION_ITERATIONS) # new root state also contains essentially the snapshot of the vidur simulator provided by the mcts after search is done.
                action , reward = self.select_action(new_root_state, temperature, temperature_threshold, len(game_history.state_history))
                new_root_state = mcts.take_action(action)  # TODO : Implement take_action function in mcts.py to return the new state after taking action on current state.
                # TODO : This line will need change as new_root_state object will be properly defined in mcts.py
                game_history.store_search_statistics(new_root_state, action_space)

                # Appending the relevant details to game history
                game_history.action_history.append(action)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(new_player)



            return reward


    def select_action(self, node, temperature , temperature_threshold, game_history_length):
        
        """Selects an action based on the visit counts and temperature parameter."""
        
        if not temperature_threshold or game_history_length < temperature_threshold:
            temperature = 0
        visit_counts = numpy.array([child.visit_count for child in node.children.values()])
        actions = list(node.children.keys())

        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float('inf'):
            action = numpy.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = numpy.random.choice(actions, p=visit_count_distribution)




class GameHistory:
    
    """A class to store the history of a single game played.
    
    FORMAT :
    ---------------------------------------------------
    INDEX        | 1    | 2     | 3     |
    STATE        |state1|state2 |state3 | ...
    MCTS_VALUE   |VALUE1| VALUE2 | VALUE3 | ... aka root_values below
    VIDUR_REWARD |REWARD1|REWARD2|REWARD3| ...
    MCTS_POLICY  |POLICY1|POLICY2|POLICY3| ... aka child_visits below
    PLAYER       | P1   |  P2   |  P1   | ... aka to_play_history below
    ---------------------------------------------------
    """
    

    def __init__(self):

        """Initializes an empty game history."""
        self.state_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []

    def store_search_statistics(self, root_state, action_space):

        # Turn visit count from root into a policy
        if root_state is not None :
            sum_visits = sum(child.visit_count for child in root_state.children.values())
            self.child_visits.append(
                [
                    root_state.children[action].visit_count / sum_visits if action in root_state.children else 0
                    for action in range(action_space)
                ]
            )

            self.root_values.append(root_state.value())
        else:
            self.root_values.append(None)


































