
# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

import copy 
import torch 
# import ray

## @ray.remote

##==============================
# HARDCODED PARAMS 
##==============================
MODEL_SAVE_PATH = "vidur/vidur/mcts/DNN/saved_models/"


class SharedStorage:

    """A class to store and share the latest model weights across multiple self-play workers."""

    def __init__(self, checkpoint : any = None):

        """Initializes the SharedStorage with an optional checkpoint."""
        self.current_model_checkpoint = copy.deepcopy(checkpoint)

    def save_checkpoint(self, path: str = None):

        """Saves the current model checkpoint to the specified path."""
        if path is None:
            path = MODEL_SAVE_PATH/"model.checkpoint"

        torch.save(self.current_model_checkpoint, path)

    def get_checkpoint(self):

        """Returns the latest model checkpoint."""
        return self.current_model_checkpoint

    def get_info(self, keys):

        """Returns specific information from the current model checkpoint based on provided keys."""
        if isinstance(keys, str):
            return self.current_model_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_model_checkpoint[key] for key in keys}
        else:
            raise TypeError("Keys must be a string or a list of strings.")

    def set_info(self, keys, values = None):

        """Sets specific information in the current model checkpoint based on provided key/s and values."""
        if isinstance(keys, str):
            self.current_model_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_model_checkpoint.update(keys)
        else:
            raise TypeError("Keys must be a string or a dictionary.")








































