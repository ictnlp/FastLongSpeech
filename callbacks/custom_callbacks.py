from transformers.trainer_callback import TrainerCallback
import pdb

class CustomCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model
    
    def on_step_begin(self, args, state, control, **kwargs):
        
        self.model.base_model.model.current_step = state.global_step