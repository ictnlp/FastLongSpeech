
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

class CTC_Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.ctc_decoder == "projector_llm_embed":
            self.projector = nn.Linear(config.audio_config.d_model, config.text_config.hidden_size, bias=True)
        self.linear = nn.Linear(config.ctc_decoder_input_dim, config.ctc_decoder_num_embeddings + 1, bias=False)

    def forward(self, audio_features):
        outputs = self.linear(audio_features)
        return outputs
    
class Self_CTC_Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.ctc_decoder == "projector_llm_embed":
            self.projector = nn.Linear(config.audio_config.d_model, config.text_config.hidden_size, bias=True)
        self.linear = nn.Linear(config.ctc_decoder_input_dim, config.ctc_embed_num + 1, bias=False)

    def forward(self, audio_features):
        outputs = self.linear(audio_features)
        return outputs