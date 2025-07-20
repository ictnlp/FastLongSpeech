from .ctc_generator import CTC_Generator, Self_CTC_Generator

def build_ctc_decoder(config):
        return CTC_Generator(config)

def build_self_ctc_decoder(config):
        return Self_CTC_Generator(config)