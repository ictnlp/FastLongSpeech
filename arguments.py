import transformers

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen-Audio")
    version: Optional[str] = field(default="qwen2_audio")
    freeze_backbone: bool = field(default=False)
    ctc_decoder_tokenizer = field(default="path", metadata={"help": "The path the sentencepiece spm.model."})
    ctc_decoder: Optional[str] = field(default="llm_embed", metadata={"help": "You can use llm_embed, which denotes only appending the ctc_decocer, or projector_llm_embed, which denotes adding the projector and llm_embed."})
    ctc_embed_num: Optional[int] = field(default=0, metadata={"help": "You can use 0, which denotes use the vocabulary of LLMs, otherwise, we use the vocabulary of ours."})

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    validate_path: str = field(default=None,
                           metadata={"help": "Path to the validation data."})
    ctc_training: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    freeze_speechLLMs: bool = field(default=False)
    model_max_length: int = field(
        default=8192,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    ctc_decoder_lr: Optional[float] = None
    tune_ctc_decoder: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)