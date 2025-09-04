from diuzhev_scratch_llm.trans_parts import HeadAttention
import torch

att = HeadAttention(12, 8, 10)
att.forward(torch.rand((1,6,12)))