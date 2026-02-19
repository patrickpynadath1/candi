from .ar import *
from .duo import *
from .distillation import *
from .mdlm import *
from .sedd_absorb import *
from .semicat import *
from .d3pm_absorb import *
from .candi import *
from .cont_diff_onehot import *
from .cont_diff_embed import *
from .cadd import *


all_algos = {
    "ar": AR,
    "mdlm": MDLM,
    "duo_base": DUO_BASE,
    "d3pm": D3PMAbsorb,
    "sedd": SEDDAbsorb,
    "duo": DUO,
    "distillation": Distillation,
    "candi": CANDI,
    "semicat": Semicat,
    "cont_diff_onehot": ContDiffOnehot,
    "cont_diff_embed": ContDiffEmbed,
    "cadd": CADD,
}