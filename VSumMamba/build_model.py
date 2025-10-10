#from . import models
from VSumMamba.models.VSumMamba import VSumMamba_A

def build_model(args):
    #edited model
    model = VSumMamba_A(dataset=args.dataset)
    return model
