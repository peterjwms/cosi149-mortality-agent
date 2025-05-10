import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import model as model_module

class Inspector:

    @staticmethod
    def load_ckpt(ckpt_path, device=None):
        if not os.path.exists(ckpt_path):
            raise Exception("Checkpoint " + ckpt_path + " does not exist.")
        # Load checkpoint.
        checkpt = torch.load(ckpt_path)
        ckpt_args = checkpt['args']
        state_dict = checkpt['state_dicts']


        model = getattr(getattr(model_module, ckpt_args.model), ckpt_args.model)(ckpt_args)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(state_dict) 
        # 3. load the new state dict
        model.load_state_dict(state_dict)

        if not device:
            device = ckpt_args.device 
        model.to(device)
        return model, ckpt_args