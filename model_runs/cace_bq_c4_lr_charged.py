import numpy as np
import os
import torch
import torch.nn as nn
import logging

import cace
from cace.representations import Cace
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

LR = True
dtype = "charged"

cutoff = 4.0
batch_size = 4
name = f"cace_bq_c4_{dtype}"
if LR:
    name += "_lr"
else:
    name += "_sr"

on_cluster = False
if 'SLURM_JOB_CPUS_PER_NODE' in os.environ.keys():
    on_cluster = True
if on_cluster:
    root = f"/global/scratch/users/king1305/data/spice-dipep-{dtype}.h5"
else:
    root = f"/home/king1305/ORBITAL_LABELING/cace_lr/data/spice-dipep-{dtype}.h5"

#Load data
torch.set_default_dtype(torch.float32)
cace.tools.setup_logger(level='INFO')
logging.info("reading data")
from cace_lr.data import SpiceData
data = SpiceData(root,cutoff,batch_size=batch_size,in_memory=on_cluster,valid_p=0.1,test_p=0)
train_loader = data.train_dataloader()
valid_loader = data.val_dataloader()

#Device
use_device = 'cuda'
device = cace.tools.init_device(use_device)
logging.info(f"device: {use_device}")

logging.info("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff, p=5)
cace_representation = Cace(
    zs=[1, 6, 7, 8],
    n_atom_basis=4,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=8
    max_l=3,
    max_nu=3,
    device=device,
    num_message_passing=1,
    type_message_passing=[“M”, “Ar”, “Bchi”],
    args_message_passing={‘Bchi’: {‘shared_channels’: False, ‘shared_l’: False}},
    timeit=False
)

cace_representation.to(device)
logging.info(f"Representation: {cace_representation}")

from cace.models import NeuralNetworkPotential
from cace.modules import Atomwise, Forces

atomwise = Atomwise(n_layers=3,
                    output_key="pred_energy",
                    n_hidden=[32,16],
                    n_out=1,
                    use_batchnorm=False,
                    add_linear_nn=True)

forces = Forces(energy_key="pred_energy",
                forces_key="pred_force")

logging.info("building CACE NNP")
cace_nnp = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[atomwise, forces]
)

if LR:
    q = cace.modules.Atomwise(
        n_layers=3,
        n_hidden=[24,12],
        n_out=1,
        per_atom_output_key='q',
        output_key = 'tot_q',
        residual=False,
        add_linear_nn=True,
        bias=False)
    
    ep = cace.modules.EwaldPotential(dl=3,
                        sigma=1.5,
                        feature_key='q',
                        output_key='ewald_potential',
                        remove_self_interaction=False,
                       aggregation_mode='sum')
    
    forces_lr = cace.modules.Forces(energy_key='ewald_potential',
                                        forces_key='ewald_forces')
    
    cace_nnp_lr = NeuralNetworkPotential(
        input_modules=None,
        representation=cace_representation,
        output_modules=[q, ep, forces_lr]
    )

    pot1 = {'pred_energy': 'pred_energy', 
            'pred_force': 'pred_force',
            'weight': 1,
           }
    
    pot2 = {'pred_energy': 'ewald_potential', 
            'pred_force': 'ewald_forces',
            'weight': 1,
           }
    
    cace_nnp = cace.models.CombinePotential([cace_nnp, cace_nnp_lr], [pot1,pot2])

cace_nnp.to(device)

logging.info(f"First train loop:")
#Losses
from cace.tasks import GetLoss
e_loss = GetLoss(
    target_name="energy",
    predict_name='pred_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10, #10 first?
)
f_loss = GetLoss(
    target_name="force",
    predict_name='pred_force',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000,
)

#Metrics
from cace.tools import Metrics
e_metric = Metrics(
            target_name="energy",
            predict_name='pred_energy',
            name='e',
            # metric_keys=["rmse"],
            per_atom=True,
        )
f_metric = Metrics(
            target_name="force",
            predict_name='pred_force',
            # metric_keys=["rmse"],
            name='f',
        )

# Example usage
logging.info("creating training task")

optimizer_args = {'lr': 1e-2, 'betas': (0.99, 0.999)}  
scheduler_args = {'mode': 'min', 'factor': 0.8, 'patience': 20}

for _ in range(5):
    task = TrainingTask(
        model=cace_nnp,
        losses=[e_loss,f_loss],
        metrics=[e_metric, f_metric],
        device=device,
        optimizer_args=optimizer_args,
        #scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=True,
        ema_start=10,
        warmup_steps=10,
    )

    logging.info("training")
    task.fit(train_loader, valid_loader, epochs=300, screen_nan=False)

task.fit(train_loader, valid_loader, epochs=700, screen_nan=False)
task.save_model('lr-model-1.pth')

e_loss = GetLoss(
    target_name="energy",
    predict_name='pred_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000,
)
task.update_loss([e_loss,f_loss])

task.fit(train_loader, valid_loader, epochs=1000, screen_nan=False)
task.save_model('lr-model-2.pth')
logging.info("Finished!")

trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
logging.info(f"Number of trainable parameters: {trainable_params}")



