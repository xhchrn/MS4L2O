import os
import numpy as np
import configargparse
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import optimizers
import random
import utils

from optimizees import OPTIMIZEE_DICT


# Argument Parsing
parser = configargparse.get_arg_parser(description='Configurations for ALISTA experiement')

parser.add('-c', '--config', is_config_file=True, help='Config file path.')

parser.add('--optimizer', type=str, metavar='STR',
           help='What optimizer to use for the current experiment.')
parser.add('--cpu', action='store_true',
           help='Force to use CPU instead of GPU even if CUDA compatible GPU '
                'devices are available.')
parser.add('--test', action='store_true', help='Run in test mode.')
parser.add('--device', type=str, default = None, help='cuda:0')

# Optimizee general options
parser.add('--optimizee-type',
           choices=['QuadraticUnconstrained', 'LASSO', 'LigisticL1'],
           help='Type of optimizees to be trained on')
parser.add('--input-dim', type=int, metavar='INT',
           help='Dimension of the input (optimization variable)')
parser.add('--output-dim', type=int, metavar='INT',
           help='Dimension of the output (labels used to calculate loss)')
parser.add('--rho', type=float, default=0.1, metavar='FLOAT',
           help='Parameter for reg. term in the objective function.')
parser.add('--sparsity', type=int, default=5, metavar='INT',
           help='Sparisty of the input variable.')
parser.add('--W-cond-factor', type=float, default=0.0, metavar='FLOAT',
           help='W: The ratio of randn and ones.')
parser.add('--x-mag', type=float, default=1.0, metavar='FLOAT',
           help='x: magnitude of nonzeros in x.')
parser.add('--W-cond-rand', action='store_true',
           help='Using random W-cond-factor in training.')
parser.add('--dist-rand', action='store_true',
           help='W-cond, x-mag, s and rho are generated randomly.')
parser.add('--save-to-mat', action='store_true',
           help='save optmizees to mat file.')
parser.add('--optimizee-dir', type=str, metavar='STR',
           help='dir of optimizees.')
parser.add('--load-mat', action='store_true',
           help='load optmizees from mat file.')
parser.add('--save-sol', action='store_true',
           help='save solutions of optimizees.')
parser.add('--load-sol', action='store_true',
           help='save solutions of optimizees.')

# Unconstrained Quadratic
parser.add('--fixed-dict', action='store_true',
           help='Use a fixed dictionary for the optimizees')

# Model parameters
parser.add('--layers', type=int, default=20, metavar='INT',
           help='Number of layers of the neural network')
parser.add('--symm', action='store_true',
           help='Use the new symmetric matrix parameterization')

parser.add('--step-size', type=float, default=None, metavar='FLOAT',
           help='Step size for the classic optimizers')

# Data parameters
parser.add('--seed', type=int, default=118, metavar='INT',
           help='Random seed for reproducibility')

# Training parameters
parser.add('--train-objective',
           type=str, default='GT', metavar='{OBJECTIVE,L2,L1,GT}',
           help='Objective used for the training')
parser.add('--save-dir', type=str, default='temp',
           help='Saving directory for saved models and logs')
parser.add('--ckpt-path', type=str, default=None, metavar='STR',
           help='Path to the checkpoint to be loaded.')
parser.add('--loss-save-path', type=str, default=None, metavar='STR',
           help='Path to save the testing losses.')
parser.add('--train-size', type=int, default=32000, metavar='N',
           help='Number of training samples')
parser.add('--val-size', type=int, default=128, metavar='N',
           help='Number of validation samples')
parser.add('--test-size', type=int, default=1024, metavar='N',
           help='Number of testing samples')
parser.add('--train-batch-size', type=int, default=256, metavar='N',
           help='Batch size for training')
parser.add('--val-batch-size', type=int, default=128, metavar='N',
           help='Batch size for validation')
parser.add('--test-batch-size', type=int, default=32, metavar='N',
           help='Batch size for testing')
parser.add('--init-lr', type=float, default=0.1, metavar='FLOAT',
           help='Initial learning rate')
parser.add('--lr-decay-layer', type=float, default=0.3, metavar='FLOAT',
           help='Decay learning rates of trained layers')
parser.add('--lr-decay-stage2', type=float, default=0.2,
           metavar='FLOAT', help='Decay rate for training stage2 in each layer')
parser.add('--lr-decay-stage3', type=float, default=0.02, metavar='FLOAT',
           help='Decay rate for training stage3 in each layer')
parser.add('--best-wait', type=int, default=5, metavar='N',
           help='Wait time for better validation performance')

parser.add('--test-length', type=int, default=20,
           help='Total length of optimization during testing')

opts, _ = parser.parse_known_args()

# Save directory
opts.save_dir = os.path.join('results', opts.save_dir)
if not os.path.isdir(opts.save_dir):
    os.makedirs(opts.save_dir)
# Logging file
logger_file = os.path.join(opts.save_dir, 'train.log')
opts.logger = utils.setup_logger(logger_file)
opts.logger('Checkpoints will be saved to directory `{}`'.format(opts.save_dir))
opts.logger('Log file for training will be saved to file `{}`'.format(logger_file))

# Use cuda if it is available
if opts.cpu:
    opts.device = 'cpu'
elif opts.device is None:
    if torch.cuda.is_available():
        opts.device = 'cuda'
    else:
        opts.device = 'cpu'
        opts.logger('WARNING: No CUDA available. Run on CPU instead.')
opts.logger('Using device: {}'.format(opts.device)) # Output the type of device used
opts.dtype  = torch.float
opts.logger('Using device: {}'.format(opts.device)) # Output the type of device used
# opts.logger('Using tau: {}'.format(opts.tau)) # Output the tau used in current exp


# Set random seed for reproducibility
torch.manual_seed(opts.seed)
random.seed(opts.seed + 7)
np.random.seed(opts.seed + 42)

if opts.fixed_dict:
    W = torch.randn(opts.output_dim, opts.input_dim).to(opts.device)
else:
    W = None

def make_train_step(optimzer, meta_optimizer):

    def train_step(optimizees, network_layer, x_gt=None):
        optimzer.train() # Set the optimizer to training mode

        for _ in range(network_layer):
            optimizees = optimzer(optimizees)
        loss = ((optimizees.X_ref - optimizees.X)**2.0).sum(dim=(1,2)).mean()

        meta_optimizer.zero_grad() # Set gradient to zero
        loss.backward()
        meta_optimizer.step() # Update the weights using the optimizer

        return loss.item()

    return train_step


optimizee_kwargs = {
    'layers': opts.layers,
    'input_dim': opts.input_dim,
    'output_dim': opts.output_dim,
    'rho': opts.rho,
    's': opts.sparsity,
    'device': opts.device,
    'W_cond_factor': opts.W_cond_factor,
    'x_mag': opts.x_mag,
}

if opts.optimizer == 'AdaLISTA':
    optimizer = optimizers.AdaLISTA(
        layers = opts.layers,
        input_dim = opts.input_dim,
        output_dim = opts.output_dim
    )
else:
    raise ValueError('Invalid optimizer name')

optimizer = optimizer.to(device=opts.device, dtype=opts.dtype)
fista = optimizers.ProximalGradientDescentMomentum()

if not opts.test:
    training_losses   = [] # initialize the array storing training loss function
    validation_losses = [] # initialize the array storing validation loss function
    
    num_train_batches = opts.train_size // opts.train_batch_size
    train_optimizee_batches = []

    # Conduct training layer-wise in increasing depth.
    for j in range(opts.layers):
        current_layer = j + 1

        epoch = 0
        batch_losses = [] # Initialize batch losses
        # Loop over stage 1,2,3
        for stage in range(1, 4):
            # Set up optimizer
            meta_optimizer = optimizer.get_meta_optimizer(
                layer = current_layer,
                stage = stage,
                init_lr = opts.init_lr,
                lr_decay_layer  = opts.lr_decay_layer,
                lr_decay_stage2 = opts.lr_decay_stage2,
                lr_decay_stage3 = opts.lr_decay_stage3,
            )
            best_val_nmse = 1e30
            best_val_epoch = epoch  # Starting each stage, the best epoch is the current epoch
            opts.logger('Training layer {} - stage {}'.format(current_layer, stage))
            # print(optimizer)

            train_step = make_train_step(optimizer, meta_optimizer)
            
            batch_order = np.random.permutation(num_train_batches)

            while True:
                for i in range(num_train_batches):
                    if len(train_optimizee_batches) < num_train_batches:
                        optimizees = OPTIMIZEE_DICT[opts.optimizee_type](
                            opts.train_batch_size, W, **optimizee_kwargs
                        )
                        fista.reset_state(optimizees, None)
                        for _ in range(5000):
                            optimizees = fista(optimizees)
                        optimizees.X_ref = optimizees.X.detach()
                        train_optimizee_batches.append(optimizees)
                        print(f'batch {i+1} added to list')
                    else:
                        optimizees = train_optimizee_batches[batch_order[i]]
                    optimizees.initialize()
                    optimizer.reset_state(optimizees, opts.step_size)
                    loss = train_step(optimizees, network_layer=current_layer)
                    batch_losses.append(loss) # Add loss to list

                training_loss = np.mean(batch_losses) # Compute the average of the batch losses
                training_losses.append(training_loss) # Append this new value to the array of losses
                epoch += 1

                # Do validation
                optimizer.eval()
                val_losses = [] # Initialize list of validation losses

                optimizees = OPTIMIZEE_DICT[opts.optimizee_type](
                    opts.val_size, W, seed=opts.seed + 77, **optimizee_kwargs)
                for l in range(current_layer):
                    optimizees = optimizer(optimizees)
                val_loss = optimizees.objective(compute_grad=False).item()

                val_losses.append(val_loss)  # Add current loss to list
                validation_loss = np.mean(val_losses) # Compute the average of the batch losses
                validation_losses.append(validation_loss) # Append this new value to the array of losses

                # output the epoch results to the terminal
                opts.logger(
                    '[%(first)d] Training loss: %(second).5e\t Validation loss: %(third)0.5e' % \
                    {"first":epoch, "second":training_loss, "third":validation_loss}
                )

                if validation_loss < best_val_nmse:
                    best_val_nmse = validation_loss
                    best_val_epoch = epoch
                if epoch - best_val_epoch > opts.best_wait or epoch > stage * 200:
                    break

        checkpoint_name = optimizer.name() + '.pt'
        save_path = os.path.join(opts.save_dir, checkpoint_name)
        torch.save(optimizer.state_dict(), save_path)
        opts.logger('Saved the optimizer to file: ' + save_path)

else:
    checkpoint_name = optimizer.name() + '.pt'
    save_path = os.path.join(opts.save_dir, checkpoint_name)
    optimizer.load_state_dict(torch.load(save_path, map_location='cpu'))
    optimizer.eval()
    
    if not opts.test_batch_size:
        opts.test_batch_size = opts.test_size
    num_test_batches = opts.test_size // opts.test_batch_size

    # testing_losses_per_layer = [0.0]
    # for current_layer in range(1, model.layers + 1):
    #     # Do testing
    #     test_losses = [] # Initialize list of testing losses

    #     optimizees = OPTIMIZEE_DICT[opts.optimizee_type](
    #         opts.val_size, W, seed=opts.seed + 777, **kwargs)
    #     solved = model(optimizees, K=current_layer)
    #     test_loss = solved.objective(compute_grad=False).item()

    #     test_losses.append(test_loss)  # Add current loss to list
    #     testing_loss = np.mean(test_losses) # Compute the average of the batch losses
    #     testing_losses_per_layer.append(testing_loss) # Append this new value to the array of losses

    test_losses = [0.0] * (opts.test_length + 1)
    if opts.save_sol:
        test_losses_batch = np.zeros((opts.test_length + 1, opts.test_batch_size))
        
    for i in range(num_test_batches):
        seed = opts.seed + 777 * (i+1)

        if opts.dist_rand:
            optimizee_kwargs['W_cond_factor'] = random.random()
            optimizee_kwargs['rho'] = 10 ** ( random.random() * (-2) ) # 1e-2 ~1e0
            optimizee_kwargs['x_mag'] = 10 ** ( random.random() * (-2) + 1 ) # 1e-1 ~1e1
            optimizee_kwargs['s'] = int( (random.random()*0.15 + 0.1) * optimizee_kwargs['input_dim'] ) # input-dim * (0.1 ~ 0.25)
        elif opts.W_cond_rand:
            optimizee_kwargs['W_cond_factor'] = random.random()
        optimizees = OPTIMIZEE_DICT[opts.optimizee_type](
            opts.test_batch_size, W, seed=seed, **optimizee_kwargs
        )

        if opts.load_mat:
            optimizees.load_from_file(opts.optimizee_dir + '/' + str(i) + '.mat')
            print("Loaded:", opts.optimizee_dir + '/' + str(i) + '.mat')
            
        if opts.load_sol:
            optimizees.load_sol(opts.optimizee_dir + '/sol_' + str(i) + '.mat')
            print("Sol Loaded.", i)
        
        if opts.save_to_mat:
            if not os.path.exists(opts.optimizee_dir):
                os.mkdir(opts.optimizee_dir)
            optimizees.save_to_file(opts.optimizee_dir + '/' + str(i) + '.mat')
        
        optimizer.reset_state(optimizees, opts.step_size)
        if not opts.load_sol:
            test_losses[0] += optimizees.objective().detach().cpu().item()
        else:
            test_losses[0] += optimizees.objective_shift().detach().cpu().item()
            
        if opts.save_sol:
            test_losses_batch[0] = optimizees.objective_batch().cpu().numpy()
        
        for j in range(opts.test_length):
            # Fixed data samples for test
            optimizees = optimizer(optimizees)
            if not opts.load_sol:
                loss = optimizees.objective()
            else:
                loss = optimizees.objective_shift()
            
            test_losses[j+1] += loss.detach().cpu().item()
            if opts.save_sol:
                test_losses_batch[j+1] = optimizees.objective_batch().cpu().numpy()
        
        if opts.save_sol:
            obj_star = np.min(test_losses_batch, axis = 0)
            optimizees.save_sol(obj_star, opts.optimizee_dir + '/sol_' + str(i) + '.mat')
            print("Obj star saved.", i)
            print(obj_star.shape, obj_star)

        
    test_losses = [loss / num_test_batches for loss in test_losses]

    # output the epoch results to the terminal
    opts.logger('Testing losses:')
    for ii,t_loss in enumerate(test_losses):
        opts.logger('{}, {}'.format(ii, t_loss))
    if not opts.loss_save_path:
        opts.loss_save_path = os.path.join(opts.save_dir, 'test_losses.txt')
    else:
        opts.loss_save_path = os.path.join(opts.save_dir, opts.loss_save_path)
    opts.logger(f'testing losses saved to {opts.loss_save_path}')
    np.savetxt(opts.loss_save_path, np.array(test_losses))

    # output the epoch results to the terminal
    # opts.logger('Testing losses:')
    # for t_loss in testing_losses_per_layer:
    #     opts.logger('{}'.format(t_loss))
    # loss_save_path = os.path.join(opts.save_dir, 'test_losses.txt')
    # print(f'testing losses saved to {loss_save_path}')
    # np.savetxt(loss_save_path, np.array(testing_losses_per_layer))

