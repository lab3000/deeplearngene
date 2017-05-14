from sacred import Experiment
ex = Experiment('DLGn1e1p1')

from clades import *


@ex.config
def my_config():
     # deeplearngene
     # Genetically Programmed Architectures
     # for Directed Deep Learning Experimentation

    #----General properties----
    environment = 'DLGn1e1p1'
    clade = 'GAFC1'
    population_size = 50
    # plan to later add an automated evolution loop in main()
    interactive = True

    #----Callback properties----
    max_train_time = 120  # in seconds

    # preprocessing (for load_data; not yet harnessed for diversification)
    # max_words = [10000, 20000, 30000, 40000, 50000, 60000, 70000]

    # FC layer-specific properties for spawn function)
    LR_bounds = {'bins': [0, 2], 'bounds': [[1e-3, 1e-2], [1e-2, 1e-1], [1e-1, 0.5]],
                 'probs': {'bins': 'uniform', 'bounds': 'uniform'}}
    nb_layers = {'type': 'range', 'bounds': [1, 12]}
    #^^^ plan to extend alternate type: 'list'
    # possible number of units in hidden layer
    nb_units = {'type': 'range', 'bounds': [2, 512]}
    #^^^ plan to extend alternate type: 'list'
    activations = ['softmax', 'elu', 'softplus', 'softsign', 'relu', 'tanh',
                   'sigmoid', 'hard_sigmoid', 'linear']
    final_activation = 'softmax'

    #----model type, compile, and fit properties----
    model = 'Sequential'

    # possible losses (could add binary_crossentropy for alternate
    # systems)
    losses = ['categorical_crossentropy']
    optimizers = ['sgd', 'RMSProp', 'Adagrad',
                  'Adadelta', 'Adam', 'Adamax', 'Nadam']
    batch_size = [8, 16, 32, 64, 128, 256, 512]
    epochs = range(2, 20)

    # regularizers:
    # reg_model_chance=chance regularization is used at all in the model
    # reg_choices = a function executed with eval() to select reg type
    regularizers = {'reg_model_chance': 0,
                    'reg_choices': '''\
    np.random.choice(['l1','l2','dropout'],1,p=[0.2,0.3,0.5])[0]'''}

    #----select_parents properties----
    # fractions of the population that will be selected into theese
    # tiers:
    top_tier = 0.2
    random_tier = 0.1

    #----breeding properties----
    mutate_prob = 0.2
    max_mutations = 2


# the sacred decorator provides access to the _config file
@ex.automain
def main(_config):
    """Run a sacred experiment

    Parameters
    ----------
    _config : special dict populated by sacred with the local variables computed
    in my_config() which can be overridden from the command line or with
    ex.run(config_updates=<dict containing config values>)

    This function will be run if this script is run either from the command line
    with

    $ python train.py

    or from within python by

    >>> from train import ex
    >>> ex.run()
    """

    if _config['interactive'] == True:
        return
