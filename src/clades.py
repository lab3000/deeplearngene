# define GAFC1 architectures for experimentation in this file

from clade import Clade
from keras.regularizers import l2
from keras.layers import Dropout
import numpy as np
import pandas as pd
import random
import pickle
import os
from copy import deepcopy
from datetime import datetime


class GAFC1(Clade):
    """First Set of Genetic Algorithm Operations for Fully
    Connected Architectures"""

    def spawn(self):
        '''builds up individual "genes" with hyperparameters specifying Fully
        connected DL architectures into rows of a dataframe, and saves the
        dataframe, both as an object property and to a pickeled df file'''

        self.current_generation = 0

        for individual in range(self.C['population_size']):
            # gene parameters will be stored in this dictionary
            gene = {}

            # pick the learning rate
            bin_ = random.randint(self.C['LR_bounds']['bins'][0],
                                  self.C['LR_bounds']['bins'][1])

            LR = random.uniform(self.C['LR_bounds']['bounds'][bin_][0],
                                self.C['LR_bounds']['bounds'][bin_][1])
            gene['LR'] = LR

            # pick the number of layers
            nb_layers = random.randint(self.C['nb_layers']['bounds'][0],
                                       self.C['nb_layers']['bounds'][1])
            gene['nb_layers'] = nb_layers

            # get the layer activations and layer units
            layer_activations = []
            layer_units = []
            for layer in range(nb_layers):
                layer_activations.append(
                    np.random.choice(self.C['activations']))
                layer_units.append(
                    random.randint(self.C['nb_units']['bounds'][0],
                                   self.C['nb_units']['bounds'][1]))
            gene['activations'] = [layer_activations]
            gene['layer_units'] = [layer_units]

            # choose if regularization will be used,
            # and pick regularizers if yes. ***Initially not using this,
            # until have evol working in the simpler case of no reg
            p_yes = self.C['regularizers']['reg_model_chance']
            choose_regularization = np.random.choice(['yes', 'no'], 1,
                                                     p=[p_yes, 1 - p_yes])[0]
            if choose_regularization == 'yes':
                num_reg_layers = random.randint(1, nb_layers)
                layers = [i for i in range(nb_layers)]
                reg_layers = np.random.choice(layers, num_reg_layers,
                                              replace=False)
                regularlizations = {}
                for reg_layer in reg_layers:
                    regularlizations[str(reg_layer)] = eval(
                        self.C['reg_choices'])

            # pick the optimizer
            gene['optimizer'] = np.random.choice(self.C['optimizers'])

            # pick the loss-->current default is just
            # categorical_crossentropy
            gene['loss'] = np.random.choice(self.C['losses'])

            # pick the batch size
            gene['batch_size'] = np.random.choice(self.C['batch_size'])

            # pick the epochs
            gene['epochs'] = np.random.choice(self.C['epochs'])

            # add the gene name and model name to the gene entry
            gene_name = self.C['environment'] + '+Gen'\
                + str(self.current_generation) + \
                '+gene' + str(individual)
            gene['gene_name'] = gene_name

            gene['model_name'] = gene['gene_name'] + '+model.h5'

            # add the gene to a dataframe containing all genes, one per
            # row
            if individual == 0:
                generation_df = pd.DataFrame.from_dict(gene)
            else:
                temp_df = pd.DataFrame.from_dict(gene)
                generation_df = pd.concat(
                    [generation_df, temp_df], axis=0)

        # save (pickle) the dataframe containing the gene information
        pickle_name = self.C['environment'] + '+Gen' +\
            str(self.current_generation) + '+genotypes.p'
        genotypeDF_p = self.experiment_folder + '/' + pickle_name
        generation_df.to_pickle(genotypeDF_p)

        # also save genotype dataframe within current object instance
        self.genotypes = generation_df

    def breed(self, nextgen_pop_size=None):
        '''Make a new generation of genes based on the parent genes:
        1) select two parent genes
        2) recombine the parent genes to make a child genes
        3) possibly mutate the child gene, with probabiltiy mutate_prob'''

        if nextgen_pop_size == None:
            nextgen_pop_size = self.C['population_size']

        self.current_generation = self.current_generation + 1

        for individual in range(nextgen_pop_size):
            # select two parents at random from the parent genes
            parents = deepcopy(
                self.parent_genes.ix[:, ['batch_size', 'epochs', 'LR',
                                         'optimizer', 'nb_layers', 'activations',
                                         'layer_units']].sample(n=2))
            parent_A = deepcopy(pd.DataFrame(parents.iloc[0]).T)
            parent_B = deepcopy(pd.DataFrame(parents.iloc[1]).T)

            # print('parent_A = ', parent_A)
            # print('parent_B = ', parent_B)

            # gene recombination
            child = {}
            trait_source = {}
            for i in range(parents.shape[1]):
                # print('i = ', i)
                parent = np.random.choice(['parent_A', 'parent_B'])
                inheritance = eval(
                    parent + '[parents.columns[i]].iloc[0]')

                # print('inheritance = ', inheritance)
                child[parents.columns[i]] = [inheritance]
                trait_source[parents.columns[i]] = parent
                # print('child = ', child)
                # print('trait_source = ', trait_source)

            # gene editing
            if trait_source['nb_layers'] != trait_source['activations']:
                if child['nb_layers'][0] > len(child['activations'][0]):
                    # expand activation functions of the child
                    for i in range(child['nb_layers'][0]
                                   - len(child['activations'][0])):
                        child['activations'][0].append(
                            np.random.choice(
                                eval(trait_source['activations'] +
                                     '''['activations'].iloc[0]''')))

                if child['nb_layers'][0] < len(child['activations'][0]):
                    # delete activation functions of the child
                    for i in range(len(child['activations'][0])
                                   - child['nb_layers'][0]):
                        del_index = random.randint(0,
                                                   len(child['activations'][0]) - 1)
                        del child['activations'][0][del_index]

            # # adjust layer_units to match nb_layers
            if trait_source['nb_layers'] != trait_source['layer_units']:
                if child['nb_layers'][0] > len(child['layer_units'][0]):
                    # expand layer_units of the child
                    for i in range(child['nb_layers'][0]
                                   - len(child['layer_units'][0])):
                        child['layer_units'][0].append(
                            random.randint(self.C['nb_units']['bounds'][0],
                                           self.C['nb_units']['bounds'][0]))

                if child['nb_layers'][0] < len(child['layer_units'][0]):
                    # delete activation functions of the child
                    for i in range(len(child['layer_units'][0]) - child['nb_layers'][0]):
                        del_index = random.randint(
                            0, len(child['layer_units'][0]) - 1)
                        del child['layer_units'][0][del_index]

            # mutate with probability
            if random.random() < self.C['mutate_prob']:
                # pick a limited number of mutations (default: 1 or 2)
                num_mutations = random.randint(
                    1, self.C['max_mutations'])

                for m in range(num_mutations):
                    mut_posn = np.random.choice(parents.columns)

                    if mut_posn == 'batch_size':
                        child['batch_size'] = np.random.choice(
                            self.C['batch_size'])
                    if mut_posn == 'epochs':
                        child['epochs'] = np.random.choice(
                            self.C['epochs'])
                    if mut_posn == 'LR':
                        bin_ = random.randint(self.C['LR_bounds']['bins'][0],
                                              self.C['LR_bounds']['bins'][1])
                        child['LR'] = random.uniform(
                            self.C['LR_bounds'][
                                'bounds'][bin_][0],
                            self.C['LR_bounds']['bounds'][bin_][1])
                    if mut_posn == 'optimizer':
                        child['optimizer'] = np.random.choice(
                            self.C['optimizers'])
                    if mut_posn == 'nb_layers':
                        # implement this later
                        # must edit layer_units and activations list
                        # if this is chose
                        pass
                        # child['nb_layers'][0] = random.randint(
                        # self.C['nb_layers'][
                        #    'bounds'][0],
                        # self.C['nb_layers']['bounds'][1])
                    if mut_posn == 'activations':
                        # pick layer activation function to mutate
                        mut_index = random.randint(
                            0, len(child['activations'][0]) - 1)
                        child['activations'][0][mut_index] = np.random.choice(
                            self.C['activations'])
                    if mut_posn == 'layer_units':
                        # pick layer to mutate
                        mut_index = random.randint(
                            0, len(child['layer_units'][0]) - 1)
                        child['layer_units'][0][mut_index] = random.randint(
                            self.C['nb_units'][
                                'bounds'][0],
                            self.C['nb_units']['bounds'][1])

            # add the gene name and model name to the child gene entry
            gene_name = self.C['environment'] + '+Gen'\
                + str(self.current_generation) + \
                '+gene' + str(individual)
            child['gene_name'] = gene_name

            child['model_name'] = child['gene_name'] + '+model.h5'

            # add the gene to a dataframe containing all genes, one per
            # row
            if individual == 0:
                generation_df = pd.DataFrame.from_dict(child)
            else:
                temp_df = pd.DataFrame.from_dict(child)
                generation_df = pd.concat(
                    [generation_df, temp_df], axis=0)

        # also save genotype dataframe within current object
        # instance
        generation_df = generation_df.reset_index().iloc[:, 1:]
        self.genotypes = generation_df

        # save (pickle) the dataframe containing the gene information
        pickle_name = self.C['environment'] + '+Gen' +\
            str(self.current_generation) + '+genotypes.p'
        genotypeDF_p = self.experiment_folder + '/' + pickle_name
        generation_df.to_pickle(genotypeDF_p)

    def seed_models(self, new_genepool=None):
        '''
        uses parameters saved in genotypes df to define, compile, and save
        the encoded models; each model is saved to its own .h5 file

        new_genepool (if not None, it should be a string of a pickled df
        path and filename) indicates whether self genotypes (a pandas df) should
        be used as the source of genetic information, or whether a new gene pool
        will be read in instead
        '''
        # may need to save these imports along with the models
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from keras import optimizers

        if new_genepool is not None:
            gene_pool = pd.read_pickle(new_genepool)
        else:
            gene_pool = self.genotypes

        # build models from gene info
        for index, gene in gene_pool.iterrows():
            # loop through each df row-wise and use gene info from each row
            # to specify model architecture
            model = Sequential()

            # define layers
            for layer in range(gene['nb_layers']):
                if layer == 0:
                    layer_input = Dense(units=gene['layer_units'][layer],
                                        activation=gene[
                        'activations'][layer],
                        input_shape=(self.max_words,))
                else:
                    layer_input = Dense(units=gene['layer_units'][layer],
                                        activation=gene[
                        'activations'][layer])
                model.add(layer_input)

            # add final layer
            final_layer = Dense(units=self.nb_classes,
                                activation=self.C['final_activation'])
            model.add(final_layer)

            model.compile(optimizer=gene['optimizer'],
                          loss='categorical_crossentropy', metrics=['accuracy'])

            self.model_seeds_path = self.experiment_folder + '/model_seeds+' +\
                self.C['environment'] + '+Gen' + \
                str(self.current_generation)
            if not os.path.exists(self.model_seeds_path):
                os.makedirs(self.model_seeds_path)

            # save the model
            model_path = self.model_seeds_path + \
                '/' + gene['model_name']
            model.save(model_path)

    def grow_models(self, nb_splits=1, split_index=None, new_genepool=None):
        '''
        Opens saved models (e.g., which were defined, compiled, and saved in
        the in the self.seed_models function), and fits and evaluates them here

        Two types of outputs are generated:
            1) one phenotypes_df that summarizes the final evaluation metrics
                of all models grown here (one model summary per row);
                this is pickled and also saved as a property
            2) For each grown model, one [model_name]_growth_metrics_df
                is pickled; this records evaluation metrics
                for each batch and epoch

        nb_splits, split_index: These parameters are meant to
        assist in splitting model training across different provisioned VMs
        if desired.

            nb_splits is the number of segments to split the gene pool into

            split_index indicates the genepool segment that will be grown
            (trained and evaluated) in this function.

        new_genepool (if not None, it should be a string of a pickled df path+
        filename) indicates whether self.genotypes (a pandas df)should be used
        as the source of genetic information, or whether a new gene pool will
        be read in instead
        '''
        from keras.models import load_model
        from callbacks import MonitorGrowth
        from evaluations import onehot_misclassified

        if new_genepool is not None:
            gene_pool = pd.read_pickle(new_genepool)
        else:
            gene_pool = self.genotypes

        if nb_splits > 1:
            # if nb_splits>1, this indicates a fraction of the models,
            # len(gene_pool)/nb_splits will be fit and evaluated
            gene_pool = np.array_split(DNA, nb_splits)[split_index]

        # make the save dest. folder for batch- and epoch- growth
        # analysis dfs
        self.growth_analyses_path = self.experiment_folder + '/growth_analyses+'\
            + self.C['environment'] + '+Gen' + \
            str(self.current_generation)
        if not os.path.exists(self.growth_analyses_path):
            os.makedirs(self.growth_analyses_path)

        index_ = 0
        for index, gene in gene_pool.iterrows():
            print('this is the index: ', index_)
            print('and this is the gene: ', gene)
            # loop through the genotype df row-wise and pull out the model name
            # each model name is used as a handle for loading the .h5 model
            # file, which should be stored in the self.experiment folder

            model_path = self.model_seeds_path + \
                '/' + gene['model_name']
            model = load_model(model_path)

            # define callback
            monitor_growth = MonitorGrowth(gene['gene_name'],
                                           self.C['max_train_time'])

            model.fit(self.train_x, self.train_y,
                      batch_size=gene['batch_size'],
                      epochs=gene['epochs'],
                      verbose=True,
                      validation_data=(self.val_x, self.val_y),
                      callbacks=[monitor_growth])

            # save model evaluation metrics: loss, acc, time, and
            # a dictionary of miclassed datapoints
            time_lapse = datetime.now() - monitor_growth.train_start_time
            evaluation = {}
            evaluation['gene_name'] = gene['gene_name']
            score = model.evaluate(
                self.test_x, self.test_y, verbose=False)
            evaluation['test_loss'] = score[0]
            evaluation['test_accuracy'] = score[1]
            score = model.evaluate(
                self.train_x, self.train_y, verbose=False)
            evaluation['train_loss'] = score[0]
            evaluation['train_accuracy'] = score[1]
            evaluation['time'] = time_lapse.total_seconds()
            # gets a dictionary of misclassed data points including
            # mis_classed['true_class'] and mis_classed['pred_class']
            # where the value of each of key is a list with indices that
            # correspond to the same datapoint
            mis_classed = onehot_misclassified(
                model, self.test_x, self.test_y)
            # wrapping the dict in a list allows it to be slotted into a single
            # cell of a pandas df
            evaluation['misclassed'] = [mis_classed]

            # append the evaluation dict as a row to the phenotypes_df
            if index_ == 0:
                phenotypes_df = pd.DataFrame.from_dict(evaluation)
            else:
                print('in the else')
                temp_df = pd.DataFrame.from_dict(evaluation)
                phenotypes_df = pd.concat(
                    [phenotypes_df, temp_df], axis=0)

            # save (pickle) the dataframe containing the batch- and epoch-
            # evaluation metrics
            pickle_name = gene['gene_name'] + '+growth_analysis.p'
            growth_analysis_p = self.growth_analyses_path + '/' + pickle_name
            monitor_growth.df.to_pickle(growth_analysis_p)

            self.trained_models_path = self.experiment_folder +\
                '/trained_models+' +\
                self.C['environment'] + '+Gen' + \
                str(self.current_generation)
            if not os.path.exists(self.trained_models_path):
                os.makedirs(self.trained_models_path)

            # save the trained model
            model_path = self.trained_models_path + \
                '/' + gene['model_name']
            model.save(model_path)

            index_ += 1

        # save the phenotypes dataframe as a property and via pickle
        self.phenotypes = phenotypes_df
        pickle_name = self.C['environment'] + '+Gen' +\
            str(self.current_generation) + '+phenotypes.p'
        phenotypes_p = self.experiment_folder + '/' + pickle_name
        phenotypes_df.to_pickle(phenotypes_p)

    def select_parents(self, genepool=None, phenotypes=None):
        '''select winning individual genes from the genotypes DF based
        on phenotype performance

        genepool and phenotype: if None, they indicate whether that
        self.genotypes and self.phenotypes (pandas dfs) should be used as the
        source of genetic and phenotypic information. If not None, these
        parameters should be strings of pickled df path+filenames referring to
        a new gene pool or a new phenotypes df that will be read in and used
        instead of the self values.'''

        if genepool is not None:
            gene_pool = pd.read_pickle(genepool)
        else:
            gene_pool = self.genotypes

        if genepool is not None:
            gene_results = pd.read_pickle(phenotypes)
        else:
            gene_results = self.phenotypes

        # get row index for top genes subsetting
        last_top_row = round(len(gene_results) * self.C['top_tier'])
        if last_top_row == 0:
            last_top_row = 1

        if last_top_row == 1 or\
                round(len(gene_results) * self.C['random_tier']) == 0:
            num_random = 1
        else:
            num_random = round(len(gene_results) *
                               self.C['random_tier'])

        # get top genes
        top_phenotypes = gene_results.sort_values(by='test_accuracy',
                                                  ascending=False).iloc[:last_top_row]
        top_gene_names = list(top_phenotypes['gene_name'])
        top_genes = gene_pool.ix[
            gene_pool['gene_name'].isin(top_gene_names)]

        # get random genes
        random_phenotypes = gene_results.sort_values(by='test_accuracy',
                                                     ascending=False).iloc[last_top_row:].sample(n=num_random)
        random_gene_names = list(random_phenotypes['gene_name'])
        random_genes = gene_pool.ix[
            gene_pool['gene_name'].isin(random_gene_names)]

        self.parent_genes = pd.concat([top_genes, random_genes], axis=0)
