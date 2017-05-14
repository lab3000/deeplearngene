deeplearngene Demos
====================

Two demos are provided here. Both utilize the Reuters news dataset from Keras.

If you would like to run these demo notebooks, they should be moved to the same folder as the scripts.

**Demo1** is an overly simplistic scenario. The chosen hyperparameters, especially a population size of 3 models, does not lead to effective model optimization. Instead, this demo is meant to highlight the functionality of clade objects, and show the basic workflow of evolving a generation of models via a clade object.
- Demo1 is accompanied by a pdf file that has screen shots that show the progression of running each cell in the notebook; these screenshots are intended to highlight:
 * files that are generated and saved when clade functions are run
 * suggested usage of an experiment notebook to accompany interactive use of deeplearngene.

 - An experiment notebook reference, input as the 'environment' to the clade object, links the name of the clade object variable, the experiment folder name (i.e. the destination of clade function output), the filenames of files output by clade functions, the ipynb filename, and notes in the notebook that journal the interactive steps during the experiment. This is a recommend workflow, but the code defining the modules does not force it.


**Demo2** demonstrates a more realistic model evolution process. 30 models are produced in each generation, and 3 generations are run.


# Links
The experiment notebook page associated with these demos can be found [here](https://www.evernote.com/l/AOQRky_C_25IH46VJT2UmqX40x5GOpwuA3E).
The experiment folders associated with these demos can be found [here](https://www.amazon.com/clouddrive/share/3jJRoXtk5DgZuBjHi2eX7OAKSDT3RclDSDNuIgeymbV?ref_=cd_ph_share_link_copy).
