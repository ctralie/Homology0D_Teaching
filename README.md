# Lecture on 0D Persistent Homology

This notebook is meant to cover the basics of lower star filtrations, with a focus on 0D homology.   It covers the following topics

* 0D point cloud filtrations
* 1D Time Series Lower Star Filtrations
* Lower star on images
* Lower star on 3D shapes
* Merge Trees

Directions are below for installing the code.  Please e-mail chris.tralie@gmail.com if you encounter any issues.

## Installing Jupyter Notebook And Other Dependencies

To run these modules, you will need to have jupyter notebook installed with a *Python 3* backend with numpy, scipy, and matplotlib.  If you don't have this on your computer yet, the easiest way to install this is with Anaconda:

<a href = "https://www.anaconda.com/download/">https://www.anaconda.com/download/</a>

Once you have downloaded and installed all of these packages, type the following commands in the terminal, which will install dependencies

~~~~~ bash
pip install cython
pip install ripser
~~~~~

## Running the code

First, check out the code from github

~~~~~ bash
git clone --recursive https://github.com/ctralie/Homology0D_Teaching.git
cd Homolgy0D_Teaching
~~~~~

At the root of this directory, type

~~~~~ bash
jupyter notebook
~~~~~

This will launch a browser window, where you can run the modules.  Click on ConnectedComponentsLecture.ipynb.  If you want to see the solutions we went through in class, click on ConnectedComponentsLecture_Completed.ipynb.
