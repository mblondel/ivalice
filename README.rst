.. -*- mode: rst -*-

ivalice
=======

Boosting and ensemble learning library in Python.

Algorithms supported:

- Classification and regression trees (work in progress)
- Random forests (work in progress)
- Gradient Boosting
- McRank
- LambdaMART

ivalice follows the `scikit-learn <http://scikit-learn.org>`_ API conventions.
Computationally demanding parts are implemented using `Numba
<http://numba.pydata.org>`_.

Dependencies
------------

ivalice needs Python >= 2.7, setuptools, Numpy >= 1.3, SciPy >= 0.7,
scikit-learn >= 0.15.1 and Numba >= 0.13.4.

To run the tests you will also need nose >= 0.10.

Installation
------------

To install ivalice from pip, type::

    pip install https://github.com/mblondel/ivalice/archive/master.zip

To install ivalice from source, type::

  git clone https://github.com/mblondel/ivalice.git
  cd ivalice
  sudo python setup.py install

On Github
---------

https://github.com/mblondel/ivalice

Author
------

Mathieu Blondel, 2014-present
