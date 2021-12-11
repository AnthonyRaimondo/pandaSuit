# pandaSuit <img align="right" alt="Panda in a Suit" height="138" width="96" src="https://github.com/AnthonyRaimondo/pandaSuit/raw/main/static/logo/pandaSuit-mini.ico?raw=true" title="Panda in a Suit" />
Extension of the [pandas](https://github.com/pandas-dev/pandas#what-is-it) library to encapsulate some of the most used methods of querying and manipulating DataFrames. Also featuring reversible DataFrame operations, creation of plot and dashboard objects, and methods for producing regression models.

The underlying data structure of a pandaSuit DF object is a pandas DataFrame. An AttributeError thrown while using a pandaSuit DF will fall back to \_\_setattr\_\_ or \_\_getattribute\_\_ on this underlying DataFrame, so a pandaSuit DF object can be treated as a pandas DataFrame.

The additional features merely augment the core pandas DataFrame, hence a _panda in a suit_.


### How to install - [PyPI](https://pypi.org/project/pandaSuit/)
```pip install pandaSuit```


### Dependencies
* [pandas](https://github.com/pandas-dev/pandas#where-to-get-it) >=1.3.4
* [scikit-learn](https://github.com/scikit-learn/scikit-learn#user-installation) >=1.0.1


### Examples
<h3>Manipulation</h3>
__<h4>Update values at specific locations</h4>__
<h3>Querying</h3>
__<h4>Query DF object by column name</h4>__
<img alt="select-by-column-name" width="768" src="https://github.com/AnthonyRaimondo/pandaSuit/raw/main/static/examples/select-by-column-name.PNG?raw=true" title="select-by-column-name" />
____________________________________________________________________________________________________<br /><br /><br />
__<p>Or by index, possibly for rows and columns</p>__
<img alt="select-by-index" width="768" src="https://github.com/AnthonyRaimondo/pandaSuit/raw/main/static/examples/select-by-index.PNG?raw=true" title="select-by-index" />
____________________________________________________________________________________________________<br /><br /><br />
__<p>Equivalent to existing pandas functionality</p>__
<img alt="use-core-pandas-methods" width="768" src="https://github.com/AnthonyRaimondo/pandaSuit/raw/main/static/examples/use-core-pandas-methods.PNG?raw=true" title="use-core-pandas-methods" />
