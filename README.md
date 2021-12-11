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
__<p>Query DF object by column name</p>__
<img align="left" width="256" alt="select-by-column-name" src="https://github.com/AnthonyRaimondo/pandaSuit/raw/main/static/examples/select-by-column-name.PNG?raw=true" title="select-by-column-name" />
____________________________________________________________________________________________________<br /><br /><br />
__<p>Or by index, possibly for rows and columns</p>__
<img alt="select-by-index" src="https://github.com/AnthonyRaimondo/pandaSuit/raw/main/static/examples/select-by-index.PNG?raw=true" title="select-by-index" />
____________________________________________________________________________________________________<br /><br /><br />
__<p>Equivalent to existing pandas functionality</p>__
<img alt="use-core-pandas-methods" src="https://github.com/AnthonyRaimondo/pandaSuit/raw/main/static/examples/use-core-pandas-methods.PNG?raw=true" title="use-core-pandas-methods" />
