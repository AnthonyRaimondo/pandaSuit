# pandaSuit <img align="right" alt="Panda in a Suit" height="138" width="96" src="https://github.com/AnthonyRaimondo/pandaSuit/blob/main/pandaSuit-mini.ico?raw=true" title="Panda in a Suit" />
Extension of the pandas library to encapsulate some of the most used methods of querying and manipulating DataFrames. Also featuring reversible DataFrame operations, creation of plot and dashboard objects, and methods for producing regression models.

The underlying data structure of a pandaSuit DF object is a pandas DataFrame. An AttributeError thrown while using a pandaSuit DF will fall back to \_\_setattr\_\_ or \_\_getattribute\_\_ on this underlying DataFrame, so a pandaSuit DF object can be treated as a pandas DataFrame.

The additional features merely augment the core pandas DataFrame, hence a <ins>panda in a suit</ins>.
