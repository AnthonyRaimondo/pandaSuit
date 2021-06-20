import pytest

from pandaSuit.df import DF


@pytest.fixture(scope="function")
def sample_df() -> DF:
    data = [{'a': 1, 'b': 2, 'c': 3},
            {'a': 4, 'b': 5, 'c': 6},
            {'a': 7, 'b': 8, 'c': 9}]
    return DF(data=data)


@pytest.fixture(scope="function")
def sample_df_with_row_names() -> DF:
    data = [{'a': 1, 'b': 2, 'c': 3},
            {'a': 4, 'b': 5, 'c': 6},
            {'a': 7, 'b': 8, 'c': 9}]
    df = DF(data=data)
    df.df = df.df.rename({0: 'd', 1: 'e', 2: 'f'}, axis="index")
    return df


class TestDF:

    def test_select_by_index(self, sample_df):
        result = sample_df.select(row=0)
        assert result.shape == (3,)

        result = sample_df.select(column=0)
        assert result.shape == (3,)

        result = sample_df.select(row=[0, 1])
        assert result.shape == (2, 3)

        result = sample_df.select(column=[0, 1])
        assert result.shape == (3, 2)

        result = sample_df.select(row=0, column=0)
        assert result == 1

        result = sample_df.select(row=[0, 1], column=[0, 1])
        assert result.shape == (2, 2)

    def test_select_by_name(self, sample_df_with_row_names):
        result = sample_df_with_row_names.select(row='d')
        assert result.shape == (3,)

        result = sample_df_with_row_names.select(column='a')
        assert result.shape == (3,)

        result = sample_df_with_row_names.select(row=['d', 'e'])
        assert result.shape == (2, 3)

        result = sample_df_with_row_names.select(column=['a', 'b'])
        assert result.shape == (3, 2)

        result = sample_df_with_row_names.select(row='d', column='a')
        assert result == 1

        result = sample_df_with_row_names.select(row=['d', 'e'], column=['a', 'b'])
        assert result.shape == (2, 2)
