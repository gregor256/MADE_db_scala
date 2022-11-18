import numpy as np
import pandas as pd
import click


@click.command(name="get_threads")
@click.argument('_n_objects')
def gen_data(_n_objects):
    n_objects = int(_n_objects)
    n_features = 4
    border = int(0.66 * n_objects)
    w_true = np.random.normal(size=(n_features,))
    x = np.random.uniform(-5, 5, (n_objects, n_features))
    x *= (np.arange(n_features) * 2 + 1)[np.newaxis, :]
    y = x.dot(w_true) + np.random.normal(0, 1, n_objects)

    data = pd.DataFrame(x.copy())
    data[n_features] = y

    train_data_frame = data.iloc[:border, :]
    test_data_frame = data.iloc[border:, :]

    train_data_frame.to_csv("data/train_data.csv", index=False)
    test_data_frame.to_csv("data/test_data.csv", index=False)


if __name__ == '__main__':
    gen_data()
