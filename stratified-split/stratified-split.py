import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    X = np.array(X)
    y = np.array(y)

    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    train_indices = []
    test_indices = []

    classes = np.unique(y)

    for cls in classes:
        cls_indices = np.where(y == cls)[0]

        rng.shuffle(cls_indices)

        n_test = round(len(cls_indices) * test_size)

        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])

    # ✅ sort indices to match reference
    train_indices = np.sort(np.array(train_indices, dtype=int))
    test_indices = np.sort(np.array(test_indices, dtype=int))

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test