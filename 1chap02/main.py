# coding = 'utf-8'
import time

import numpy as np
import pandas as pd
import tm


def main():
    y = np.random.randint(2, size=(100000, 1))
    x = np.random.randint(10, size=(100000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    start = time.time()
    result_1 = tm.target_mean_v3(data, 'y', 'x')
    end = time.time()
    print(end - start)

    start = time.time()
    result_2 = tm.target_mean_v4(data, 'y', 'x')
    end = time.time()
    print(end - start)
    diff = np.linalg.norm(result_1 - result_2)
    print(diff)


if __name__ == '__main__':
    main()
