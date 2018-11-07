This is an implementation of the **LOCI** (local correlation integral) fast outlier detection algorithm in Python, based on the paper:  
Papadimitriou, S., Kitagawa, H., Gibbons, P.B. and Faloutsos, C., 2003, March. *Loci: Fast outlier detection using the local correlation integral*. In Data Engineering, 2003. Proceedings. 19th International Conference on (pp. 315-326). IEEE.

This is an initial first attempt implementation, it is functional (I think) however performance is very limited.


##### Installation

```bash
pip install loci
```

##### Dependencies

- Scipy, Numpy
- Python version 3.6 (probably works with 3.x, not tested though)

##### Example

```python
import numpy as np
import matplotlib.pyplot as plt

from loci.loci import run_loci

data = np.concatenate((np.array([(10, 10), ]), np.random.normal(50, 10, (200, 2))))

loci_res = run_loci(data)
outlier_indices = loci_res.outlier_indices

plt.scatter(data[:, 0], data[:, 1])
plt.scatter(data[outlier_indices, 0], data[outlier_indices, 1], c='r')
plt.show()
```


