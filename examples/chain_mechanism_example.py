from mlu.modules.chain import Chain

import numpy as np

# Sample array
array = np.array([1, 2, 3, 4, 5])

# Demonstrating chaining mechanism
result = Chain(array).filter(lambda x: x > 2).aggregate('sum').value()
print("Result of chaining filter and sum operations:", result)