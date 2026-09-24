# FedDFHMC

**Federated Deep Matrix Factorization with Hypergraph Regularization for Multi-view Clustering**

Run the HW2sources experiment from the repository root:

```matlab
main
```

The script uses `seed=1`, evaluates the final representation with ACC, NMI, ARI, Purity, and FScore, and saves the results in `results/`.

The `secure_aggregation_reference/` directory provides a readable and reproducible MATLAB implementation of the secure aggregation protocol described in the paper. It is not intended to be a production cryptographic system.

The required GSPBox functions are included. If a GSP-related error still occurs, install GSPBox through the MATLAB Add-On Explorer or MATLAB File Exchange, and ensure it is available on the MATLAB path.
