
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from operator import mul
from functools import reduce
import copy
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from typing import List


# dir = os.getcwd()
dir = os.path.dirname(__file__)

from dtype import time_bar_dtype, run_bar_dtype


def main():
    data_path = os.path.join(dir, "data/bars.parquet")

    input_dtype = np.dtype([(k, v) for k, v in time_bar_dtype.items()])
    output_dtype = np.dtype([(k, v) for k, v in run_bar_dtype.items()])

    time_bar = pd.read_parquet(data_path).reset_index(drop=True)
    input_array = time_bar.to_records(index=False).astype(input_dtype)
    input_bytes = input_array.tobytes()

    print(time_bar)
    print("Num bars:", input_array.shape[0])
    print("Bytes per record:", input_array.dtype.itemsize)

    from cpp import Pipeline  # type: ignore
    try:
        output_bytes = Pipeline.process_bars(input_bytes, input_array.shape[0])
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Caught exception:", str(e))

    vrun_bar = pd.DataFrame(np.frombuffer(output_bytes, dtype=output_dtype))
    # vrun_bar['return'] = np.log1p(vrun_bar['close'].pct_change().fillna(0))
    print(vrun_bar)

    from collections import deque
    lookback = 24
    bars_per_pip = 4
    n_pips = int(lookback/bars_per_pip)
    hold_period = int(lookback/2)

    price = ((time_bar['high'] + time_bar['low'])/2).to_numpy(dtype=np.float32)
    miner = PIPPatternMiner(n_pips, lookback, hold_period)
    miner.train(price)
    miner.predict()

    # import plotly.graph_objects as go
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(y=df['close'], mode='lines+markers', showlegend=False))
    # for result in tqdm(results):
    #     fig.add_trace(go.Scatter(x=result[0], y=result[1], mode='lines', showlegend=False))
    # fig.show()


class PIPPatternMiner:
    """
    Perceptually Important Point (PIP) Pattern Miner that:
    1. Identifies PIP patterns in price data
    2. Clusters similar patterns
    3. Selects best performing patterns for trading signals
    4. Evaluates performance using Martin ratio
    """

    def __init__(self, n_pips: int, lookback: int, hold_period: int):
        """
        Initialize pattern miner with key parameters

        Parameters:
        -----------
        n_pips : int
            Number of pivot points to identify in each pattern
        lookback : int 
            Window size (in bars) for pattern detection
        hold_period : int
            Number of bars to hold positions after pattern detection
        """
        self._n_pips = n_pips
        self._lookback = lookback
        self._hold_period = hold_period
        self._dist_measure = 1  # 1 - Euclidean 2 - Perpendicular 3 - Vertical

        # Pattern storage
        self._unique_pip_patterns = []                      # Stores normalized PIP patterns
        # self._unique_pip_futures = []
        self._unique_pip_indices = []                       # Stores indices where patterns occur
        self._unique_pip_martins = []                       # Stores martins as label where patterns occur

        # Clusters
        self._num_clusters = 0
        self._pip_clusters_centers: List[List[float]] = []  # Stores cluster centroids
        self._pip_clusters_indexes: List[List[int]] = []    # Stores cluster assignments
        self._pip_clusters_martins: List[List[float]] = []  # Stores cluster labels

        # Performance metrics
        self._labels_mean: List[float] = []
        self._labels_max: List[float] = []
        self._labels_min: List[float] = []
        self._long_clusters = []
        self._short_clusters = []

        # Data storage
        self._data = np.array([])                           # Array of log closing prices

    def _find_pips(self, data: np.ndarray):
        """
        Find 'n_pips' perceptually important points (PIPs) from a 1D price data array.
        Returns:
            pips_x (list): x-coordinates (indices) of PIPs
            pips_y (list): y-coordinates (price values) of PIPs
        """

        # Start with the first and last points as initial PIPs
        pips_x = [0, len(data) - 1]        # Indices
        pips_y = [data[0], data[-1]]      # Prices

        # Iteratively add one PIP until we have n_pips
        for curr_point in range(2, self._n_pips):
            md = 0.0               # Maximum distance found so far
            md_i = -1              # Index of point with max distance
            insert_index = -1      # Where to insert this point in the list

            # Go through each segment defined by two existing PIPs
            for k in range(0, curr_point - 1):
                left_adj = k
                right_adj = k + 1

                # Compute line parameters between two PIPs
                time_diff = pips_x[right_adj] - pips_x[left_adj]
                price_diff = pips_y[right_adj] - pips_y[left_adj]
                slope = price_diff / time_diff
                intercept = pips_y[left_adj] - pips_x[left_adj] * slope

                # Evaluate points between left_adj and right_adj
                for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                    d = 0.0
                    if self._dist_measure == 1:
                        # Euclidean distance to both endpoints
                        d = ((pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2) ** 0.5
                        d += ((pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2) ** 0.5
                    elif self._dist_measure == 2:
                        # Perpendicular distance from the line
                        d = abs((slope * i + intercept) - data[i]) / ((slope ** 2 + 1) ** 0.5)
                    else:
                        # Vertical distance from the line
                        d = abs((slope * i + intercept) - data[i])

                    # Update max distance if this point is farther
                    if d > md:
                        md = d
                        md_i = i
                        insert_index = right_adj

            # Insert the point with the maximum distance
            pips_x.insert(insert_index, md_i)
            pips_y.insert(insert_index, data[md_i])

        return pips_x, pips_y

    def _find_martin(self, data: np.ndarray) -> float:
        """
        Compute the Martin ratio from a price series using only NumPy.
        Equivalent to _get_martin(), which expects log-returns.

        Martin Ratio = Total Log Return / Ulcer Index
        """
        if len(data) < 2:
            return 0.0

        # Log returns
        rets = np.diff(data)
        total_return = np.sum(rets)
        short = False

        # If return is negative, treat as short position
        if total_return < 0.0:
            rets *= -1
            total_return *= -1
            short = True

        # Reconstruct equity curve
        csum = np.cumsum(rets)
        eq = np.exp(csum)

        # Compute drawdown and Ulcer Index
        running_max = np.maximum.accumulate(eq)
        drawdown = (eq / running_max) - 1.0
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))

        min_ulcer_index = max(1e-4, total_return * 0.05)  # ensure martin cannot exceed N* total returns
        if ulcer_index < min_ulcer_index:
            ulcer_index = min_ulcer_index

        martin = total_return / ulcer_index
        return -martin if short else martin

    def _find_unique_patterns(self):
        """Identify all unique PIP patterns in the data"""
        self._unique_pip_indices.clear()
        self._unique_pip_patterns.clear()
        self._unique_pip_martins.clear()

        # Track last pattern to avoid duplicates
        last_pips_x = [0] * self._n_pips

        assert isinstance(self._data, np.ndarray)

        # Slide window through data
        for i in tqdm(range(self._lookback - 1, len(self._data) - self._hold_period)):
            if (i % 4) != 0:
                continue
            
            start_i = i - self._lookback + 1
            window_lookback = self._data[start_i: i + 1]
            window_hold = self._data[i: i + self._hold_period]

            # Find PIPs in current window
            pips_x, pips_y = self._find_pips(window_lookback)
            pips_x = [j + start_i for j in pips_x]  # Convert to global indices
            # pips_y = np.concatenate([pips_y, window_hold])

            # Check if internal PIPs are same as last pattern (avoid duplicates)
            # https://link.springer.com/chapter/10.1007/11539506_146
            # only check index for trivial matches
            # conservative, as it still allow identical but shifted pattern to pass
            # to have strong filtering effect, lookback cannot be too large
            same = True
            for j in range(1, self._n_pips - 1):  # discard ~50%
                if pips_x[j] != last_pips_x[j]:
                    same = False
                    break

            if not same:
                # Normalize pattern by z-scoring
                pips_y = list((np.array(pips_y) - np.mean(pips_y)) / (np.std(pips_y) + 1e-8))
                self._unique_pip_indices.append(i)
                self._unique_pip_patterns.append(pips_y)
                # self._unique_pip_futures.append(pips_y)
                self._unique_pip_martins.append(self._find_martin(window_hold))

            last_pips_x = pips_x

    def _kmeans_cluster_patterns(self):
        """
        Cluster PIP patterns using k-means++
        """
        # Initialize centers using k-means++ algorithm
        initial_centers = kmeans_plusplus_initializer(self._unique_pip_patterns, self._num_clusters).initialize()

        # Perform k-means clustering
        kmeans_instance = kmeans(self._unique_pip_patterns, initial_centers)
        kmeans_instance.process()

        # Store results
        self._pip_clusters_centers = kmeans_instance.get_centers()  # type: ignore
        self._pip_clusters_indexes = kmeans_instance.get_clusters()  # type: ignore

    def _get_cluster_performance(self):
        self._labels_mean = []
        self._labels_max = []
        self._labels_min = []
        for i in range(self._num_clusters):
            martins: List[float] = [self._unique_pip_martins[idx] for idx in self._pip_clusters_indexes[i]]
            self._pip_clusters_martins.append(martins)
            self._labels_mean.append(float(np.mean(martins)))
            self._labels_max.append(float(np.max(martins)))
            self._labels_min.append(float(np.min(martins)))

        long_to_short = np.argsort(self._labels_mean)[::-1]  # descending order (high → low)
        good_clusters = int(self._num_clusters/8)
        self._long_clusters = long_to_short[:good_clusters]
        self._short_clusters = long_to_short[-good_clusters:]
        print(f"best patterns from long to short: {long_to_short}:{[float(self._labels_mean[i]) for i in long_to_short]}")

    def train_VAE(self):
        from VAE import VAE
        import numpy as np
        import plotly.graph_objects as go
        from scipy.stats import gaussian_kde
        import umap
        from sklearn.mixture import GaussianMixture

        # Step 1: Prepare data
        X = np.array(self._unique_pip_patterns, dtype=np.float32)
        # Apply exponential weights to columns (features): heavier on recent points
        num_features = X.shape[1]
        exp_weights = np.linspace(1.0, 10.0, num_features, dtype=np.float32)  # or use np.geomspace(1.0, 2.0, num=num_features)

        X_weighted = X * exp_weights  # shape: (N, D), element-wise weighting
        
        vae = VAE(input_dim=X_weighted.shape[1], hidden_dim=64, latent_dim=8, epochs=200)
        vae.train_model(X_weighted)
        latent_means = vae.encode_data(X_weighted)  # (N, latent_dim)

        # Step 2: UMAP to 3D
        n_neighbors = min(15, X_weighted.shape[0] - 1)
        algo_umap = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.1)
        X_3d = algo_umap.fit_transform(latent_means)  # (N, 3)

        # Step 3: GMM Clustering in Latent Space
        self._num_clusters = 40  # change as needed
        gmm = GaussianMixture(n_components=self._num_clusters, covariance_type='full', random_state=42)
        gmm.fit(latent_means)
        cluster_labels = gmm.predict(latent_means)

        self._pip_clusters_centers = gmm.means_  # shape: (n_components, latent_dim)
        self._pip_clusters_indexes = [
            np.where(cluster_labels == i)[0].tolist()
            for i in range(self._num_clusters)
        ]

        # Step 4: 3D KDE
        kde = gaussian_kde(X_3d.T)
        grid_size = 50
        x = np.linspace(X_3d[:, 0].min(), X_3d[:, 0].max(), grid_size)
        y = np.linspace(X_3d[:, 1].min(), X_3d[:, 1].max(), grid_size)
        z = np.linspace(X_3d[:, 2].min(), X_3d[:, 2].max(), grid_size)
        Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
        coords = np.vstack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
        density = kde(coords).reshape(grid_size, grid_size, grid_size)

        # Step 5: Create Plot
        fig = go.Figure()

        # Volumetric KDE
        fig.add_trace(go.Volume(
            x=Xg.flatten(),
            y=Yg.flatten(),
            z=Zg.flatten(),
            value=density.flatten(),
            isomin=density.min(),
            isomax=density.max(),
            opacity=0.05,
            surface_count=15,
            colorscale='Viridis',
            showscale=False
        ))

        # Clustered points in 3D
        fig.add_trace(go.Scatter3d(
            x=X_3d[:, 0],
            y=X_3d[:, 1],
            z=X_3d[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=cluster_labels,  # categorical color
                colorscale='Rainbow',
                opacity=0.9,
                colorbar=dict(title='Cluster')
            ),
            text=[f"Cluster {label}" for label in cluster_labels],
            name='Clustered Samples'
        ))

        fig.update_layout(
            title="3D Volumetric KDE with GMM Clustered Latent Points",
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3'
            )
        )
        fig.show()



    def train(self, arr: np.ndarray):
        """
        Train pattern miner on price data

        Parameters:
        -----------
        arr : np.array
            Array of log prices
        n_reps : int
            Number of permutation tests to run (-1 for none)
        """
        data = arr.astype(np.float32)  # work on log price for pattern recognition
        self._data = data[:10000]
        self._test = data[10000:20000]
        # self._returns = np.append(np.diff(self._data)[1:], np.nan)

        # Step 1: Find all unique PIP patterns
        self._find_unique_patterns()
        # self.plot_unique_samples()

        self.train_VAE()
        

        self._get_cluster_performance()


    def predict(self):
        """
        Predict trading signal (long/short/neutral) for new PIP pattern
        """
        
        hold = int(self._hold_period/2)
        
        signals = np.zeros(len(self._test), dtype=int)
        for i in tqdm(range(self._lookback - 1, len(self._test) - self._hold_period)):
            start_i = i - self._lookback + 1
            window_lookback = self._test[start_i: i + 1]

            pips_x, pips_y = self._find_pips(window_lookback)
            pips_x = [j + start_i for j in pips_x]  # Convert to global indices

            norm_y = (np.array(pips_y) - np.mean(pips_y)) / (np.std(pips_y) + 1e-8)

            # Find closest cluster center
            best_dist = 1.e30
            best_clust = -1
            for clust_i in range(len(self._pip_clusters_centers)):
                center = np.array(self._pip_clusters_centers[clust_i])
                dist = np.linalg.norm(norm_y-center)
                if dist < best_dist:
                    best_dist = dist
                    best_clust = clust_i
                    
            # Return appropriate signal
            if best_clust in self._long_clusters:
                signals[i: i + hold] = 1
            elif best_clust in self._short_clusters:
                signals[i: i + hold] = -1

        cum_long = 0
        cum_short = 0
        cum_long_list = []
        cum_short_list = []
        ret = np.append(np.diff(self._test), 0.0)
        direction = 0
        for i, signal in enumerate(signals):
            if signal != 0:
                direction = signal
                
            if direction == 1:
                cum_long += ret[i]
            elif direction == -1:
                cum_short += -ret[i]
            cum_long_list.append(cum_long)
            cum_short_list.append(cum_short)

        cum_total_list = cum_long_list + cum_short_list

        import plotly.graph_objects as go
        fig = go.Figure()

        fig.add_trace(go.Scatter(y=cum_long_list))
        fig.add_trace(go.Scatter(y=cum_short_list))
        # fig.add_trace(go.Scatter(y=cum_total_list))
        fig.show()

        fig = go.Figure()
        colors = np.where(signals == 1, 'red', np.where(signals == -1, 'green', 'gray'))
        l = 5000
        fig = go.Figure(go.Scatter(
            x=np.arange(l),
            y=self._test[:l],
            mode='markers',
            marker=dict(color=colors, size=6),
            name='Signal Colored'
        ))
        fig.show()

if __name__ == "__main__":
    main()
