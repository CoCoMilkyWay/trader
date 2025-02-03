import numpy as np

class Primitives_CPU:
    """Time series operations following torch implementations"""
    @staticmethod
    def f(x: float) -> float:
        """Basic addition"""
        return x
    
    @staticmethod
    def and_(x: bool, y: bool) -> bool:
        """Basic addition"""
        return x and y
    
    @staticmethod
    def or_(x: bool, y: bool) -> bool:
        """Basic addition"""
        return x or y

    @staticmethod
    def ge(x: float, y: float) -> bool:
        """Basic addition"""
        return x >= y

    @staticmethod
    def se(x: float, y: float) -> bool:
        """Basic addition"""
        return x <= y

    @staticmethod
    def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Basic addition"""
        return x + y
        
    @staticmethod
    def sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Basic subtraction"""
        return x - y
        
    @staticmethod
    def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Basic multiplication"""
        return x * y
        
    @staticmethod
    def div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Protected division"""
        return np.divide(x, y, out=np.ones_like(x), where=y!=0)
        
    @staticmethod
    def log(x: np.ndarray) -> np.ndarray:
        """Natural logarithm"""
        return np.log(np.abs(x) + 1e-10)
        
    @staticmethod
    def sqrt(x: np.ndarray) -> np.ndarray:
        """Square root"""
        return np.sqrt(np.abs(x))
        
    @staticmethod
    def neg(x: np.ndarray) -> np.ndarray:
        """Negation"""
        return -x
        
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation: 1/(1+exp(-x))"""
        return 1 / (1 + np.exp(-x))
        
    @staticmethod
    def sign(x: np.ndarray) -> np.ndarray:
        """Sign function: 1 if positive, -1 if negative, 0 if zero"""
        return np.sign(x)

    @staticmethod
    def delay(x: np.ndarray, d: int) -> np.ndarray:
        """Delay time series by d periods"""
        result = np.zeros_like(x)
        result[d:] = x[:-d]
        return result
        
    @staticmethod
    def delta(x: np.ndarray, d: int) -> np.ndarray:
        """d-period difference"""
        return x - Primitives_CPU.delay(x, d)
        
    @staticmethod
    def delay_pct(x: np.ndarray, d: int) -> np.ndarray:
        """d-period percentage change"""
        delayed = Primitives_CPU.delay(x, d)
        return np.divide(x - delayed, delayed, out=np.zeros_like(x), where=delayed!=0)
        
    @staticmethod
    def correlation(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
        """d-period rolling correlation"""
        result = np.full_like(x, np.nan)
        for i in range(d-1, len(x)):
            x_window = x[i-d+1:i+1]
            y_window = y[i-d+1:i+1]
            if np.all(~np.isnan(x_window)) and np.all(~np.isnan(y_window)):
                result[i] = np.corrcoef(x_window, y_window)[0,1]
        return result
        
    @staticmethod
    def covariance(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
        """d-period rolling covariance"""
        result = np.full_like(x, np.nan)
        for i in range(d-1, len(x)):
            x_window = x[i-d+1:i+1]
            y_window = y[i-d+1:i+1]
            if np.all(~np.isnan(x_window)) and np.all(~np.isnan(y_window)):
                result[i] = np.cov(x_window, y_window)[0,1]
        return result
        
    @staticmethod
    def argmin(x: np.ndarray, d: int) -> np.ndarray:
        """Rolling argmin over d periods"""
        result = np.zeros_like(x)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            result[i] = np.argmin(window)
        return result

    @staticmethod
    def argmax(x: np.ndarray, d: int) -> np.ndarray:
        """Rolling argmax over d periods"""
        result = np.zeros_like(x)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            result[i] = np.argmax(window)
        return result
        
    @staticmethod
    def rank(x: np.ndarray, d: int) -> np.ndarray:
        """Rolling rank over d periods"""
        result = np.zeros_like(x)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            result[i] = np.argsort(np.argsort(window))[-1] / (d-1)
        return result

    @staticmethod
    def decay_linear(x: np.ndarray, d: int) -> np.ndarray:
        """Linear decay weight sum over d periods"""
        weights = np.arange(d, 0, -1)
        result = np.zeros_like(x)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
        
    @staticmethod
    def min(x: np.ndarray, d: int) -> np.ndarray:
        """Rolling minimum over d periods"""
        result = np.zeros_like(x)
        for i in range(d-1, len(x)):
            result[i] = np.min(x[i-d+1:i+1])
        return result
        
    @staticmethod
    def max(x: np.ndarray, d: int) -> np.ndarray:
        """Rolling maximum over d periods"""
        result = np.zeros_like(x)
        for i in range(d-1, len(x)):
            result[i] = np.max(x[i-d+1:i+1])
        return result
        
    @staticmethod
    def stddev(x: np.ndarray, d: int) -> np.ndarray:
        """Rolling standard deviation over d periods"""
        result = np.zeros_like(x)
        for i in range(d-1, len(x)):
            result[i] = np.std(x[i-d+1:i+1])
        return result
        
    @staticmethod
    def sum(x: np.ndarray, d: int) -> np.ndarray:
        """Rolling sum over d periods"""
        result = np.zeros_like(x)
        for i in range(d-1, len(x)):
            result[i] = np.sum(x[i-d+1:i+1])
        return result
        
    @staticmethod
    def meandev(x: np.ndarray, d: int) -> np.ndarray:
        """Rolling mean absolute deviation over d periods"""
        result = np.zeros_like(x)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            result[i] = np.mean(np.abs(window - np.mean(window)))
        return result

    @staticmethod
    def groupby_ascend(x: np.ndarray, y: np.ndarray, d: int, n: int) -> np.ndarray:
        """Group by ascending values"""
        result = np.zeros_like(x)
        for i in range(d-1, len(x)):
            window_x = x[i-d+1:i+1]
            window_y = y[i-d+1:i+1]
            sorted_indices = np.argsort(window_y)
            group = np.searchsorted(np.linspace(0, d, n+1), 
                                  np.where(sorted_indices == d-1)[0][0])
            result[i] = np.mean(window_x[sorted_indices[group*d//n:(group+1)*d//n]])
        return result
        
    @staticmethod
    def groupby_descend(x: np.ndarray, y: np.ndarray, d: int, n: int) -> np.ndarray:
        """Group by descending values"""
        result = np.zeros_like(x)
        for i in range(d-1, len(x)):
            window_x = x[i-d+1:i+1]
            window_y = y[i-d+1:i+1]
            sorted_indices = np.argsort(-window_y)
            group = np.searchsorted(np.linspace(0, d, n+1), 
                                  np.where(sorted_indices == d-1)[0][0])
            result[i] = np.mean(window_x[sorted_indices[group*d//n:(group+1)*d//n]])
        return result

    @staticmethod
    def groupby_diff(x: np.ndarray, y: np.ndarray, d: int, n: int) -> np.ndarray:
        """Group by value differences"""
        asc = Primitives_CPU.groupby_ascend(x, y, d, n)
        desc = Primitives_CPU.groupby_descend(x, y, d, n)
        return asc - desc
        
    @staticmethod
    def rank_pct(x: np.ndarray) -> np.ndarray:
        """Full history percentile rank"""
        return np.argsort(np.argsort(x)).astype(float) / (len(x) - 1)
        
    @staticmethod
    def rank_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Average of ranks"""
        return (Primitives_CPU.rank_pct(x) + Primitives_CPU.rank_pct(y)) / 2
        
    @staticmethod
    def rank_sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Difference of ranks"""
        return Primitives_CPU.rank_pct(x) - Primitives_CPU.rank_pct(y)
        
    @staticmethod
    def rank_div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ratio of ranks"""
        return np.divide(Primitives_CPU.rank_pct(x), Primitives_CPU.rank_pct(y),
                        out=np.ones_like(x), where=Primitives_CPU.rank_pct(y)!=0)
