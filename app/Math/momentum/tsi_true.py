import array

class tsi_true:
    def __init__(self,
                 closes: array.array,
                 first_period: int = 25,   # First EMA period (default 25)
                 second_period: int = 13,  # Second EMA period (default 13)
                 ):
        self.closes = closes
        self.first_period = first_period
        self.second_period = second_period
        
        # Initialize arrays for final TSI values only
        self.tsi = array.array('d', [])
        
        # Store previous values for EMA calculations
        self.pc_prev = 0.0       # Previous Price Change
        self.pcs_prev = 0.0      # Previous PCS (first EMA of PC)
        self.pcds_prev = 0.0     # Previous PCDS (second EMA of PCS)
        self.apc_prev = 0.0      # Previous Absolute Price Change
        self.apcs_prev = 0.0     # Previous APCS (first EMA of APC)
        self.apcds_prev = 0.0    # Previous APCDS (second EMA of APCS)
        
        self.data_points = 0
        
    def calculate_ema(self, current_value: float, prev_ema: float, period: int, is_first_value: bool) -> float:
        """Calculate EMA for a single value"""
        alpha = 2.0 / (period + 1)
        if is_first_value:
            return current_value
        return current_value * alpha + prev_ema * (1 - alpha)
    
    def update(self):
        """
        Calculate TSI using double EMA smoothing:
        TSI = (PCDS/APCDS) × 100
        
        Where:
        PC = Current Close - Previous Close
        PCS = first_period EMA of PC
        PCDS = second_period EMA of PCS
        APC = |PC| (Absolute Price Change)
        APCS = first_period EMA of APC
        APCDS = second_period EMA of APCS
        """
        self.data_points += 1
        
        # Need at least 2 data points to calculate price change
        if self.data_points < 2:
            self.tsi.append(0.0)
            return
            
        # Calculate PC (Price Change)
        pc = self.closes[-1] - self.closes[-2]
        apc = abs(pc)
        
        # For the first value, use it as initial EMA
        is_first = self.data_points == 2
        
        # Calculate first smoothing (PCS and APCS)
        pcs = self.calculate_ema(pc, self.pcs_prev, self.first_period, is_first)
        apcs = self.calculate_ema(apc, self.apcs_prev, self.first_period, is_first)
        
        # Calculate second smoothing (PCDS and APCDS)
        pcds = self.calculate_ema(pcs, self.pcds_prev, self.second_period, is_first)
        apcds = self.calculate_ema(apcs, self.apcds_prev, self.second_period, is_first)
        
        # Store values for next iteration
        self.pcs_prev = pcs
        self.pcds_prev = pcds
        self.apcs_prev = apcs
        self.apcds_prev = apcds
        
        # Calculate final TSI value
        tsi = 0.0
        if apcds != 0:
            tsi = (pcds / apcds) * 100
            
        self.tsi.append(tsi)
        LEN = 100
        if len(self.tsi) > 2*LEN:
            del self.tsi[:-LEN]
        return