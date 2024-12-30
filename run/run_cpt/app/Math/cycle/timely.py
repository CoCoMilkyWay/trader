import math
from datetime import datetime
class timely:
    def __init__(self,
                 timestamp: float,
                 ):
        self.timestamp = timestamp
        
    def update(self):
        dt = datetime.fromtimestamp(self.timestamp)
        self.month_of_year = dt.month       # Month of the year (1-12)
        self.day_of_week = dt.weekday()     # Day of the week (0=Monday, 6=Sunday)
        self.hour_of_day = dt.hour          # Hour of the day (0-23)
        return