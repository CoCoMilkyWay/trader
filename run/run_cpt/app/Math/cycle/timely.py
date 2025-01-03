from datetime import datetime
import array
class timely:
    def __init__(self,
                 timestamp: array.array,
                 ):
        self.timestamp = timestamp
        
    def update(self):
        dt = datetime.fromtimestamp(self.timestamp[-1])
        self.month_of_year = dt.month       # Month of the year (1-12)
        self.day_of_week = dt.weekday()     # Day of the week (0=Monday, 6=Sunday)
        self.hour_of_day = dt.hour          # Hour of the day (0-23)
        return