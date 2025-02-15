
def list_timestamps(start: int, end: int, use_datetime:bool) -> list:
    from datetime import datetime, timedelta
    
    def parse_time(time: int) -> datetime:
        time_str = str(time)
        # Expecting a 12-digit integer: YYYYMMDDHHMM
        year   = int(time_str[-12:-8])
        month  = int(time_str[-8:-6])
        day    = int(time_str[-6:-4])
        hour   = int(time_str[-4:-2])
        minute = int(time_str[-2:])
        return datetime(year, month, day, hour, minute)

    def format_time(dt: datetime) -> int:
        # Format the datetime back into the 12-digit integer form
        # [202502130000, 202502130001, ...]
        return int(dt.strftime('%Y%m%d%H%M'))

    # Parse the input integers into datetime objects
    start_time = parse_time(start)
    end_time   = parse_time(end)

    # Generate timestamps for every minute from start_time to end_time (inclusive)
    current_time = start_time
    timestamps = []
    while current_time < end_time:
        if use_datetime:
            timestamps.append(current_time)
        else:
            timestamps.append(format_time(current_time))
        current_time += timedelta(minutes=1)
    
    return timestamps