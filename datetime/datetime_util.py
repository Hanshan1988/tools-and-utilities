import calendar
from datetime import datetime, timedelta, date
from calendar import MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY

def get_first_xday_of_month(your_date, dow, first=True):
    # Get the first "day" of the month and the number of days in the month
    d_map = {1: MONDAY, 2: TUESDAY, 3: WEDNESDAY,
             4: THURSDAY, 5: FRIDAY, 6:SATURDAY, 7: SUNDAY}
    
    month_range = calendar.monthrange(your_date.year, your_date.month)

    if first: # First Xday of the month
        date_corrected = date(your_date.year, your_date.month, 1)
        delta = (d_map[dow] - month_range[0]) % 7
        return date_corrected + timedelta(days = delta)

    else: # Last Xday of the month
        date_corrected = date(your_date.year, your_date.month, month_range[1])
        delta = (your_date.weekday() - d_map[dow]) % 7
        return date_corrected - timedelta(days = delta) 