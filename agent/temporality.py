from datetime import datetime, timedelta
from typing import Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import random
import re

class TimeFrame(Enum):
    IMMEDIATE = "immediate"     # < 5 minutes
    RECENT = "recent"          # < 1 hour
    TODAY = "today"           # < 24 hours
    YESTERDAY = "yesterday"    # < 48 hours
    THIS_WEEK = "this_week"    # < 7 days
    THIS_MONTH = "this_month"  # < 30 days
    OLDER = "older"           # > 30 days

@dataclass
class TemporalExpression:
    base_expression: str
    time_context: Optional[str] = None

class TemporalParser:
    """
    Converts datetime objects into natural language temporal expressions.
    Focuses on relative time descriptions with optional contextual time-of-day information.
    """
    
    TIME_BRACKETS = {
        timedelta(minutes=5): TimeFrame.IMMEDIATE,
        timedelta(hours=1): TimeFrame.RECENT,
        timedelta(days=1): TimeFrame.TODAY,
        timedelta(days=2): TimeFrame.YESTERDAY,
        timedelta(days=7): TimeFrame.THIS_WEEK,
        timedelta(days=30): TimeFrame.THIS_MONTH
    }
    
    TIME_EXPRESSIONS = {
        TimeFrame.IMMEDIATE: [
            "just now",
            "moments ago",
            "right now",
        ],
        TimeFrame.RECENT: [
            "{} minutes ago",
            "less than an hour ago",
        ],
        TimeFrame.TODAY: [
            "{} hours ago",
            "earlier today",
        ],
        TimeFrame.YESTERDAY: [
            "yesterday",
            "a day ago",
        ],
        TimeFrame.THIS_WEEK: [
            "{} days ago",
            "earlier this week",
        ],
        TimeFrame.THIS_MONTH: [
            "{} weeks ago",
            "a few weeks back",
        ],
        TimeFrame.OLDER: [
            "{} months ago",
            "{} years ago",
            "long ago"
        ]
    }
    
    TIME_CONTEXTS = {
        "early_morning": (5, 8),
        "morning": (8, 12),
        "afternoon": (12, 17),
        "evening": (17, 22),
        "night": (22, 5)
    }
    
    def __init__(self, reference_time: Optional[datetime] = None):
        """Initialize with optional reference time"""
        self.reference_time = reference_time or datetime.now()
    
    def _parse_timestamp(self, timestamp: str) -> Optional[datetime]:
        """Parse timestamp in format HH:MM [DD/MM/YY] or (HH:MM [DD/MM/YY])"""
        try:
            if isinstance(timestamp, datetime):
                return timestamp
                
            if isinstance(timestamp, str):
                # Remove any parentheses that might be present
                timestamp = timestamp.strip('()')
                
                # Try direct format first (from currentmoment())
                try:
                    return datetime.strptime(timestamp, "%H:%M [%d/%m/%y]")
                except ValueError:
                    pass
                    
                # Try with flexible whitespace (from memory text)
                match = re.match(r'(\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]', timestamp)
                if match:
                    hour, minute, date = match.groups()
                    try:
                        return datetime.strptime(f"{hour}:{minute} {date}", "%H:%M %d/%m/%y")
                    except ValueError:
                        pass
            return None
        except (ValueError, TypeError):
            return None
    
    def _get_timeframe(self, dt: datetime) -> TimeFrame:
        """Determine appropriate timeframe for a datetime using clear range checks."""
        time_diff = self.reference_time - dt

        # Check ranges from smallest to largest
        if time_diff < timedelta(minutes=5):
            return TimeFrame.IMMEDIATE
        elif time_diff < timedelta(hours=1):
            return TimeFrame.RECENT
        elif time_diff < timedelta(days=1):
            return TimeFrame.TODAY
        elif time_diff < timedelta(days=2):
            return TimeFrame.YESTERDAY
        elif time_diff < timedelta(days=7):
            return TimeFrame.THIS_WEEK
        elif time_diff < timedelta(days=30):
            return TimeFrame.THIS_MONTH
        else:
            return TimeFrame.OLDER
    
    def _get_time_context(self, dt: datetime) -> Optional[str]:
        """Get time-of-day context if within last 24 hours"""
        if self.reference_time - dt > timedelta(days=1):
            return None
            
        hour = dt.hour
        for context, (start, end) in self.TIME_CONTEXTS.items():
            if start <= hour < end or (context == "night" and (hour >= start or hour < end)):
                return context
        return None
    
    def get_temporal_expression(self, dt: Union[str, datetime]) -> TemporalExpression:
        """
        Convert datetime to natural language temporal expression
        
        Args:
            dt: DateTime object or timestamp string in format "HH:MM [DD/MM/YY]"
            
        Returns:
            TemporalExpression with base expression and optional time context
        """
        if isinstance(dt, str):
            parsed_dt = self._parse_timestamp(dt)
            if not parsed_dt:
                return TemporalExpression("at an unknown time")
            dt = parsed_dt
        
        timeframe = self._get_timeframe(dt)
        time_diff = self.reference_time - dt
        
        # Select expression pattern
        patterns = self.TIME_EXPRESSIONS[timeframe]
        pattern = random.choice(patterns)
        
        # Generate base expression
        if timeframe == TimeFrame.IMMEDIATE:
            expression = pattern
        elif timeframe == TimeFrame.RECENT:
            minutes = int(time_diff.total_seconds() / 60)
            expression = pattern.format(minutes) if "{}" in pattern else pattern
        elif timeframe == TimeFrame.TODAY:
            hours = int(time_diff.total_seconds() / 3600)
            expression = pattern.format(hours) if "{}" in pattern else pattern
        elif timeframe in (TimeFrame.YESTERDAY, TimeFrame.THIS_WEEK):
            days = time_diff.days
            expression = pattern.format(days) if "{}" in pattern else pattern
        elif timeframe == TimeFrame.THIS_MONTH:
            weeks = time_diff.days // 7
            expression = pattern.format(weeks) if "{}" in pattern else pattern
        else:  # OLDER timeframe
            if time_diff.days >= 365:
                years = time_diff.days // 365
                expression = f"{years} years ago"
            else:
                months = max(1, time_diff.days // 30)
                expression = f"{months} months ago"
        
        # Add time context for recent timeframes
        time_context = self._get_time_context(dt)
        
        return TemporalExpression(expression, time_context)



'''# Usage Examples
parser = TemporalParser()

test_times = [
    datetime.now() - timedelta(minutes=2),
    datetime.now() - timedelta(minutes=30),
    datetime.now() - timedelta(hours=3),
    datetime.now() - timedelta(days=1),
    datetime.now() - timedelta(days=5),
    datetime.now() - timedelta(days=20),
    datetime.now() - timedelta(days=400)
]

for test_time in test_times:
    result = parser.get_temporal_expression(test_time)
    if result.time_context:
        print(f"{result.base_expression} ({result.time_context})")
    else:
        print(result.base_expression)'''