from datetime import datetime, timedelta
from typing import Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import random

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
    
    def _get_timeframe(self, dt: datetime) -> TimeFrame:
        """Determine appropriate timeframe for a datetime using clear range checks."""
        time_diff = self.reference_time - dt

        # Check ranges from smallest to largest
        if time_diff < timedelta(minutes=5):
            return TimeFrame.IMMEDIATE
        elif time_diff < timedelta(hours=1): # 5 min <= diff < 1 hour
            return TimeFrame.RECENT
        elif time_diff < timedelta(days=1): # 1 hour <= diff < 1 day
            return TimeFrame.TODAY
        elif time_diff < timedelta(days=2): # 1 day <= diff < 2 days
            return TimeFrame.YESTERDAY
        elif time_diff < timedelta(days=7): # 2 days <= diff < 1 week
            return TimeFrame.THIS_WEEK
        elif time_diff < timedelta(days=30): # 1 week <= diff < 1 month
            return TimeFrame.THIS_MONTH
        else: # diff >= 30 days
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
            dt: DateTime object or ISO format string
            
        Returns:
            TemporalExpression with base expression and optional time context
        """
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
            except ValueError:
                return TemporalExpression("at an unknown time")
        
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
        else:
            if time_diff.days >= 365:
                years = time_diff.days / 365
                # Round to nearest integer to avoid incorrect year display
                rounded_years = round(years)
                expression = pattern.format(rounded_years) if "{}" in pattern else pattern
            elif time_diff.days > 30:
                expression = pattern.format(time_diff.days // 30) if "{}" in pattern else pattern
            else:
                expression = pattern
        
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