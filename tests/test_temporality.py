import unittest
from datetime import datetime, timedelta
from agent.temporality import TemporalParser, TimeFrame, TemporalExpression
import argparse

class TestTemporality(unittest.TestCase):
    def setUp(self):
        # Use current time for testing to match real usage
        self.reference_time = datetime.now()
        self.parser = TemporalParser(reference_time=self.reference_time)

    def test_single_timestamp(self):
        """Test a single timestamp string and verify its temporal expression"""
        # Test a known timestamp
        now = self.reference_time
        test_time = now - timedelta(minutes=30)
        timestamp = test_time.strftime("%H:%M [%d/%m/%y]")
        
        result = self.parser.get_temporal_expression(timestamp)
        
        # Parse the timestamp for comparison
        try:
            parsed_time = datetime.strptime(timestamp.strip('():'), "%H:%M [%d/%m/%y]")
                
            # Calculate actual time difference
            diff = self.reference_time - parsed_time
            days = diff.days
            hours = diff.total_seconds() / 3600
            minutes = diff.total_seconds() / 60
            
            # Add assertions based on time difference
            if minutes < 5:
                self.assertIn(result.base_expression, ["just now", "moments ago", "right now"])
            elif minutes < 60:
                self.assertTrue(
                    "minutes ago" in result.base_expression or 
                    result.base_expression == "less than an hour ago"
                )
            elif hours < 24:
                self.assertTrue(
                    "hours ago" in result.base_expression or 
                    result.base_expression == "earlier today"
                )
                
            # Log test details for debugging
            print(f"\nTesting timestamp: {timestamp}")
            print(f"Time difference: {days} days, {int(hours % 24)} hours, {int(minutes % 60)} minutes")
            print(f"Result: {result.base_expression}")
            if result.time_context:
                print(f"Time context: {result.time_context}")
                
        except ValueError as e:
            # For invalid timestamps, verify we get the expected error message
            self.assertEqual(
                result.base_expression,
                "at an unknown time",
                f"Invalid timestamp {timestamp} should return 'at an unknown time'"
            )
            print(f"\nTesting timestamp: {timestamp}")
            print(f"Error parsing timestamp: {e}")
        
    def test_timestamp_formats(self):
        """Test all timestamp formats used in discord_bot.py"""
        # Test with actual time differences
        now = datetime.now()
        
        # Test immediate (within 5 minutes)
        test_time = now - timedelta(minutes=2)
        timestamp = test_time.strftime("%H:%M [%d/%m/%y]")
        result = self.parser.get_temporal_expression(timestamp)
        self.assertIn(result.base_expression, ["just now", "moments ago", "right now"])
        
        # Test recent (within hour)
        test_time = now - timedelta(minutes=30)
        timestamp = test_time.strftime("%H:%M [%d/%m/%y]")
        result = self.parser.get_temporal_expression(timestamp)
        self.assertTrue(
            "minutes ago" in result.base_expression or 
            result.base_expression == "less than an hour ago"
        )
        
        # Test today
        if now.hour > 3:  # Only test if we have enough hours in the day
            test_time = now - timedelta(hours=3)
            timestamp = test_time.strftime("%H:%M [%d/%m/%y]")
            result = self.parser.get_temporal_expression(timestamp)
            self.assertTrue(
                "hours ago" in result.base_expression or 
                result.base_expression == "earlier today"
            )
        
        # Test yesterday
        test_time = now - timedelta(days=1)
        timestamp = test_time.strftime("%H:%M [%d/%m/%y]")
        result = self.parser.get_temporal_expression(timestamp)
        self.assertIn(result.base_expression, ["yesterday", "a day ago"])
        
        # Test this week
        diff_days = 5
        test_time = now - timedelta(days=diff_days)
        timestamp = test_time.strftime("%H:%M [%d/%m/%y]")
        result = self.parser.get_temporal_expression(timestamp)
        self.assertTrue(
            f"{diff_days} days ago" in result.base_expression or 
            result.base_expression == "earlier this week"
        )
        
        # Test this month
        test_time = now - timedelta(days=20)
        timestamp = test_time.strftime("%H:%M [%d/%m/%y]")
        result = self.parser.get_temporal_expression(timestamp)
        self.assertTrue(
            "weeks ago" in result.base_expression or 
            result.base_expression == "a few weeks back"
        )
        
    def test_time_context(self):
        """Test time context generation for different hours of the day"""
        # Test each time context with current date
        now = datetime.now()
        base_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Match the exact time ranges from TemporalParser.TIME_CONTEXTS
        contexts = {
            "early_morning": 6,  # Within 5-8
            "morning": 10,       # Within 8-12
            "afternoon": 14,     # Within 12-17
            "evening": 19,       # Within 17-22
            "night": 23         # Within 22-5
        }
        
        for expected_context, hour in contexts.items():
            test_time = base_time + timedelta(hours=hour)
            timestamp = test_time.strftime("%H:%M [%d/%m/%y]")
            result = self.parser.get_temporal_expression(timestamp)
            
            # Only verify context if within last 24 hours
            if abs((now - test_time).total_seconds()) < 24*3600:
                self.assertEqual(
                    result.time_context, 
                    expected_context,
                    f"Wrong context at {hour}:00, expected {expected_context}, got {result.time_context}"
                )
                
            # Verify time context is None for times > 24 hours ago
            test_time_old = test_time - timedelta(days=2)
            timestamp_old = test_time_old.strftime("%H:%M [%d/%m/%y]")
            result_old = self.parser.get_temporal_expression(timestamp_old)
            self.assertIsNone(
                result_old.time_context,
                f"Expected no time context for time > 24h ago, got {result_old.time_context}"
            )
        
    def test_invalid_timestamp(self):
        invalid_timestamps = [
            "invalid",
            "25:00 [01/01/24]",    # Invalid hour
            "14:60 [01/01/24]",    # Invalid minute
            "14:30 [32/01/24]",    # Invalid day
            "14:30 [01/13/24]",    # Invalid month
            "14:30 01/01/24",      # Missing brackets
            "[01/01/24]",          # Missing time
            "14:30",               # Missing date
            "14:30 [01/01]",       # Incomplete date
            "14:30 [//24]",        # Missing day/month
            "14:30 [01/01/2024]",  # Full year instead of YY
        ]
        for timestamp in invalid_timestamps:
            result = self.parser.get_temporal_expression(timestamp)
            self.assertEqual(
                result.base_expression, 
                "at an unknown time",
                f"Expected 'at an unknown time' for invalid timestamp: {timestamp}"
            )

    def test_older_timeframes(self):
        """Test expressions for timestamps months and years ago"""
        now = self.reference_time
        
        # Test months ago (31-364 days)
        test_time = now - timedelta(days=90)
        timestamp = test_time.strftime("%H:%M [%d/%m/%y]")
        result = self.parser.get_temporal_expression(timestamp)
        self.assertTrue(
            "months ago" in result.base_expression,
            f"Expected 'months ago' in expression for 90 days ago, got: {result.base_expression}"
        )
        
        # Test years ago (365+ days)
        test_time = now - timedelta(days=400)
        timestamp = test_time.strftime("%H:%M [%d/%m/%y]")
        result = self.parser.get_temporal_expression(timestamp)
        self.assertTrue(
            "years ago" in result.base_expression,
            f"Expected 'years ago' in expression for 400 days ago, got: {result.base_expression}"
        )
        
        # Verify no time context for old timestamps
        self.assertIsNone(result.time_context, "Expected no time context for timestamps over 24h old")

def main():
    parser = argparse.ArgumentParser(description='Test temporal expression parsing')
    parser.add_argument('--timestamp', '-t', type=str, help='Test a specific timestamp string')
    parser.add_argument('--reference', '-r', type=str, help='Set reference time (format: YYYY-MM-DD HH:MM)')
    parser.add_argument('--run-tests', '-a', action='store_true', help='Run all unit tests')
    
    args = parser.parse_args()
    
    if args.timestamp:
        # Create parser instance
        reference_time = datetime.now()
        
        # Set custom reference time if provided
        if args.reference:
            try:
                reference_time = datetime.strptime(args.reference, "%Y-%m-%d %H:%M")
                print(f"Using reference time: {reference_time}")
            except ValueError:
                print("Invalid reference time format. Use: YYYY-MM-DD HH:MM")
                return
                
        parser = TemporalParser(reference_time=reference_time)
        
        # Parse the timestamp
        result = parser.get_temporal_expression(args.timestamp)
        print(f"\nResult for timestamp {args.timestamp}:")
        print(f"Expression: {result.base_expression}")
        if result.time_context:
            print(f"Time context: {result.time_context}")
            
    elif args.run_tests or not any(vars(args).values()):
        # Run all tests if --run-tests is specified or no args provided
        unittest.main(argv=['first-arg-is-ignored'])
    else:
        parser.print_help()

if __name__ == '__main__':
    main()