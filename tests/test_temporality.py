import unittest
from datetime import datetime, timedelta
from temporality import TemporalParser, TimeFrame, TemporalExpression
import argparse

class TestTemporality(unittest.TestCase):
    def setUp(self):
        # Use current time for testing to match real usage
        self.reference_time = datetime.now()
        self.parser = TemporalParser(reference_time=self.reference_time)

    def test_single_timestamp(self, timestamp):
        """Test a single timestamp string"""
        result = self.parser.get_temporal_expression(timestamp)
        
        # Parse the timestamp for comparison
        try:
            if isinstance(timestamp, str):
                timestamp = timestamp.strip('():')
                parsed_time = datetime.strptime(timestamp, "%H:%M [%d/%m/%y]")
            else:
                parsed_time = timestamp
                
            # Calculate actual time difference
            diff = self.reference_time - parsed_time
            days = diff.days
            hours = diff.total_seconds() / 3600
            minutes = diff.total_seconds() / 60
            
            print(f"\nTesting timestamp: {timestamp}")
            print(f"Time difference: {days} days, {int(hours % 24)} hours, {int(minutes % 60)} minutes")
            print(f"Result: {result.base_expression}")
            if result.time_context:
                print(f"Time context: {result.time_context}")
        except ValueError as e:
            print(f"\nTesting timestamp: {timestamp}")
            print(f"Error parsing timestamp: {e}")
            print(f"Result: {result.base_expression}")
            if result.time_context:
                print(f"Time context: {result.time_context}")
        
        return result
        
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
        test_time = now - timedelta(days=5)
        timestamp = test_time.strftime("%H:%M [%d/%m/%y]")
        result = self.parser.get_temporal_expression(timestamp)
        self.assertTrue(
            str(test_time.days) + " days ago" in result.base_expression or 
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
        # Test each time context with current date
        now = datetime.now()
        base_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        contexts = {
            "early_morning": 6,  # 6 AM
            "morning": 10,       # 10 AM
            "afternoon": 14,     # 2 PM
            "evening": 19,       # 7 PM
            "night": 23         # 11 PM
        }
        
        for expected_context, hour in contexts.items():
            test_time = base_time + timedelta(hours=hour)
            timestamp = test_time.strftime("%H:%M [%d/%m/%y]")
            result = self.parser.get_temporal_expression(timestamp)
            if abs((now - test_time).total_seconds()) < 24*3600:  # Only check context if within 24 hours
                self.assertEqual(result.time_context, expected_context,
                               f"Wrong context at {hour}:00, expected {expected_context}, got {result.time_context}")
        
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

def main():
    parser = argparse.ArgumentParser(description='Test temporal expression parsing')
    parser.add_argument('--timestamp', '-t', type=str, help='Test a specific timestamp string')
    parser.add_argument('--reference', '-r', type=str, help='Set reference time (format: YYYY-MM-DD HH:MM)')
    parser.add_argument('--run-tests', '-a', action='store_true', help='Run all unit tests')
    
    args = parser.parse_args()
    
    if args.timestamp:
        # Create test instance and initialize it
        test = TestTemporality()
        test.setUp()  # Initialize the parser
        
        # Set custom reference time if provided
        if args.reference:
            try:
                test.reference_time = datetime.strptime(args.reference, "%Y-%m-%d %H:%M")
                test.parser = TemporalParser(reference_time=test.reference_time)
                print(f"Using reference time: {test.reference_time}")
            except ValueError:
                print("Invalid reference time format. Use: YYYY-MM-DD HH:MM")
                return
        
        # Test the provided timestamp
        test.test_single_timestamp(args.timestamp)
    elif args.run_tests or not any(vars(args).values()):
        # Run all tests if --run-tests is specified or no args provided
        unittest.main(argv=['first-arg-is-ignored'])
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 