#!/usr/bin/env python3
"""
Accurate attention trigger test script using actual attention.py and YAML resources.
Mirrors the exact logic from your attention system with detailed scoring breakdown.
"""

import sys
import os
import yaml
import logging
from fuzzywuzzy import fuzz

# Import your actual attention function
from agent.attention import check_attention_triggers_fuzzy

# Set up logging to capture debug output
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def load_system_prompts(prompt_path):
    """Load system prompts from yaml file."""
    system_prompts_file = os.path.join(prompt_path, 'system_prompts.yaml')
    with open(system_prompts_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def list_available_agents():
    """List available agent prompt directories."""
    prompts_dir = os.path.join('agent', 'prompts')
    if not os.path.exists(prompts_dir):
        print(f"Error: {prompts_dir} not found")
        return []
    
    agents = []
    for item in os.listdir(prompts_dir):
        agent_path = os.path.join(prompts_dir, item)
        if os.path.isdir(agent_path):
            system_prompts_file = os.path.join(agent_path, 'system_prompts.yaml')
            if os.path.exists(system_prompts_file):
                agents.append(item)
    
    return sorted(agents)

def analyze_single_trigger_with_logging(content: str, trigger: str):
    """
    Test a single trigger and capture debug output from the actual attention function.
    """
    # Capture logs for this specific trigger test
    import io
    import contextlib
    
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger('agent.attention')
    logger.addHandler(handler)
    
    try:
        # Test just this one trigger using the actual function
        result = check_attention_triggers_fuzzy(content, [trigger])
        
        # Get the captured logs
        log_output = log_capture.getvalue()
        
        return {
            'triggered': result,
            'debug_output': log_output,
            'trigger': trigger
        }
    finally:
        logger.removeHandler(handler)

def get_word_count_limit():
    """Get the actual word count limit from attention.py by inspection."""
    import inspect
    source = inspect.getsource(check_attention_triggers_fuzzy)
    
    # Look for the word count check pattern
    lines = source.split('\n')
    for line in lines:
        if 'len(words)' in line and '<' in line:
            # Extract the number from "if len(words) < X:"
            try:
                parts = line.split('<')
                if len(parts) > 1:
                    number_part = parts[1].strip().rstrip(':')
                    return int(number_part)
            except:
                pass
    
    return 8  # fallback

def main():
    print("🎯 Accurate Attention Trigger Test Tool")
    print("Uses actual attention.py logic and YAML resources")
    print("=" * 60)
    
    # Select agent
    agents = list_available_agents()
    if not agents:
        print("No agent prompt directories found")
        sys.exit(1)
    
    print("Available agents:")
    for i, agent in enumerate(agents, 1):
        print(f"  {i:2d}. {agent}")
    
    try:
        choice = int(input("\nSelect agent (number): ")) - 1
        if choice < 0 or choice >= len(agents):
            print("Invalid choice")
            sys.exit(1)
    except ValueError:
        print("Invalid input")
        sys.exit(1)
    
    selected_agent = agents[choice]
    prompt_path = os.path.join('agent', 'prompts', selected_agent)
    
    try:
        system_prompts = load_system_prompts(prompt_path)
        triggers = system_prompts.get('attention_triggers', [])
        
        print(f"\n🤖 Testing {selected_agent} with {len(triggers)} triggers")
        print("=" * 60)
        
        print("Loaded triggers:")
        for i, trigger in enumerate(triggers, 1):
            if trigger:
                print(f"{i:2d}. '{trigger}'")
            else:
                print(f"{i:2d}. (empty trigger)")

        # Get actual word count limit from your script
        word_limit = get_word_count_limit()
        
        while True:
            message = input("\n📝 Test message (or 'quit'): ")
            if message.lower() == 'quit':
                break
            
            # Check word count using actual limit from your script
            words = message.lower().strip().split()
            print(f"\n📊 Input Analysis:")
            print(f"   Message: '{message}'")
            print(f"   Word count: {len(words)}")
            print(f"   Word limit: {'PASS' if len(words) >= word_limit else 'FAIL'} (need {word_limit}+)")
            
            # Call the actual function first to get the real result
            actual_result = check_attention_triggers_fuzzy(message, triggers)
            print(f"\n🤖 Actual function result: {'TRIGGERED' if actual_result else 'NO TRIGGER'}")
            
            if len(words) < word_limit:
                print("   ❌ Message too short - attention system returns False")
                print("=" * 60)
                continue
            
            print(f"\n🔍 Individual Trigger Testing:")
            print("-" * 60)
            
            # Test each trigger individually to see detailed results
            for i, trigger in enumerate(triggers, 1):
                if not trigger:
                    continue
                
                # Test this single trigger and capture its debug output
                result = analyze_single_trigger_with_logging(message, trigger)
                
                status = "🔴 MATCH" if result['triggered'] else "🟢 no match"
                print(f"{i:2d}. {status} '{trigger}'")
                
                # Show debug output from your actual function
                if result['debug_output'].strip():
                    debug_lines = result['debug_output'].strip().split('\n')
                    for line in debug_lines:
                        if line.strip():
                            print(f"    📝 {line}")
                else:
                    print("    (no debug output - likely failed early checks)")
                
                print()
            
            print("=" * 60)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()