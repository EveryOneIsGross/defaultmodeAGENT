#!/bin/bash

# Function to clear screen
clear_screen() {
    clear
}

# Function to show main menu
show_menu() {
    clear_screen
    echo "Discord Bot Selection and Configuration"
    echo "------------------------------------"
    echo "Available Bots:"
    
    # List available bots from prompts directory
    if [ -d "agent/prompts" ]; then
        for dir in agent/prompts/*/; do
            if [ -d "$dir" ]; then
                basename=$(basename "$dir")
                # Skip directories starting with . or __
                if [[ ! "$basename" =~ ^\. ]] && [[ ! "$basename" =~ ^__ ]]; then
                    echo "  $basename"
                fi
            fi
        done
    fi
    echo ""
}

# Function to show API menu
show_api_menu() {
    local bot_name="$1"
    clear_screen
    echo "API Selection for $bot_name"
    echo "----------------------"
    echo "1. Ollama (Local)"
    echo "2. OpenAI"
    echo "3. Anthropic"
    echo "4. vLLM (Local)"
    echo "5. Gemini"
    echo "----------------------"
    echo "6. Back to Bot Selection"
    echo "7. Exit"
    echo ""
}

# Function to handle Ollama selection
handle_ollama() {
    local bot_name="$1"
    clear_screen
    echo "Selected: Ollama (via API)"
    echo ""
    
    read -p "Enter model name (or press Enter for default): " model
    
    if [ -z "$model" ]; then
        python agent/discord_bot.py --api ollama --bot-name "$bot_name"
    else
        python agent/discord_bot.py --api ollama --model "$model" --bot-name "$bot_name"
    fi
}

# Function to handle other API selections
handle_api() {
    local api_name="$1"
    local bot_name="$2"
    clear_screen
    echo "Selected: $api_name"
    read -p "Enter model name (or press Enter for default): " model
    
    if [ -z "$model" ]; then
        python agent/discord_bot.py --api "$api_name" --bot-name "$bot_name"
    else
        python agent/discord_bot.py --api "$api_name" --model "$model" --bot-name "$bot_name"
    fi
}

# Main function
main() {
    while true; do
        show_menu
        read -p "Enter bot name (or press Enter for default): " bot_name
        
        # Set default bot name if empty
        if [ -z "$bot_name" ]; then
            bot_name="default"
        fi
        
        # API selection loop
        while true; do
            show_api_menu "$bot_name"
            read -p "Enter your choice (1-7): " choice
            
            case $choice in
                1)
                    handle_ollama "$bot_name"
                    exit 0
                    ;;
                2)
                    handle_api "openai" "$bot_name"
                    exit 0
                    ;;
                3)
                    handle_api "anthropic" "$bot_name"
                    exit 0
                    ;;
                4)
                    handle_api "vllm" "$bot_name"
                    exit 0
                    ;;
                5)
                    handle_api "gemini" "$bot_name"
                    exit 0
                    ;;
                6)
                    # Go back to bot selection
                    break
                    ;;
                7)
                    echo "Exiting..."
                    exit 0
                    ;;
                *)
                    echo "Invalid choice. Please try again."
                    sleep 2
                    ;;
            esac
        done
    done
}

# Run main function
main 