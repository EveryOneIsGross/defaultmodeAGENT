@echo off
setlocal EnableDelayedExpansion

:menu
cls
echo Discord Bot Selection and Configuration
echo ------------------------------------
echo Available Bots:
dir /b /ad "agent\prompts" 2>nul | findstr /v /i "\..*" | findstr /v /i "__.*"
echo.
set /p bot_name="Enter bot name (or press Enter for default): "

if "!bot_name!"=="" set bot_name=default

:api_menu
cls
echo API Selection for !bot_name!
echo ----------------------
echo 1. Ollama (Local)
echo 2. OpenAI
echo 3. Anthropic
echo 4. vLLM (Local)
echo 5. Gemini
echo ----------------------
echo 6. Back to Bot Selection
echo 7. Exit
echo.

set /p choice="Enter your choice (1-7): "

if "!choice!"=="1" (
    cls
    echo Selected: Ollama
    echo.
    echo Available Ollama Models:
    echo ----------------------
    ollama list
    echo ----------------------
    echo.
    set /p model="Enter model name (or press Enter for default): "
    if "!model!"=="" (
        python agent/discord_bot.py --api ollama --bot-name !bot_name!
    ) else (
        python agent/discord_bot.py --api ollama --model !model! --bot-name !bot_name!
    )
    goto end
)

if "!choice!"=="2" (
    cls
    echo Selected: OpenAI
    set /p model="Enter model name (or press Enter for default): "
    if "!model!"=="" (
        python agent/discord_bot.py --api openai --bot-name !bot_name!
    ) else (
        python agent/discord_bot.py --api openai --model !model! --bot-name !bot_name!
    )
    goto end
)

if "!choice!"=="3" (
    cls
    echo Selected: Anthropic
    set /p model="Enter model name (or press Enter for default): "
    if "!model!"=="" (
        python agent/discord_bot.py --api anthropic --bot-name !bot_name!
    ) else (
        python agent/discord_bot.py --api anthropic --model !model! --bot-name !bot_name!
    )
    goto end
)

if "!choice!"=="4" (
    cls
    echo Selected: vLLM
    set /p model="Enter model name (or press Enter for default): "
    if "!model!"=="" (
        python agent/discord_bot.py --api vllm --bot-name !bot_name!
    ) else (
        python agent/discord_bot.py --api vllm --model !model! --bot-name !bot_name!
    )
    goto end
)

if "!choice!"=="5" (
    cls
    echo Selected: Gemini
    set /p model="Enter model name (or press Enter for default): "
    if "!model!"=="" (
        python agent/discord_bot.py --api gemini --bot-name !bot_name!
    ) else (
        python agent/discord_bot.py --api gemini --model !model! --bot-name !bot_name!
    )
    goto end
)

if "!choice!"=="6" (
    goto menu
)

if "!choice!"=="7" (
    echo Exiting...
    goto end
)

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto api_menu

:end
endlocal 