# Agent Definitions

./agent/prompts/

system_prompts.yaml

prompt_formats.yaml


The folder contains the individual assets for each agent, each agent has their own reasoning style and personality when interacting with the available tools. They use tailored prompts to grok the multi stream context, as well as {string} variables to inject the relevent data frome framework into the agents context.

Current cast of characters:

TARS

Delphi

loop

grossBOT

CASFY

default



# AGENT REQS

Each agent is made up of a set of contextual system prompts and prompt formats with a set of required {string} variables.

```
prompts/{name}/images/

# SOCIAL MEDIA ASSETS

1:1 PROFILE PIC
9:16 CHARACTER ANIMATION
680x240 DISCORD BANNER

```

```
prompts/{name}/character_sheet.md

prompts/{name}/system_prompts.md

    # Default chat system prompt
    `default_chat`

    # Default web chat system prompt
    `default_web_chat`

    # Ask a query about a single item from the repo
    `repo_file_chat`

    # Channel summarization system prompt
    `channel_summarization`

    # Repository analysis system prompt, for processing the multiple search results from a GitHub repository
    `ask_repo`

    # Thought generation system prompt vital component for building persistence of agents' sense of self
    `thought_generation`

    # Text file analysis system prompt
    `file_analysis`

    # Image analysis system prompt
    `image_analysis`

    # Combined analysis system prompt
    `combined_analysis`


prompts/{name}/prompt_formats.md


`chat_with_memory`

  {context}
  Current interaction:
  {user_name}: {user_message}

`introduction`

`introduction_web`

`analyze_code`

`summarize_channel`

`ask_repo`

`repo_file_chat`

`generate_thought`

# Prompt for managing included images
`analyze_image`

# Text File analysis prompt
`analyze_file`

# Combined analysis prompt when both and external file and image are included in the context.
`analyze_combined`

```
