- temporality is something saying 2 years ago when that is impossible

- process image sendlong message typing indicator is triggering twice and not encapsulating the process correctly.

- long message splitting is not appending broken markdown wraps causing @mention transforms to occasionally break.

- DMs aren't handling the removal of '@' in direct chats. possible not calling the sanitize_mentions method

- DMs need renaming in the Model Context Prompt (lol).. j/k #MCPrip... no srsly tho. I need to rename the DM from discords abstraction to "Private Message" and test it to see if the agents silo memories by choice better between recalled plain text context without index enforcement. 🤔

- enable toggling `hippocampus` ranking off or on
    + tune the hippocampus and amgdelya hyperparameter interactions for homeostatic equilibrium management of the in memory memory space for the agent

- logging events need aligning and correcting where possible

- `discord_bot.py` def `action` scratch placeholder needs action/tool wrapper definition per agent
    + the idea being after interactions and reflections the bot generates an action using MCP or whatever other fashionable context-shapes, this tool if warrented or defined per agent will trigger and is defined via yaml a per agent action prompt
    this will enable a workflow with toolspace calling through sequential action without fucking with the rest of the flows abstractions. agents choose to act with tools defined or not, nbd.

- `agent/skeleton.py` defines the shape of the discord calls for abstracting this framework out to be platform agnostic
    + I have defined the `mind` but "platforms" are much like the "body" but just in different contexts.
    + So much like my relationships change when I am in a different environment, my body adapts and my mind will learn it's new bounds

- handle main script termination with a graceful shutdown

- `agent/memory.py` needs to be refactored to be more efficient and scalable after adding IDF weighting to the memory search

- kb or user prior ingesting
    + a way of extracting and preformatting social media/email/archive
    + then ingesting them into the inverted index for the agent to grok

- image handling automatically resize large images under a threshold value

- the `user_name` is unsafe and being transformed by an async operation causing mentioned users to hijack the index and response causing a Freaky Friday event.

- `defaultmode.py` needs a context window limit... I'd prefer not to chunk or truncate if possible. only an issue on sub 32k context window limit llms.
    To ensure information is not lost in the 0 shot compression step. It's up to the agent to capture important details as trimming is aggressive.
    Because I don't use chunking by default and rely more on truncation and item n limits...

- `defaultmode.py` needs the hyperparam values moved to bot config from DMN class. Also preset modes need honouring per bot init. This needs thoughtful integration considerations. 

COMPLETED

~~- image caching isn't using the temporty agent designated cache folders...~~ - done

- #channelnames are handled with #102989861208 numerical IDS and also need special handling simialr to mention transforms between channels and agent *sigh*
    example mentoning one by name to the agent and it reproduces id instyead of name back to channel exposing probably the id instead of the name back afwd to the agent, needs mention sanitazation logic 
    ```ATTENTION SHIFT
    ------------
    Location: <#1360228315209666682>
    Status: [PRESERVED GAME]
    Mood: [APPRECIATIVE]
    ```