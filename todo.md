FEAT. IDEAS

- create an instance of the bots that can do realtime speechtotext and then texttospeech for the voice channel
- editable notepad for the agent, note taking or editing step after reflection
- expand webscraper

CURRENT BUGS

- `process_files` is hairy, the loops are causing the typing indicator to momentarily retrigger on events post the main loop. SOMETIMES...
- run_bot.bat might need you to run "pip install cffi pynacl --force-reinstall"
- logging events need aligning and correcting where possible
- `agent/skeleton.py` defines the shape of the discord calls for abstracting this framework out to be platform agnostic
    + I have defined the `mind` but "platforms" are much like the "body" but just in different contexts.
    + So much like my relationships change when I am in a different environment, my body adapts and my mind will learn it's new bounds
- kb or user prior ingesting
    + a way of extracting and preformatting social media/email/archive
    + then ingesting them into the inverted index for the agent to grok
- the `user_name` is unsafe and being transformed by an async operation causing mentioned users to hijack the index and response causing a Freaky Friday event.
- `defaultmode.py` needs a context window limit... I'd prefer not to chunk or truncate if possible. only an issue on sub 32k context window limit llms.
    To ensure information is not lost in the 0 shot compression step. It's up to the agent to capture important details as trimming is aggressive.
    Because I don't use chunking by default and rely more on truncation and item n limits...
- `defaultmode.py` needs the hyperparam values moved to bot config from DMN class. Also preset modes need honouring per bot init. This needs thoughtful integration considerations. 

COMPLETED

0525
- long message splitting is not appending broken markdown wraps causing @mention transforms to occasionally break.
- handle main script termination with a graceful shutdown

0325
~~- image caching isn't using the temporty agent designated cache folders...~~ - done
- image handling automatically resize large images under a threshold value

0324
- #channelnames are handled with #102989861208 numerical IDS and also need special handling simialr to mention transforms between channels and agent *sigh*
    example mentoning one by name to the agent and it reproduces id instyead of name back to channel exposing probably the id instead of the name back afwd to the agent, needs mention sanitazation logic 
    ```ATTENTION SHIFT
    ------------
    Location: <#1360228315209666682>
    Status: [PRESERVED GAME]
    Mood: [APPRECIATIVE]
    ```
- temporality is something saying 2 years ago when that is impossible

040524
- enable toggling `hippocampus` ranking off or on
    + tune the hippocampus and amgdelya hyperparameter interactions for homeostatic equilibrium management of the in memory memory space for the agent
- DMs aren't handling the removal of '@' in direct chats. possible not calling the sanitize_mentions method
