## UPDATES:

14/08/2025

Refactored `api_client` to include both `google-genai` and `openrouter`, the api now includes stateful and stateless operations to allow different models to be used with the front facing and remembering model and the backend DMN model. So you can have your more expensive front facing cloud api calls for the discord layer, and the "dreaming" can run at higher tics on a local or cheaper api. 🧠

Attention no includes sparsegrams derived from the agents memories as well as keeping the defined attention triggers. This allows the bot to track themes from their memories without an explicit look up. Memory based attention, now allows the DMN to prioritise most active users first, which will stop churn on interactions that have fewer datapoints. Allowing pruning or updating on those who interact with the bot most often. 🔍

Subtle prompt adjustments have been made to the template bots to address "who this?" confusion... these need tuning based on your model. Avoid using reasoning models with the framework, they will work, but I don't intend to manage "thinking" as a seperate process, thinking is done in reflection, the defined prompt is the planning step. My experiencec is putting your "thinking" tokens into the memory phase means recalled relevance inherits reasoning, which is imo more natural to the human experience and cognitive model. 

None of these changes distrupt the legacy memories, so existing precious pickles remain unaffected. 🥒

---
24/06/2025

With the addition of `attention_triggers` the prompts all need new consideration to help nurse the bilateral user/bot paradigm we are all familair with to be more multilateral. If an attention trigger has the agent respond to a user, engaging with another or bot with bot it will over focus on responding DIRECTLY without response context... this can be solved via some more selective prompting. 

I also added `top_p` to be dynamically adjusted with the `amygdala` response... its clamped in the bot_config but is a bit wild and can cause some OS models to meltdown... token selection might be a fun variable to maintain a degree of "interesting" outputs overtime, but I might also just remove it or scale it seperately...

---