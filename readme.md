defaultMODE is a Discord framework for running persistent, self-aware agents. Unlike traditional Discord bots, our defaultMODE framework maintains a continuously updating memory index—similar to the human brain's Default Mode Network—which allows the agents to generate reflective background thoughts and maintain context over long periods. Continually compressing, expanding and pruning to maintain a homeostatic representation of their own sense of self.

The framework integrates with several language model APIs (including OpenAI, Anthropic, Ollama, and vLLM) to process both text and image inputs. It also enables multi-agent interactions through Discord's mention system, allowing agents to communicate, collaborate, and scheme together. This design simulates a collective consciousness where agents not only retain memories and conversational context but also engage in ongoing, dynamic interactions, much like human thought processes and social exchanges. 

<div align="center">

![alt text](/docs/assests/image.png)

</div>

## Core Features

- **Memory System**
  - Inverted index for lightweight semantic memory storage
  - Adaptive memory pruning based on relevance weights
  - Background thought generation via DMN processor
  - Temporal context parsing for natural time expressions
  - At inference hypocampus reranking using embeddings 

- **LLM Integration**
  - Multi-provider support (OpenAI, Anthropic, Ollama, vLLM)
  - Embedding model support (OpenAI, Ollama, vLLM)
  - Text and image processing capabilities
  - Dynamic temperature control via amygdala response (0-100)
  - File and GitHub repository processing

- **Discord Embodiment**
  - Context-aware message processing
  - Mention handling (ID ↔ username conversion)
  - Dynamic user/agent tagging
  - Multi-agent interaction support

The system maintains context through an inverted index rather than embeddings, using memory weights and n-gram comparisons for semantic relationships. The DMN processor generates background thoughts and adjusts response temperature based on context density, while the amygdala system modulates response creativity from deterministic (0) to highly dynamic (100).

```mermaid
graph TB
    subgraph Environment["Discord Interface"]
        DM[DM Channel] & MC[Message Channel] & FI[File Input] -->|Event| AT[Attention]
    end

    subgraph Processing["Core Processing"]
        AT -->|Parse| CP[Context Processing]
        CP -->|Build| WC[Working Context]
        
        subgraph HPC["Hippocampal Formation"]
            WC -->|Index| MI[Memory Index]
            MI -->|Associate| II[Inverted Index]
            II -->|Weight| MW[Memory Weights]
        end
        
        subgraph AMG["Amygdala Complex"]
            MD[Memory Density] -->|Calculate| AR[Arousal Response]
            AR -->|Regulate| TC[Temperature Control]
        end
        
        subgraph DMN["Default Mode Network"]
            direction TB
            RW[Random Walk] -->|Select| SM[Seed Memory]
            SM -->|Search| RM[Related Memories]
            RM -->|Analyze| MD
            
            RM -->|Extract| TO[Term Overlap]
            TO -->|Prune| UP[Update Patterns]
            UP -->|Strengthen| MI
        end
    end

    subgraph Integration["Temporal Integration"]
        TP[Temporal Parser] -->|Format| TC[Time Context]
        TC -->|Enrich| WC
    end

    subgraph Response["Output Generation"]
        WC & AR -->|Generate| RG[Response Generation]
        RG -->|Format| DR[Discord Response]
    end

    %% Feedback Loops
    DR -.->|Store| MI
    AR -.->|Modulate| RG
    MW -.->|Guide| RW
    MD -.->|Update| AR

    classDef env fill:#f0f0f0,stroke:#333,stroke-width:2px
    classDef hpc fill:#9f9,stroke:#333,stroke-width:4px
    classDef amg fill:#9cf,stroke:#333,stroke-width:4px
    classDef dmn fill:#f96,stroke:#333,stroke-width:4px
    classDef integration fill:#f9f,stroke:#333,stroke-width:2px
    classDef response fill:#ddd,stroke:#333,stroke-width:2px
    
    class DM,MC,FI,AT env
    class MI,II,MW hpc
    class MD,AR,TC amg
    class RW,SM,RM,TO,UP dmn
    class TP,TC integration
    class RG,DR response
```

## Installation and Usage

### Prerequisites

- Python 3.8 or later
- Required dependencies listed in `requirements.txt`
- Environment variables configured for:
  - Discord token(s)
  - API keys for the supported LLM providers
  - Other settings as outlined in the `.env.example` file

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-discord-agent.git
   cd your-discord-agent
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**  
   Create a `.env` file (or set the variables in your environment) based on the provided `.env.example`.

4. **Define your agents prompts**

    Point to or define your own agents prompts. These require specific variables to be injected into the prompts to allow the agents grok their current context. Shown in the provided agent examples.

    Bots should be defined in:

    `/agent/prompts/yr_agents_name/system_prompts.yaml`
    `/agent/prompts/yr_agents_name/prompt_formats.yaml`

    The system prompts should include the `{string}` variables that will be injected into the prompts. For more information look up the included DOCS on prompting. It will work without but you'll just get a goldfish. 🐟

### Running the Bot

You can start the bot by running:

```bash
python your_agent_script.py --api ollama --model hermes3 --bot-name YourBotName
```

Replace the arguments with your preferred API and bot name as needed.


```mermaid
flowchart TD
    subgraph External["External Interfaces"]
        DISCORD["Discord Interface"]
        APIs["Multiple LLM APIs"]
        GH["GitHub Integration"]
    end

    subgraph Memory["Memory System"]
        MI["Memory Index"]
        II["Inverted Index"]
        UM["User Memories"]
        MW["Memory Weights"]
        style MI fill:#f9f,stroke:#333,stroke-width:2px
    end

    subgraph DMN["Default Mode Network"]
        DMNProc["DMN Processor"]
        TP["Temporal Parser"]
        DC["Decay Controller"]
        PC["Amygdala"]
    end

    subgraph Processing["Core Processing"]
        CM["Context Manager"]
        PR["Prompt Router"]
        TC["Temperature Control"]
    end

    %% Main message flow
    DISCORD -->|"Incoming Messages"| CM
    CM -->|"Extract Context"| PR
    PR -->|"Route Request"| APIs
    APIs -->|"Generate Response"| CM
    CM -->|"Format Response"| DISCORD

    %% Memory operations
    CM -->|"Store Memory"| MI
    MI <-->|"Index Terms"| II
    MI <-->|"User Association"| UM
    
    %% DMN operations
    DMNProc -->|"Random Walk"| MI
    DMNProc -->|"Update Weights"| MW
    DMNProc -->|"Generate Insights"| APIs
    TP -->|"Time Context"| DMNProc
    DC -->|"Decay Rates"| MW
    PC -->|"Adjust Temperature"| TC
    
    %% GitHub integration
    GH -->|"Repository Content"| MI
    
    %% Bidirectional flows
    MI <-->|"Search & Retrieve"| PR
    TC <-->|"Dynamic Adjustment"| APIs
    
    %% State influence
    MW -->|"Influence"| DMNProc
    PC -->|"Persona State"| PR

    classDef primary fill:#f9f,stroke:#333,stroke-width:2px
    classDef secondary fill:#bbf,stroke:#333,stroke-width:1px
    classDef external fill:#ddd,stroke:#333,stroke-width:1px
    
    class DMNProc,MI primary
    class CM,PR,TC secondary
    class DISCORD,APIs,GH external
```
```mermaid
flowchart TD
    subgraph DMN["Default Mode Network"]
        direction LR
        SRP[Self-Referential Processing]
        AM[Autobiographical Memory]
        SP[Social Processing]
        MT[Mental Time Travel]
    end

    subgraph AMG["Amygdala Complex"]
        direction LR
        ES[Emotional Salience]
        AR[Arousal Regulation]
        ST[Stimulus Tagging]
    end

    subgraph HPC["Hippocampal Formation"]
        direction LR
        MC[Memory Consolidation]
        CP[Context Processing]
    end

    subgraph REG["Regulatory Systems"]
        direction LR
        ER[Emotional Regulation]
        HB[Homeostatic Balance]
    end

    %% Core Interactions
    AMG -->|Salience Signals| DMN
    DMN -->|Memory Activation| HPC
    HPC -->|Context Integration| AMG

    %% Regulatory Pathways
    ES -->|Emotional Weight| AM
    AR -->|Arousal State| SRP
    ST -->|Memory Tags| MC
    
    %% Feedback Loops
    SP -->|Social Context| ES
    MT -->|Temporal Context| CP
    MC -->|Memory State| AR
    CP -->|Contextual State| ER
    
    %% Homeostatic Control
    ER -->|Regulation| AR
    HB -->|Balance| DMN

    classDef dmn fill:#f96,stroke:#333,stroke-width:4px
    classDef amg fill:#9cf,stroke:#333,stroke-width:4px
    classDef reg fill:#f9f,stroke:#333,stroke-width:2px
    classDef mem fill:#9f9,stroke:#333,stroke-width:2px

    class SRP,AM,SP,MT dmn
    class ES,AR,ST amg
    class ER,HB reg
    class MC,CP mem
```