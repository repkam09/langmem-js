# LangMem-js

## Note: This is a Claude Code port of the [LangMem](https://langchain-ai.github.io/langmem/) Python library to TypeScript. Check out the original for more complete documentation and examples.

LangMem helps agents learn and adapt from their interactions over time.

It provides tooling to extract important information from conversations, optimize agent behavior through prompt refinement, and maintain long-term memory.

It offers both functional primitives you can use with any storage system and native integration with LangGraph's storage layer.

This lets your agents continuously improve, personalize their responses, and maintain consistent behavior across sessions.

## Key features

- 🧩 **Core memory API** that works with any storage system
- 🧠 **Memory management tools** that agents can use to record and search information during active conversations "in the hot path"
- ⚙️ **Background memory manager** that automatically extracts, consolidates, and updates agent knowledge
- ⚡ **Native integration with LangGraph's Long-term Memory Store**, available by default in all LangGraph Platform deployments

## Installation

```bash
npm install langmem-js @langchain/langgraph @langchain/core
```

Configure your environment with an API key for your favorite LLM provider:

```bash
export ANTHROPIC_API_KEY="sk-..."  # Or another supported LLM provider
```

## Creating an Agent

Here's how to create an agent that actively manages its own long-term memory in just a few lines:

```typescript
// Import core components (1)
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { InMemoryStore } from "@langchain/langgraph";
import { createManageMemoryTool, createSearchMemoryTool } from "langmem-js";
import { ChatAnthropic } from "@langchain/anthropic";

// Set up storage (2)
const store = new InMemoryStore({
  index: {
    dims: 1536,
    embed: "openai:text-embedding-3-small",
  }
});

const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
  temperature: 0
});

// Create an agent with memory capabilities (3)
const agent = createReactAgent({
  llm: model,
  tools: [
    // Memory tools use LangGraph's BaseStore for persistence (4)
    createManageMemoryTool({ namespace: ["memories"] }),
    createSearchMemoryTool({ namespace: ["memories"] }),
  ],
  store,
});
```

1. The memory tools work in any LangGraph app. Here we use [`createReactAgent`](https://langchain-ai.github.io/langgraphjs/reference/modules/prebuilt.html#createReactAgent) to run an LLM with tools, but you can add these tools to your existing agents or build custom memory systems without agents.

2. [`InMemoryStore`](https://langchain-ai.github.io/langgraphjs/reference/classes/index.InMemoryStore.html) keeps memories in process memory—they'll be lost on restart.

3. The memory tools ([`createManageMemoryTool`](reference/tools.md#langmem.create_manage_memory_tool) and [`createSearchMemoryTool`](reference/tools.md#langmem.create_search_memory_tool)) let you control what gets stored.

Then use the agent:

```typescript
// Store a new memory (1)
await agent.invoke(
  { messages: [{ role: "user", content: "Remember that I prefer dark mode." }] },
  { configurable: { thread_id: "1" } }
);

// Retrieve the stored memory (2)
const response = await agent.invoke(
  { messages: [{ role: "user", content: "What are my lighting preferences?" }] },
  { configurable: { thread_id: "2" } }
);
console.log(response.messages[response.messages.length - 1].content);
// Output: "You've told me that you prefer dark mode."
```

1. The agent gets to decide what and when to store the memory. No special commands needed—just chat normally and the agent uses [`create_manage_memory_tool`](reference/tools.md#langmem.create_manage_memory_tool) to store relevant details.

2. The agent maintains context between chats. When you ask about previous interactions, the LLM can invoke [`create_search_memory_tool`](reference/tools.md#langmem.create_search_memory_tool) to search for memories with similar content. See [Memory Tools](guides/memory_tools.md) to customize memory storage and retrieval, and see the [hot path quickstart](https://langchain-ai.github.io/langmem/hot_path_quickstart) for a more complete example on how to include memories without the agent having to explicitly search.

The agent can now store important information from conversations, search its memory when relevant, and persist knowledge across conversations.

> [!TIP]
> For developing, debugging, and deploying AI agents and LLM applications, see [LangSmith](https://docs.langchain.com/langsmith/home).

## Next Steps

For more examples and detailed documentation:

- [Hot Path Quickstart](https://langchain-ai.github.io/langmem/hot_path_quickstart) - Learn how to let your LangGraph agent manage its own memory "in the hot path"
- [Background Quickstart](https://langchain-ai.github.io/langmem/background_quickstart) - Learn how to use a memory manager "in the background"
- [Core Concepts](https://langchain-ai.github.io/langmem/concepts/conceptual_guide) - Learn key ideas
- [API Reference](https://langchain-ai.github.io/langmem/reference) - Full function documentation
- Build RSI 🙂
