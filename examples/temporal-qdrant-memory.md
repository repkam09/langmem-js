# Temporal + Qdrant Memory Integration

Integrating langmem-js into a Temporal-based ReAct agent using Qdrant for storage,
with no dependency on LangGraph or the LangGraph server.

## Architecture

```
MainConversationWorkflow (Temporal)
  ├── Activity: fetchRelevantMemories(query, userId)
  │     └── Qdrant vector search → injects into system prompt
  ├── [conversation turns stored in workflow state as short-term memory]
  └── startChild: BackgroundMemoryWorkflow (fire-and-forget)

BackgroundMemoryWorkflow (Temporal)
  └── Activity: extractAndStoreMemories(messages, userId)
        ├── createMemoryManager (langmem) → LLM extracts structured facts
        └── Qdrant upsert → persists memories for future sessions
```

## Why No LangGraph

langmem-js has two layers:

- **`createMemoryManager`** — pure LLM extraction, no storage dependency.
  Returns `Array<{ id: string, content: Record<string, unknown> }>`.
- **`createMemoryStoreManager`** — wraps the above with a `BaseStore` (LangGraph abstraction)
  for automatic load/persist. Requires LangGraph.

By using `createMemoryManager` directly and handling persistence ourselves,
we eliminate the LangGraph dependency entirely.

## Storage Design

Each extracted memory is stored as a Qdrant point in a dedicated collection
(e.g. `agent_memories`), separate from your RAG document collection.

```
Qdrant point
├── id      → memory_id (stable UUID from langmem — reused as Qdrant point ID)
├── vector  → embedding of JSON.stringify(content)
└── payload
      ├── memory_id   : string
      ├── user_id     : string
      ├── content     : Record<string, unknown>   ← structured memory data
      └── updated_at  : number                    ← unix ms, for filtering
```

Using the langmem-assigned UUID directly as the Qdrant point ID means upserts
naturally handle both creates and updates without extra lookups.

## Memory Schemas

Pass JSON Schema objects (or Zod schemas) to `createMemoryManager` to define
what the LLM should extract:

```typescript
const memoryManager = createMemoryManager(model, {
  schemas: {
    user_preference: {
      type: "object",
      description: "User preferences, habits, and stated desires",
      properties: { fact: { type: "string" } },
    },
    episodic: {
      type: "object",
      description: "Key events, decisions, and facts from past conversations",
      properties: {
        event:   { type: "string" },
        context: { type: "string" },
      },
    },
  },
});
```

Add or remove schema types to control what kinds of memory are extracted.

## Extraction Activity

```typescript
export async function extractAndStoreMemories(
  messages: { role: "human" | "ai"; content: string }[],
  userId: string
): Promise<void> {
  // 1. Load existing memories so the LLM can update rather than duplicate
  const existing = await qdrantClient.scroll(COLLECTION, {
    filter: { must: [{ key: "user_id", match: { value: userId } }] },
    with_payload: true,
    with_vector: false,
    limit: 200,
  });

  const existingForManager = existing.points.map(p => [
    p.payload!.memory_id as string,
    p.payload!.content as Record<string, unknown>,
  ] as [string, Record<string, unknown>]);

  // 2. Run LLM extraction
  const langchainMessages = messages.map(m =>
    m.role === "human" ? new HumanMessage(m.content) : new AIMessage(m.content)
  );

  const extracted = await memoryManager.invoke({
    messages: langchainMessages,
    existing: existingForManager,
  });

  // 3. Embed and upsert into Qdrant
  const points = await Promise.all(
    extracted.map(async mem => ({
      id: mem.id,
      vector: await embedText(JSON.stringify(mem.content)),
      payload: {
        memory_id: mem.id,
        user_id: userId,
        content: mem.content,
        updated_at: Date.now(),
      },
    }))
  );

  if (points.length > 0) {
    await qdrantClient.upsert(COLLECTION, { points });
  }
}
```

## Retrieval Activity

```typescript
export async function fetchRelevantMemories(
  query: string,
  userId: string,
  limit = 5
): Promise<string[]> {
  const results = await qdrantClient.search(COLLECTION, {
    vector: await embedText(query),
    limit,
    filter: { must: [{ key: "user_id", match: { value: userId } }] },
    with_payload: true,
  });

  return results.map(r => JSON.stringify(r.payload!.content));
}
```

## Main Workflow Usage

```typescript
// Fetch memories and inject into system prompt
const memories = await activities.fetchRelevantMemories(latestUserMessage, userId);

const systemPrompt = memories.length > 0
  ? `${basePrompt}\n\nRelevant memories:\n${memories.map(m => `- ${m}`).join("\n")}`
  : basePrompt;

// After responding, trigger background extraction (fire-and-forget)
await startChild(backgroundMemoryWorkflow, {
  args: [serializedMessages, userId],
  workflowId: `memory-${userId}-${workflowInfo().runId}`,
  parentClosePolicy: ParentClosePolicy.ABANDON,
});
```

## Notes on Deletes

`manager.invoke()` with `existing` populated returns updated memory objects but
does not explicitly signal deletions — memories the LLM considers obsolete are
simply omitted from the result. If you need true delete support, diff the
returned IDs against the existing IDs after extraction:

```typescript
const returnedIds = new Set(extracted.map(m => m.id));
const toDelete = existing.points
  .map(p => p.id as string)
  .filter(id => !returnedIds.has(id));

if (toDelete.length > 0) {
  await qdrantClient.delete(COLLECTION, { points: toDelete });
}
```

For most use cases, upsert-only is sufficient.
