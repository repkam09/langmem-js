/**
 * Temporal + Qdrant Memory Integration
 * =====================================
 *
 * Demonstrates how to use langmem-js memory extraction with Qdrant storage
 * inside Temporal activities — no LangGraph dependency.
 *
 * This file is illustrative: `qdrantClient` and `embedText` are placeholders
 * for your existing Qdrant/embedding setup.
 *
 * See temporal-qdrant-memory.md for full architecture notes.
 */

import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { createMemoryManager } from "../src/index.js";

// ---------------------------------------------------------------------------
// Placeholders — replace with your existing Qdrant / embedding setup
// ---------------------------------------------------------------------------

declare const qdrantClient: {
  scroll(collection: string, opts: object): Promise<{ points: QdrantPoint[] }>;
  search(collection: string, opts: object): Promise<QdrantSearchResult[]>;
  upsert(collection: string, opts: object): Promise<void>;
  delete(collection: string, opts: object): Promise<void>;
};

declare function embedText(text: string): Promise<number[]>;

interface QdrantPoint {
  id: string;
  payload?: Record<string, unknown>;
}

interface QdrantSearchResult {
  id: string;
  score: number;
  payload?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const COLLECTION = "agent_memories";

const model = new ChatAnthropic({ model: "claude-sonnet-4-6" });

// Define what kinds of facts the LLM should extract from conversations.
// Add or remove schema types to suit your domain.
const memoryManager = createMemoryManager(model, {
  schemas: {
    user_preference: {
      type: "object",
      description: "User preferences, habits, and stated desires",
      properties: {
        fact: { type: "string" },
      },
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

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SerializedMessage {
  role: "human" | "ai";
  content: string;
}

interface MemoryPayload {
  memory_id: string;
  user_id: string;
  content: Record<string, unknown>;
  updated_at: number;
}

// ---------------------------------------------------------------------------
// Temporal Activities
// ---------------------------------------------------------------------------

/**
 * Background activity: extract memories from a conversation and upsert into Qdrant.
 *
 * Called from a child Temporal workflow after each conversation turn.
 * The langmem-assigned UUID is reused as the Qdrant point ID so upserts
 * handle both creates and updates without extra lookups.
 */
export async function extractAndStoreMemories(
  messages: SerializedMessage[],
  userId: string
): Promise<void> {
  // 1. Load existing memories so the LLM can update rather than duplicate them
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

  // 2. Run LLM-based extraction
  const langchainMessages = messages.map(m =>
    m.role === "human" ? new HumanMessage(m.content) : new AIMessage(m.content)
  );

  const extracted = await memoryManager.invoke({
    messages: langchainMessages,
    existing: existingForManager,
  });

  if (extracted.length === 0) return;

  // 3. Embed each memory's content and upsert into Qdrant
  const points = await Promise.all(
    extracted.map(async mem => ({
      id: mem.id,
      vector: await embedText(JSON.stringify(mem.content)),
      payload: {
        memory_id: mem.id,
        user_id: userId,
        content: mem.content,
        updated_at: Date.now(),
      } satisfies MemoryPayload,
    }))
  );

  await qdrantClient.upsert(COLLECTION, { points });

  // Optional: delete memories the LLM considers obsolete (omitted from result)
  const returnedIds = new Set(extracted.map(m => m.id));
  const toDelete = existing.points
    .map(p => p.id)
    .filter(id => !returnedIds.has(id));

  if (toDelete.length > 0) {
    await qdrantClient.delete(COLLECTION, { points: toDelete });
  }
}

/**
 * Main workflow activity: fetch memories relevant to the current user message.
 *
 * Call this at the start of each conversation turn and inject the results
 * into the agent's system prompt.
 */
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

  return results.map(r => JSON.stringify((r.payload as MemoryPayload).content));
}

// ---------------------------------------------------------------------------
// Temporal Workflow sketches (pseudocode — adapt to your workflow setup)
// ---------------------------------------------------------------------------

/**
 * Background workflow: triggered fire-and-forget from the main agent workflow.
 *
 * In your actual Temporal workflow file:
 *
 *   export async function backgroundMemoryWorkflow(
 *     messages: SerializedMessage[],
 *     userId: string
 *   ): Promise<void> {
 *     const { extractAndStoreMemories } = proxyActivities<typeof activities>({
 *       startToCloseTimeout: "60s",
 *       retry: { maximumAttempts: 3, initialInterval: "2s" },
 *     });
 *     await extractAndStoreMemories(messages, userId);
 *   }
 *
 * Triggered from the main workflow after each turn:
 *
 *   await startChild(backgroundMemoryWorkflow, {
 *     args: [serializedMessages, userId],
 *     workflowId: `memory-${userId}-${workflowInfo().runId}`,
 *     parentClosePolicy: ParentClosePolicy.ABANDON,
 *   });
 *
 * Main workflow memory injection:
 *
 *   const memories = await activities.fetchRelevantMemories(latestUserMessage, userId);
 *   const systemPrompt = memories.length > 0
 *     ? `${basePrompt}\n\nRelevant memories:\n${memories.map(m => `- ${m}`).join("\n")}`
 *     : basePrompt;
 */
