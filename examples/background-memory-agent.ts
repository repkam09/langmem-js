/**
 * Background Memory Extraction for a Chat AI Agent
 * =================================================
 *
 * This example demonstrates how to build a chat agent that extracts and stores
 * memories in the background — without blocking the main conversation response.
 *
 * The pattern:
 *   1. User sends a message.
 *   2. The agent immediately retrieves relevant past memories and generates a response.
 *   3. After responding, the agent kicks off asynchronous memory extraction from
 *      the conversation so far, storing any new facts for future sessions.
 *
 * This gives you the best of both worlds: fast responses and persistent memory.
 *
 * Dependencies: npm install @langchain/core @langchain/langgraph langsmith
 * Set env vars: ANTHROPIC_API_KEY (or your provider's key), LANGSMITH_API_KEY
 */

import { v4 as uuidv4 } from "uuid";
import { HumanMessage, AIMessage, SystemMessage, type BaseMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { Client } from "@langchain/langgraph-sdk";
import {
  SummarizationNode,
  type RunningSummary,
  createMemoryManager,
} from "../src/index.js";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const MODEL = new ChatAnthropic({ model: "claude-sonnet-4-6", maxTokens: 1024 });
const LANGGRAPH_URL = process.env.LANGGRAPH_URL ?? "http://localhost:2024";
const MEMORY_NAMESPACE = ["memories", "user"];

// Short-term memory: keep the most recent conversation within a token budget,
// summarizing older turns so the context window never overflows.
const summarizationNode = new SummarizationNode({
  model: MODEL,
  maxTokens: 4096,
  maxSummaryTokens: 512,
  // Use same key so the node replaces messages in-place
  inputMessagesKey: "messages",
  outputMessagesKey: "messages",
});

// ---------------------------------------------------------------------------
// Long-term memory manager (backed by LangGraph's store)
// ---------------------------------------------------------------------------
//
// createMemoryManager wraps a LangGraph store with tools that let the LLM
// save, update, and search memories during a conversation.
//
// Under the hood it uses a background thread to extract structured facts from
// the conversation after each turn and writes them to the store.

async function buildMemoryManager() {
  const lgClient = new Client({ apiUrl: LANGGRAPH_URL });
  return createMemoryManager({
    model: MODEL,
    // Namespace scopes memories to a specific user/session
    namespace: MEMORY_NAMESPACE,
    // Schemas describe what facts to extract. The model fills these in.
    schemas: {
      user_profile: {
        type: "object",
        description: "Persistent facts about the user",
        properties: {
          name: { type: "string" },
          preferences: { type: "array", items: { type: "string" } },
          important_dates: { type: "array", items: { type: "string" } },
        },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// In-process session state
// ---------------------------------------------------------------------------

interface SessionState {
  messages: BaseMessage[];
  runningSummary?: RunningSummary;
}

const sessions = new Map<string, SessionState>();

function getSession(sessionId: string): SessionState {
  if (!sessions.has(sessionId)) {
    sessions.set(sessionId, { messages: [] });
  }
  return sessions.get(sessionId)!;
}

// ---------------------------------------------------------------------------
// Core chat function
// ---------------------------------------------------------------------------

/**
 * Process one turn of conversation and return the assistant's reply.
 *
 * Memory extraction runs in the background (fire-and-forget) so it does NOT
 * add latency to the response. Any errors during extraction are logged but do
 * not affect the main conversation.
 */
async function chat(
  sessionId: string,
  userMessage: string,
  memoryManager?: Awaited<ReturnType<typeof buildMemoryManager>>
): Promise<string> {
  const session = getSession(sessionId);

  // 1. Append the new human message
  session.messages.push(new HumanMessage({ content: userMessage, id: uuidv4() }));

  // 2. (Optional) Retrieve relevant long-term memories and prepend as context
  let systemPrompt = "You are a helpful assistant with memory of past conversations.";
  if (memoryManager) {
    try {
      const memories = await memoryManager.search(userMessage, { limit: 5 });
      if (memories.length > 0) {
        const memoryText = memories
          .map((m: { content: string }) => `- ${m.content}`)
          .join("\n");
        systemPrompt += `\n\nRelevant memories from past conversations:\n${memoryText}`;
      }
    } catch {
      // Memory retrieval is best-effort; continue without it
    }
  }

  // 3. Apply short-term summarization to keep messages within the token budget
  //    This is synchronous because the LLM needs the condensed context to respond.
  const summarizationInput: Record<string, unknown> = {
    messages: session.messages,
  };
  if (session.runningSummary) {
    summarizationInput.context = { running_summary: session.runningSummary };
  }

  const summarizationResult = await summarizationNode.invoke(summarizationInput);
  const contextMessages = summarizationResult["messages"] as BaseMessage[];

  // Persist the updated running summary for the next turn
  if (summarizationResult["context"]) {
    session.runningSummary = (
      summarizationResult["context"] as Record<string, unknown>
    )["running_summary"] as RunningSummary | undefined;
  }

  // 4. Build the prompt: system message + (possibly summarized) context
  const promptMessages: BaseMessage[] = [
    new SystemMessage({ content: systemPrompt, id: uuidv4() }),
    // Filter out RemoveMessage sentinels — they are only meaningful to LangGraph
    ...contextMessages.filter((m) => m.getType() !== "remove"),
  ];

  // 5. Call the model
  const response = await MODEL.invoke(promptMessages);
  const assistantText =
    typeof response.content === "string" ? response.content : JSON.stringify(response.content);

  // 6. Append the assistant's reply to the session history
  session.messages.push(new AIMessage({ content: assistantText, id: uuidv4() }));

  // 7. Background memory extraction — fire-and-forget
  //    This runs after we've already prepared the response, so it adds zero
  //    latency for the user. Any facts extracted here become available in
  //    future sessions via the memory search in step 2.
  if (memoryManager) {
    extractMemoriesInBackground(session.messages, memoryManager).catch((err) => {
      console.error("[memory] Background extraction failed:", err);
    });
  }

  return assistantText;
}

/**
 * Asynchronously extracts and stores memories from the conversation.
 * Called with fire-and-forget semantics — it should never throw.
 */
async function extractMemoriesInBackground(
  messages: BaseMessage[],
  memoryManager: Awaited<ReturnType<typeof buildMemoryManager>>
): Promise<void> {
  // Only send human/AI turns — skip system messages and summaries
  const conversationMessages = messages.filter(
    (m) => m.getType() === "human" || m.getType() === "ai"
  );

  // extractAndStore analyses the recent conversation and calls manage_memory
  // to upsert/delete facts in the long-term store.
  await memoryManager.extractAndStore(conversationMessages);
}

// ---------------------------------------------------------------------------
// Simple REPL demo
// ---------------------------------------------------------------------------

async function main() {
  // In a real application you would persist sessions to a database and look
  // them up by user ID. Here we use a simple in-memory map for illustration.
  const sessionId = "demo-session";

  // Memory manager is optional. Omit it (or pass undefined) to run without
  // long-term memory (e.g. if you don't have a LangGraph server running).
  let memoryManager: Awaited<ReturnType<typeof buildMemoryManager>> | undefined;
  try {
    memoryManager = await buildMemoryManager();
    console.log("Long-term memory store connected.\n");
  } catch {
    console.log(
      "LangGraph server not available — running without long-term memory.\n"
    );
  }

  // Simulate a multi-turn conversation
  const turns = [
    "Hi! My name is Alex and I love hiking in the mountains.",
    "What are some good hiking spots in the Pacific Northwest?",
    "I prefer trails that are less crowded. Any suggestions?",
    "By the way, my birthday is on July 15th. Can you remember that?",
    "What do you know about me so far?",
  ];

  for (const userMessage of turns) {
    console.log(`User: ${userMessage}`);
    const response = await chat(sessionId, userMessage, memoryManager);
    console.log(`Assistant: ${response}\n`);
  }

  // Show what the short-term running summary looks like after several turns
  const session = getSession(sessionId);
  if (session.runningSummary) {
    console.log("--- Running conversation summary ---");
    console.log(session.runningSummary.summary);
    console.log(`(${session.runningSummary.summarizedMessageIds.size} messages summarized)`);
  }
}

// Run only when executed directly (not when imported as a module)
if (import.meta.url === new URL(process.argv[1], "file:").href) {
  main().catch(console.error);
}

export { chat, extractMemoriesInBackground, type SessionState };
