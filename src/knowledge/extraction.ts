import { v4 as uuidv4 } from "uuid";
import { z } from "zod";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage } from "@langchain/core/messages";
import type { BaseMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { BaseStore } from "@langchain/langgraph";
import { getConversation } from "../utils.js";
import { createSearchMemoryTool } from "./tools.js";

export interface MessagesState {
  messages: (BaseMessage | Record<string, unknown>)[];
}

export interface MemoryState extends MessagesState {
  existing?: Array<
    | [string, Record<string, unknown>]
    | [string, string, Record<string, unknown>]
  >;
  max_steps?: number;
}

export interface SummarizeThread {
  title: string;
  summary: string;
}

export interface ExtractedMemory {
  id: string;
  content: Record<string, unknown>;
}

const SummarizeThreadSchema = z.object({
  title: z.string().describe("Short title for the conversation."),
  summary: z.string().describe("Summary of the conversation."),
});

const MemorySchema = z.object({
  content: z
    .string()
    .describe(
      "The memory as a well-written, standalone episode/fact/note/preference/etc. Refer to the user's instructions for more information the preferred memory organization."
    ),
});

export const MEMORY_INSTRUCTIONS = `You are a long-term memory manager maintaining a core store of semantic, procedural, and episodic memory. These memories power a life-long learning agent's core predictive model.

What should the agent learn from this interaction about the user, itself, or how it should act? Reflect on the input trajectory and current memories (if any).

1. **Extract & Contextualize**
   - Identify essential facts, relationships, preferences, reasoning procedures, and context
   - Caveat uncertain or suppositional information with confidence levels (p(x)) and reasoning
   - Quote supporting information when necessary

2. **Compare & Update**
   - Attend to novel information that deviates from existing memories and expectations.
   - Consolidate and compress redundant memories to maintain information-density; strengthen based on reliability and recency; maximize SNR by avoiding idle words.
   - Remove incorrect or redundant memories while maintaining internal consistency

3. **Synthesize & Reason**
   - What can you conclude about the user, agent ("I"), or environment using deduction, induction, and abduction?
   - What patterns, relationships, and principles emerge about optimal responses?
   - What generalizations can you make?
   - Qualify conclusions with probabilistic confidence and justification

As the agent, record memory content exactly as you'd want to recall it when predicting how to act or respond.
Prioritize retention of surprising (pattern deviation) and persistent (frequently reinforced) information, ensuring nothing worth remembering is forgotten and nothing false is remembered. Prefer dense, complete memories over overlapping ones.`;

export function createThreadExtractor(
  model: BaseChatModel,
  options?: {
    schema?: z.ZodType;
    instructions?: string;
  }
): { invoke: (input: MessagesState, config?: RunnableConfig) => Promise<unknown> } {
  const schema = options?.schema ?? SummarizeThreadSchema;
  const instructions =
    options?.instructions ??
    "You are tasked with summarizing the following conversation.";

  const extractorModel = model.withStructuredOutput(schema);

  return {
    async invoke(
      input: MessagesState,
      _config?: RunnableConfig
    ): Promise<unknown> {
      const conversation = getConversation(input.messages as any[]);
      const messages = [
        { role: "system", content: instructions },
        {
          role: "user",
          content: `Call the provided tool based on the conversation below:\n\n<conversation>${conversation}</conversation>`,
        },
      ];
      return extractorModel.invoke(messages);
    },
  };
}

export class MemoryManager {
  private model: BaseChatModel;
  private schemas: z.ZodType[];
  private instructions: string;
  private enableInserts: boolean;
  private enableUpdates: boolean;
  private enableDeletes: boolean;

  constructor(
    model: BaseChatModel,
    options?: {
      schemas?: z.ZodType[];
      instructions?: string;
      enableInserts?: boolean;
      enableUpdates?: boolean;
      enableDeletes?: boolean;
    }
  ) {
    this.model = model;
    this.schemas = options?.schemas ?? [MemorySchema];
    this.instructions = options?.instructions ?? MEMORY_INSTRUCTIONS;
    this.enableInserts = options?.enableInserts ?? true;
    this.enableUpdates = options?.enableUpdates ?? true;
    this.enableDeletes = options?.enableDeletes ?? false;
  }

  private prepareMessages(
    messages: (BaseMessage | Record<string, unknown>)[],
    maxSteps = 1
  ): Record<string, unknown>[] {
    const id = uuidv4().replace(/-/g, "");
    const conversation = getConversation(messages as any[]);
    let session = `\n\n<session_${id}>\n${conversation}\n</session_${id}>`;
    if (maxSteps > 1) {
      session += `\n\nYou have a maximum of ${maxSteps - 1} attempts to form and consolidate memories from this session.`;
    }
    return [
      { role: "system", content: "You are a memory subroutine for an AI." },
      {
        role: "user",
        content:
          `${this.instructions}\n\nEnrich, prune, and organize memories based on any new information. ` +
          `If an existing memory is incorrect or outdated, update it based on the new information. ` +
          `All operations must be done in single parallel multi-tool call.` +
          ` Avoid duplicate extractions. ${session}`,
      },
    ];
  }

  private prepareExisting(
    existing?: MemoryState["existing"]
  ): Array<[string, string, Record<string, unknown>]> {
    if (!existing) return [];
    return existing.map((e) => {
      if (e.length === 3) {
        return e as [string, string, Record<string, unknown>];
      }
      const [id, value] = e as [string, Record<string, unknown>];
      const kind =
        typeof value === "object" && value !== null
          ? (value as any).__typename ?? "__any__"
          : "__any__";
      return [id, kind, value] as [string, string, Record<string, unknown>];
    });
  }

  async invoke(
    input: MemoryState,
    _config?: RunnableConfig
  ): Promise<ExtractedMemory[]> {
    const maxSteps = input.max_steps ?? 1;
    const messages = input.messages;
    const existing = input.existing;

    const preparedMessages = this.prepareMessages(messages, maxSteps);
    const preparedExisting = this.prepareExisting(existing);
    const externalIds = new Set(preparedExisting.map(([id]) => id));

    // Build tools from schemas
    const tools: Array<{
      name: string;
      description: string;
      schema: z.ZodType;
    }> = this.schemas.map((schema, i) => ({
      name: `memory_${i}`,
      description: "Record a memory.",
      schema,
    }));

    // Add a delete tool if deletes are enabled
    if (this.enableDeletes) {
      tools.push({
        name: "delete_memory",
        description: "Delete an existing memory by ID.",
        schema: z.object({
          id: z.string().describe("The ID of the memory to delete."),
        }),
      });
    }

    const existingMemoriesStr =
      preparedExisting.length > 0
        ? "\n\nExisting memories:\n" +
          preparedExisting
            .map(([id, , mem]) => `ID: ${id}\n${JSON.stringify(mem)}`)
            .join("\n\n")
        : "";

    const messagesWithExisting = [
      ...preparedMessages.slice(0, 1),
      {
        role: "user",
        content:
          (preparedMessages[1] as any).content + existingMemoriesStr,
      },
    ];

    const boundModel = this.model.bindTools!(
      tools.map((t) => ({
        name: t.name,
        description: t.description,
        parameters: t.schema,
      }))
    );

    const results = new Map<string, Record<string, unknown>>();

    let currentMessages = messagesWithExisting;

    for (let i = 0; i < maxSteps; i++) {
      const response = await boundModel.invoke(currentMessages as any);
      const aiMsg = response as AIMessage;

      if (!aiMsg.tool_calls || aiMsg.tool_calls.length === 0) break;

      const stepResults = new Map<string, Record<string, unknown>>();
      const toolMessages: Record<string, unknown>[] = [];

      for (const tc of aiMsg.tool_calls) {
        if (tc.name === "delete_memory") {
          const deleteId = (tc.args as any).id;
          if (deleteId && externalIds.has(deleteId)) {
            stepResults.set(deleteId, { __deleted: true });
          }
          toolMessages.push({
            role: "tool",
            content: `Memory ${deleteId} deleted.`,
            tool_call_id: tc.id ?? "",
          });
        } else {
          const memId = uuidv4();
          stepResults.set(memId, tc.args as Record<string, unknown>);
          toolMessages.push({
            role: "tool",
            content: `Memory ${memId} created.`,
            tool_call_id: tc.id ?? "",
          });
        }
      }

      for (const [id, value] of stepResults) {
        results.set(id, value);
      }

      // Also retain existing memories that weren't updated
      for (const [id, , mem] of preparedExisting) {
        if (!results.has(id)) {
          results.set(id, mem);
        }
      }

      if (i < maxSteps - 1) {
        currentMessages = [
          ...currentMessages,
          aiMsg as any,
          ...toolMessages,
        ];
      } else {
        break;
      }
    }

    // Filter results
    return Array.from(results.entries())
      .filter(([id, value]) => {
        if ((value as any).__deleted) {
          return externalIds.has(id);
        }
        return true;
      })
      .map(([id, content]) => ({ id, content }));
  }
}

export function createMemoryManager(
  model: BaseChatModel,
  options?: {
    schemas?: z.ZodType[];
    instructions?: string;
    enableInserts?: boolean;
    enableUpdates?: boolean;
    enableDeletes?: boolean;
  }
): MemoryManager {
  return new MemoryManager(model, options);
}

export interface CreateMemoryStoreManagerOptions {
  model: BaseChatModel;
  store?: BaseStore;
  namespace?: string | readonly string[];
  schemas?: z.ZodType[];
  instructions?: string;
  enableInserts?: boolean;
  enableUpdates?: boolean;
  enableDeletes?: boolean;
}

export function createMemoryStoreManager(
  options: CreateMemoryStoreManagerOptions
): { invoke: (input: MessagesState, config?: RunnableConfig) => Promise<void> } {
  const {
    model,
    store,
    namespace = ["memories"],
    schemas,
    instructions,
    enableInserts,
    enableUpdates,
    enableDeletes,
  } = options;

  const manager = createMemoryManager(model, {
    schemas,
    instructions,
    enableInserts,
    enableUpdates,
    enableDeletes,
  });

  return {
    async invoke(
      input: MessagesState,
      config?: RunnableConfig
    ): Promise<void> {
      const actualStore =
        store ?? (config?.configurable as any)?.store;
      if (!actualStore) {
        throw new Error(
          "No store provided. Pass a store directly or use within a LangGraph context."
        );
      }

      const ns =
        typeof namespace === "string" ? [namespace] : Array.from(namespace);

      // Load existing memories from store
      const existingItems = await actualStore.search(ns, {});
      const existing: MemoryState["existing"] = existingItems.map(
        (item: any) => [item.key, item.value ?? {}]
      );

      const result = await manager.invoke({ ...input, existing }, config);

      // Persist results back to store
      for (const { id, content } of result) {
        if ((content as any).__deleted) {
          await actualStore.delete(ns, id);
        } else {
          await actualStore.put(ns, id, content);
        }
      }
    },
  };
}

export function createMemorySearcher(options: {
  namespace: string | readonly string[];
  store?: BaseStore;
  schemas?: z.ZodType[];
}) {
  return createSearchMemoryTool({
    namespace: options.namespace,
    store: options.store,
  });
}
