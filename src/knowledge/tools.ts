import { v4 as uuidv4 } from "uuid";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import type { BaseStore } from "@langchain/langgraph";
import type { RunnableConfig } from "@langchain/core/runnables";
import { NamespaceTemplate } from "../utils.js";
import { ConfigurationError } from "../errors.js";

const DEFAULT_MANAGE_MEMORY_INSTRUCTIONS = `Proactively call this tool when you:

1. Identify a new USER preference.
2. Receive an explicit USER request to remember something or otherwise alter your behavior.
3. Are working and want to record important context.
4. Identify that an existing MEMORY is incorrect or outdated.
`;

export type MemoryAction = "create" | "update" | "delete";

export interface CreateManageMemoryToolOptions {
  namespace: string | readonly string[];
  instructions?: string;
  schema?: z.ZodType;
  actionsPermitted?: MemoryAction[];
  store?: BaseStore;
  name?: string;
}

function getStore(
  initialStore?: BaseStore,
  config?: RunnableConfig
): BaseStore {
  if (initialStore) return initialStore;
  // Try to get from config (LangGraph injects store into config.configurable)
  const store = (config?.configurable as any)?.store;
  if (store) return store;
  throw new ConfigurationError(
    "Could not get store. Please provide a store directly or use within a LangGraph context."
  );
}

function ensureJsonSerializable(content: unknown): unknown {
  if (
    content === null ||
    typeof content === "string" ||
    typeof content === "number" ||
    typeof content === "boolean"
  ) {
    return content;
  }
  if (Array.isArray(content)) return content;
  if (typeof content === "object") return content;
  return String(content);
}

export function createManageMemoryTool(
  options: CreateManageMemoryToolOptions
): DynamicStructuredTool {
  const {
    namespace,
    instructions = DEFAULT_MANAGE_MEMORY_INSTRUCTIONS,
    schema = z.string(),
    actionsPermitted = ["create", "update", "delete"],
    store: initialStore,
    name = "manage_memory",
  } = options;

  if (!actionsPermitted.length) {
    throw new Error("actionsPermitted cannot be empty");
  }

  const namespacer = new NamespaceTemplate(
    namespace as string | readonly string[]
  );
  const defaultAction = actionsPermitted.includes("create")
    ? "create"
    : actionsPermitted[0];

  const actionEnum = z.enum(
    actionsPermitted as [MemoryAction, ...MemoryAction[]]
  );

  const toolSchema = z.object({
    content: schema.optional().describe("Content for new/updated memory"),
    action: actionEnum
      .default(defaultAction as MemoryAction)
      .describe("The action to perform"),
    id: z
      .string()
      .uuid()
      .optional()
      .describe("ID of existing memory to update/delete"),
  });

  const verbs =
    actionsPermitted.length === 1
      ? `${actionsPermitted[0]} a memory`
      : actionsPermitted.length === 2
      ? `${
          actionsPermitted[0].charAt(0).toUpperCase() +
          actionsPermitted[0].slice(1)
        } or ${actionsPermitted[1]} a memory`
      : `${[
          actionsPermitted[0].charAt(0).toUpperCase() +
            actionsPermitted[0].slice(1),
          ...actionsPermitted.slice(1, -1),
        ].join(", ")}, or ${
          actionsPermitted[actionsPermitted.length - 1]
        } a memory`;

  const description = `${verbs} to persist across conversations.
Include the MEMORY ID when updating or deleting a MEMORY. Omit when creating a new MEMORY - it will be created for you.
${instructions}`;

  return new DynamicStructuredTool({
    name,
    description,
    schema: toolSchema,
    func: async (input, _runManager, config) => {
      const { content, action = defaultAction, id } = input as any;
      const store = getStore(initialStore, config);

      if (!actionsPermitted.includes(action)) {
        throw new Error(
          `Invalid action ${action}. Must be one of ${actionsPermitted.join(", ")}.`
        );
      }
      if (action === "create" && id !== undefined) {
        throw new Error(
          "You cannot provide a MEMORY ID when creating a MEMORY. Please try again, omitting the id argument."
        );
      }
      if ((action === "delete" || action === "update") && !id) {
        throw new Error(
          "You must provide a MEMORY ID when deleting or updating a MEMORY."
        );
      }

      const ns = namespacer.call(config);

      if (action === "delete") {
        await store.delete(ns as string[], id!);
        return `Deleted memory ${id}`;
      }

      const memoryId = id ?? uuidv4();
      await store.put(ns as string[], memoryId, {
        content: ensureJsonSerializable(content),
      });
      return `${action}d memory ${memoryId}`;
    },
  });
}

export interface CreateSearchMemoryToolOptions {
  namespace: string | readonly string[];
  instructions?: string;
  store?: BaseStore;
  name?: string;
  responseFormat?: "content" | "content_and_artifact";
}

export function createSearchMemoryTool(
  options: CreateSearchMemoryToolOptions
): DynamicStructuredTool {
  const {
    namespace,
    instructions = "",
    store: initialStore,
    name = "search_memory",
    responseFormat = "content",
  } = options;

  const namespacer = new NamespaceTemplate(
    namespace as string | readonly string[]
  );

  const description = `Search your long-term memories for information relevant to your current context. ${instructions}`;

  const toolSchema = z.object({
    query: z.string().describe("Search query to match against memories"),
    limit: z
      .number()
      .int()
      .default(10)
      .describe("Maximum number of results to return"),
    offset: z
      .number()
      .int()
      .default(0)
      .describe("Number of results to skip"),
    filter: z
      .record(z.unknown())
      .optional()
      .describe("Additional filter criteria"),
  });

  return new DynamicStructuredTool({
    name,
    description,
    schema: toolSchema,
    func: async (input, _runManager, config) => {
      const { query, limit = 10, offset = 0, filter } = input as any;
      const store = getStore(initialStore, config);
      const ns = namespacer.call(config);

      const memories = await store.search(ns as string[], {
        query,
        filter,
        limit,
        offset,
      });
      const serialized = JSON.stringify(
        memories.map((m: any) => ({
          namespace: m.namespace,
          key: m.key,
          value: m.value,
          created_at: m.createdAt,
          updated_at: m.updatedAt,
          score: m.score,
        }))
      );

      if (responseFormat === "content_and_artifact") {
        return [serialized, memories];
      }
      return serialized;
    },
  });
}
