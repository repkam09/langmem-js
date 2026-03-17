import type { BaseStore } from "@langchain/langgraph";
import type { RunnableConfig } from "@langchain/core/runnables";
import { NamespaceTemplate } from "./utils.js";

export interface MemoryItem {
  namespace: string[];
  key: string;
  value: Record<string, unknown>;
  created_at: Date;
  updated_at: Date;
  score?: number;
}

export interface SubmitOptions {
  config?: RunnableConfig;
  afterSeconds?: number;
  threadId?: string;
}

export interface Executor {
  submit(
    payload: Record<string, unknown>,
    options?: SubmitOptions
  ): Promise<unknown>;
  search(options?: {
    query?: string;
    filter?: Record<string, unknown>;
    limit?: number;
    offset?: number;
    namespace?: string | string[];
  }): Promise<MemoryItem[]>;
  shutdown(options?: {
    wait?: boolean;
    cancelFutures?: boolean;
  }): Promise<void>;
}

interface PendingTask {
  threadId?: string;
  payload: Record<string, unknown>;
  executeAt: number;
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
  cancelled: boolean;
  config?: RunnableConfig;
}

export class RemoteReflectionExecutor implements Executor {
  private namespace: string[];
  private assistantId: string;
  private client: unknown;
  private _url: string | undefined;

  constructor(
    namespace: string | string[],
    reflector: string,
    options?: { url?: string; client?: unknown; syncClient?: unknown }
  ) {
    this.namespace =
      typeof namespace === "string" ? [namespace] : namespace;
    this.assistantId = reflector;
    this._url = options?.url;
    this.client = options?.client ?? null;
  }

  private async getClient(): Promise<any> {
    if (this.client) return this.client;
    const { Client } = await import("@langchain/langgraph-sdk");
    this.client = new Client({ apiUrl: this._url });
    return this.client;
  }

  async submit(
    payload: Record<string, unknown>,
    options?: SubmitOptions
  ): Promise<unknown> {
    const client = await this.getClient();
    const threadId =
      options?.threadId ??
      (options?.config?.configurable as any)?.thread_id;
    const afterSeconds = options?.afterSeconds ?? 0;

    return client.runs.create(threadId, this.assistantId, {
      input: payload,
      config: {
        configurable: {
          thread_id: threadId,
          namespace: this.namespace,
        },
      },
      multitask_strategy: "rollback",
      after_seconds: afterSeconds,
      if_not_exists: "create",
    });
  }

  async search(options?: {
    query?: string;
    filter?: Record<string, unknown>;
    limit?: number;
    offset?: number;
    namespace?: string | string[];
  }): Promise<MemoryItem[]> {
    const client = await this.getClient();
    const ns = options?.namespace
      ? typeof options.namespace === "string"
        ? [options.namespace]
        : options.namespace
      : this.namespace;

    const results = await client.store.searchItems(ns, {
      query: options?.query,
      filter: options?.filter,
      limit: options?.limit ?? 10,
      offset: options?.offset ?? 0,
    });

    return (results.items ?? []) as MemoryItem[];
  }

  async shutdown(
    _options?: { wait?: boolean; cancelFutures?: boolean }
  ): Promise<void> {
    // No-op for remote executor
  }
}

export class LocalReflectionExecutor implements Executor {
  private namespace: NamespaceTemplate;
  private reflector: {
    invoke: (
      payload: Record<string, unknown>,
      config?: RunnableConfig
    ) => Promise<unknown>;
  };
  private store: BaseStore | null;
  private pendingTasks: Map<string, PendingTask>;
  private running: boolean;
  private queue: PendingTask[];
  private processingInterval: ReturnType<typeof setInterval> | null;

  constructor(
    reflector: {
      namespace?: NamespaceTemplate;
      invoke: (
        payload: Record<string, unknown>,
        config?: RunnableConfig
      ) => Promise<unknown>;
    },
    store?: BaseStore
  ) {
    if (!reflector.namespace) {
      throw new Error("reflector must have a namespace attribute");
    }
    this.namespace = reflector.namespace;
    this.reflector = reflector;
    this.store = store ?? null;
    this.pendingTasks = new Map();
    this.running = true;
    this.queue = [];
    this.processingInterval = setInterval(
      () => void this.processQueue(),
      100
    );
  }

  private async processQueue(): Promise<void> {
    if (!this.running && this.queue.length === 0) {
      if (this.processingInterval) {
        clearInterval(this.processingInterval);
        this.processingInterval = null;
      }
      return;
    }

    const now = Date.now();
    const ready: PendingTask[] = [];
    const remaining: PendingTask[] = [];

    for (const task of this.queue) {
      if (task.cancelled) {
        task.resolve(null);
        if (task.threadId) this.pendingTasks.delete(task.threadId);
      } else if (task.executeAt <= now) {
        ready.push(task);
      } else {
        remaining.push(task);
      }
    }

    this.queue = remaining;

    for (const task of ready) {
      if (task.cancelled) {
        task.resolve(null);
        if (task.threadId) this.pendingTasks.delete(task.threadId);
        continue;
      }

      try {
        const result = await this.reflector.invoke(task.payload, task.config);
        task.resolve(result);
      } catch (e) {
        task.reject(e);
      } finally {
        if (task.threadId) this.pendingTasks.delete(task.threadId);
      }
    }
  }

  submit(
    payload: Record<string, unknown>,
    options?: SubmitOptions
  ): Promise<unknown> {
    const threadId =
      options?.threadId ??
      (options?.config?.configurable as any)?.thread_id;
    const afterSeconds = options?.afterSeconds ?? 0;

    // Cancel existing task for same thread
    if (threadId && this.pendingTasks.has(threadId)) {
      const existing = this.pendingTasks.get(threadId)!;
      existing.cancelled = true;
    }

    return new Promise((resolve, reject) => {
      const task: PendingTask = {
        threadId,
        payload,
        executeAt: Date.now() + afterSeconds * 1000,
        resolve,
        reject,
        cancelled: false,
        config: options?.config,
      };

      if (threadId) {
        this.pendingTasks.set(threadId, task);
      }
      this.queue.push(task);
    });
  }

  async search(options?: {
    query?: string;
    filter?: Record<string, unknown>;
    limit?: number;
    offset?: number;
    namespace?: string | string[];
  }): Promise<MemoryItem[]> {
    if (!this.store) {
      throw new Error("No store available for search.");
    }

    const ns = options?.namespace
      ? typeof options.namespace === "string"
        ? [options.namespace]
        : options.namespace
      : (this.namespace.call() as string[]);

    const results = await this.store.search(ns, {
      query: options?.query,
      filter: options?.filter,
      limit: options?.limit ?? 10,
      offset: options?.offset ?? 0,
    });

    return results.map((item: any) => ({
      namespace: item.namespace,
      key: item.key,
      value: item.value,
      created_at: item.createdAt,
      updated_at: item.updatedAt,
      score: item.score,
    }));
  }

  async shutdown(options?: {
    wait?: boolean;
    cancelFutures?: boolean;
  }): Promise<void> {
    this.running = false;

    if (options?.cancelFutures) {
      for (const task of this.pendingTasks.values()) {
        task.cancelled = true;
        task.resolve(null);
      }
      this.pendingTasks.clear();
      this.queue = [];
    }

    if (options?.wait !== false) {
      // Wait for all pending tasks
      const pending = Array.from(this.pendingTasks.values()).map(
        (task) =>
          new Promise<void>((resolve) => {
            const original = task.resolve;
            task.resolve = (v) => {
              original(v);
              resolve();
            };
          })
      );
      await Promise.all(pending);
    }

    if (this.processingInterval) {
      clearInterval(this.processingInterval);
      this.processingInterval = null;
    }
  }
}

export function ReflectionExecutor(
  reflector:
    | string
    | {
        namespace?: NamespaceTemplate;
        invoke: (
          payload: Record<string, unknown>,
          config?: RunnableConfig
        ) => Promise<unknown>;
      },
  namespaceOrOptions?: string | string[] | { store?: BaseStore },
  options?: { url?: string; client?: unknown; store?: BaseStore }
): RemoteReflectionExecutor | LocalReflectionExecutor {
  if (typeof reflector === "string") {
    if (
      !namespaceOrOptions ||
      (typeof namespaceOrOptions === "object" &&
        !Array.isArray(namespaceOrOptions))
    ) {
      throw new Error("namespace is required for remote reflection");
    }
    const namespace = namespaceOrOptions as string | string[];
    return new RemoteReflectionExecutor(namespace, reflector, options);
  } else {
    const store =
      (
        namespaceOrOptions as { store?: BaseStore } | undefined
      )?.store ?? options?.store;
    return new LocalReflectionExecutor(reflector as any, store);
  }
}
