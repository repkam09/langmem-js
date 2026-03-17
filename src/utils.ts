import { v4 as uuidv4 } from "uuid";
import {
  BaseMessage,
  HumanMessage,
  AIMessage,
  SystemMessage,
} from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import { ConfigurationError } from "./errors.js";

export type AnyMessage = BaseMessage;
export type Namespace = readonly string[];

// NamespaceTemplate resolves namespace templates with {variable} placeholders.
export class NamespaceTemplate {
  private readonly template: readonly string[];
  private readonly vars: Map<number, string>;

  constructor(template: string | readonly string[] | NamespaceTemplate) {
    if (template instanceof NamespaceTemplate) {
      this.template = template.template;
      this.vars = template.vars;
      return;
    }
    this.template = typeof template === "string" ? [template] : template;
    this.vars = new Map();
    for (let i = 0; i < this.template.length; i++) {
      const key = getKey(this.template[i]);
      if (key !== null) {
        this.vars.set(i, key);
      }
    }
  }

  call(config?: RunnableConfig): readonly string[] {
    if (this.vars.size === 0) {
      return this.template;
    }
    const configurable = config?.configurable ?? {};
    return this.template.map((ns, i) => {
      if (this.vars.has(i)) {
        const key = this.vars.get(i)!;
        if (!(key in configurable)) {
          throw new ConfigurationError(
            `Missing key in 'configurable' field: ${key}. Available keys: ${Object.keys(configurable).join(", ")}`
          );
        }
        return String(configurable[key]);
      }
      return ns;
    });
  }
}

function getKey(ns: string): string | null {
  if (ns.startsWith("{") && ns.endsWith("}")) {
    return ns.slice(1, -1);
  }
  return null;
}

// Merge consecutive messages from the same role
function mergeMessageRuns(messages: AnyMessage[]): AnyMessage[] {
  if (messages.length === 0) return [];
  const result: AnyMessage[] = [];
  let current: AnyMessage = messages[0];
  for (let i = 1; i < messages.length; i++) {
    if (messages[i]._getType() === current._getType()) {
      const mergedContent =
        (typeof current.content === "string"
          ? current.content
          : JSON.stringify(current.content)) +
        "\n" +
        (typeof messages[i].content === "string"
          ? messages[i].content
          : JSON.stringify(messages[i].content));
      // Create a new message of the same type with merged content
      current = (current.constructor as any)(mergedContent);
    } else {
      result.push(current);
      current = messages[i];
    }
  }
  result.push(current);
  return result;
}

function prettyRepr(message: AnyMessage): string {
  const type = message._getType();
  const roleMap: Record<string, string> = {
    human: "Human",
    ai: "AI",
    system: "System",
    tool: "Tool",
    function: "Function",
  };
  const role = roleMap[type] ?? type;
  const title = getMsgTitleRepr(role);
  const content =
    typeof message.content === "string"
      ? message.content
      : JSON.stringify(message.content);
  return `${title}\n\n${content}`;
}

function getMsgTitleRepr(title: string): string {
  const padded = ` ${title} `;
  const sepLen = Math.floor((80 - padded.length) / 2);
  const sep = "=".repeat(sepLen);
  const secondSep = padded.length % 2 === 0 ? sep : sep + "=";
  return `${sep}${padded}${secondSep}`;
}

export function getConversation(
  messages: (AnyMessage | Record<string, unknown>)[],
  delimiter = "\n\n"
): string {
  const anyMessages = messages.map((m) => {
    if (m instanceof BaseMessage) return m;
    // Handle plain message objects {role, content}
    const role = (m as any).role as string;
    const content = (m as any).content as string;
    if (role === "human" || role === "user") return new HumanMessage(content);
    if (role === "assistant" || role === "ai") return new AIMessage(content);
    if (role === "system") return new SystemMessage(content);
    return new HumanMessage(content);
  });
  const merged = mergeMessageRuns(anyMessages);
  return merged.map(prettyRepr).join(delimiter);
}

export function getDilatedWindows(
  messages: AnyMessage[],
  N = 5,
  delimiter = "\n\n"
): string[] {
  if (!messages.length) return [];
  const M = messages.length;
  const seen = new Set<number>();
  const result: string[] = [];
  for (let i = 0; i < N; i++) {
    const size = Math.min(M, 1 << i);
    if (size > M) break;
    const query = getConversation(messages.slice(M - size), delimiter);
    if (!seen.has(size)) {
      seen.add(size);
      result.push(query);
    } else {
      break;
    }
  }
  return result;
}

type Session =
  | string
  | AnyMessage[]
  | [AnyMessage[], string | Record<string, unknown>];

export function formatSessions(sessions: Session | Session[]): string {
  if (!sessions) return "";

  let normalizedSessions: Array<
    [AnyMessage[] | string, string | Record<string, unknown>]
  >;

  if (typeof sessions === "string") {
    normalizedSessions = [[sessions as any, ""]];
  } else if (Array.isArray(sessions)) {
    if (sessions.length === 0) return "";
    // Check if it's a single session (array of messages or tuple)
    if (
      Array.isArray(sessions[0]) ||
      sessions[0] instanceof BaseMessage
    ) {
      // It's a list of messages
      normalizedSessions = [[sessions as AnyMessage[], ""]];
    } else if (
      Array.isArray(sessions) &&
      sessions.length === 2 &&
      Array.isArray((sessions as any)[0])
    ) {
      // It's a single (messages, feedback) tuple
      normalizedSessions = [sessions as any];
    } else {
      normalizedSessions = (sessions as Session[]).map((s) => {
        if (typeof s === "string") return [s as any, ""];
        if (Array.isArray(s) && s.length >= 2 && Array.isArray(s[0])) {
          return [s[0] as AnyMessage[], s[1] as string];
        }
        return [s as AnyMessage[], ""];
      });
    }
  } else {
    throw new Error(`Expected list of sessions, got ${typeof sessions}`);
  }

  const ids = normalizedSessions.map(() => uuidv4().replace(/-/g, ""));
  const acc: string[] = [];

  for (let i = 0; i < normalizedSessions.length; i++) {
    const id = ids[i];
    const [session, feedback] = normalizedSessions[i];
    const conv =
      typeof session === "string" ? session : getConversation(session);
    let feedbackStr = "";
    if (feedback) {
      const fbStr =
        typeof feedback === "string" ? feedback : JSON.stringify(feedback);
      feedbackStr = `\n\nFeedback for session ${id}:\n<FEEDBACK>\n${fbStr}\n</FEEDBACK>`;
    }
    acc.push(`<session_${id}>\n${conv}${feedbackStr}\n</session_${id}>`);
  }

  return acc.join("\n\n");
}

export function getVarHealer(
  vars: Set<string> | string,
  allRequired = false
): (input: string) => string {
  let varSet: Set<string>;
  if (typeof vars === "string") {
    const found = vars.match(/\{(.+?)\}/g) ?? [];
    varSet = new Set(found.map((v) => v.slice(1, -1)));
  } else {
    varSet = vars;
  }

  const varToUuid = new Map<string, string>();
  for (const v of varSet) {
    varToUuid.set(`{${v}}`, uuidv4().replace(/-/g, ""));
  }
  const uuidToVar = new Map<string, string>();
  for (const [k, v] of varToUuid) {
    uuidToVar.set(v, k);
  }

  function escape(input: string): string {
    // Escape single braces to double braces (template literals escaping)
    let result = input.replace(/(?<!\{)\{(?!\{)/g, "{{");
    result = result.replace(/(?<!\})\}(?!\})/g, "}}");
    return result;
  }

  if (varSet.size === 0) {
    return escape;
  }

  const maskPattern = new RegExp(
    Array.from(varToUuid.keys())
      .map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("|"),
    "g"
  );
  const unmaskPattern = new RegExp(
    Array.from(varToUuid.values())
      .map((v) => v.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("|"),
    "g"
  );
  const stripToOptimizePattern = /<TO_OPTIMIZE.*?>|<\/TO_OPTIMIZE>/gs;

  function assertAllRequired(input: string): string {
    if (!allRequired) return input;
    const missing = Array.from(varSet).filter(
      (v) => !input.includes(`{${v}}`)
    );
    if (missing.length > 0) {
      throw new Error(`Missing required variable: ${missing.join(", ")}`);
    }
    return input;
  }

  function mask(input: string): string {
    return input.replace(maskPattern, (m) => varToUuid.get(m) ?? m);
  }

  function unmask(input: string): string {
    return input.replace(unmaskPattern, (m) => uuidToVar.get(m) ?? m);
  }

  return function pipe(input: string): string {
    const masked = mask(assertAllRequired(input));
    const escaped = escape(masked);
    const stripped = escaped.replace(stripToOptimizePattern, "");
    return unmask(stripped);
  };
}

export function getPromptExtractionSchema(originalPrompt: string): {
  name: string;
  description: string;
  promptDescription: string;
  pipeline: (input: string) => string;
  requiredVariables: Set<string>;
} {
  const requiredVariables = new Set(
    Array.from(originalPrompt.matchAll(/\{(.+?)\}/g), (m) => m[1])
  );

  let promptDescription: string;
  if (requiredVariables.size > 0) {
    const variablesStr = Array.from(requiredVariables)
      .map((v) => `{${v}}`)
      .join(", ");
    promptDescription =
      ` The prompt section being optimized contains the following template variables: ${variablesStr}.` +
      " You must retain all of these variables in your improved prompt. No other input variables are allowed.";
  } else {
    promptDescription =
      " The prompt section being optimized contains no input template variables." +
      " Any brackets {{ foo }} you emit will be escaped and not used.";
  }

  const pipeline = getVarHealer(requiredVariables, true);

  return {
    name: "OptimizedPromptOutput",
    description: "Schema for the optimized prompt output.",
    promptDescription,
    pipeline,
    requiredVariables,
  };
}

export function dumps(obj: unknown): string {
  return JSON.stringify(obj);
}
