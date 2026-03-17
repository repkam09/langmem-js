import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  AIMessage,
  SystemMessage,
  HumanMessage,
  ToolMessage,
  RemoveMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import { REMOVE_ALL_MESSAGES } from "@langchain/langgraph";

export interface RunningSummary {
  summary: string;
  summarizedMessageIds: Set<string>;
  lastSummarizedMessageId: string | null;
}

export interface SummarizationResult {
  messages: BaseMessage[];
  runningSummary?: RunningSummary;
}

interface PreprocessedMessages {
  messagesToSummarize: BaseMessage[];
  nTokensToSummarize: number;
  maxTokensToSummarize: number;
  totalSummarizedMessages: number;
  existingSystemMessage: SystemMessage | null;
}

export type TokenCounter = (
  messages: (BaseMessage | Record<string, unknown>)[]
) => number;

function trimMessagesForSummarization(
  messages: BaseMessage[],
  maxTokens: number,
  tokenCounter: TokenCounter
): BaseMessage[] {
  // Keep the last maxTokens worth of messages from the end
  let tokens = 0;
  let startIdx = messages.length;

  for (let i = messages.length - 1; i >= 0; i--) {
    const msgTokens = tokenCounter([messages[i]]);
    if (tokens + msgTokens > maxTokens) {
      break;
    }
    tokens += msgTokens;
    startIdx = i;
  }

  let trimmed = messages.slice(startIdx);

  // Ensure starts on HumanMessage (equivalent to start_on="human")
  while (trimmed.length > 0 && !(trimmed[0] instanceof HumanMessage)) {
    trimmed = trimmed.slice(1);
  }

  return trimmed;
}

export function countTokensApproximately(
  messages: (BaseMessage | Record<string, unknown>)[]
): number {
  // Approximate: 1 token ~= 4 characters
  let total = 0;
  for (const m of messages) {
    const content = (m as any).content;
    if (typeof content === "string") {
      total += Math.ceil(content.length / 4);
    } else if (Array.isArray(content)) {
      total += Math.ceil(JSON.stringify(content).length / 4);
    } else {
      total += 10; // base overhead per message
    }
  }
  return total;
}

function preprocessMessages(options: {
  messages: BaseMessage[];
  runningSummary?: RunningSummary;
  maxTokens: number;
  maxTokensBeforeSummary?: number;
  maxSummaryTokens: number;
  tokenCounter: TokenCounter;
}): PreprocessedMessages {
  const {
    messages,
    runningSummary,
    maxTokens,
    maxSummaryTokens,
    tokenCounter,
  } = options;
  let { maxTokensBeforeSummary } = options;

  if (maxSummaryTokens >= maxTokens) {
    throw new Error("`maxSummaryTokens` must be less than `maxTokens`.");
  }

  if (maxTokensBeforeSummary === undefined) {
    maxTokensBeforeSummary = maxTokens;
  }

  let maxTokensToSummarize = maxTokens;
  let maxRemainingTokens = maxTokens - maxSummaryTokens;

  let remainingMessages = [...messages];
  let existingSystemMessage: SystemMessage | null = null;

  if (
    remainingMessages.length > 0 &&
    remainingMessages[0] instanceof SystemMessage
  ) {
    existingSystemMessage = remainingMessages[0];
    remainingMessages = remainingMessages.slice(1);
    maxRemainingTokens -= tokenCounter([existingSystemMessage]);
  }

  if (!remainingMessages.length) {
    return {
      messagesToSummarize: [],
      nTokensToSummarize: 0,
      maxTokensToSummarize,
      totalSummarizedMessages: 0,
      existingSystemMessage,
    };
  }

  let summarizedMessageIds = new Set<string>();
  let totalSummarizedMessages = 0;

  if (runningSummary) {
    summarizedMessageIds = runningSummary.summarizedMessageIds;
    maxTokensToSummarize -= tokenCounter([
      new SystemMessage(runningSummary.summary),
    ]);
    for (let i = 0; i < remainingMessages.length; i++) {
      if (
        remainingMessages[i].id === runningSummary.lastSummarizedMessageId
      ) {
        totalSummarizedMessages = i + 1;
        break;
      }
    }
  }

  const totalNTokens = tokenCounter(
    remainingMessages.slice(totalSummarizedMessages)
  );

  let nTokens = 0;
  let idx = Math.max(0, totalSummarizedMessages - 1);
  const toolCallIdToToolMessage = new Map<string, ToolMessage>();
  let shouldSummarize = false;
  let nTokensToSummarize = 0;

  for (let i = totalSummarizedMessages; i < remainingMessages.length; i++) {
    const message = remainingMessages[i];
    if (!message.id) {
      throw new Error("Messages are required to have an ID field.");
    }
    if (summarizedMessageIds.has(message.id)) {
      throw new Error(
        `Message with ID ${message.id} has already been summarized.`
      );
    }
    if (message instanceof ToolMessage && message.tool_call_id) {
      toolCallIdToToolMessage.set(message.tool_call_id, message);
    }

    nTokens += tokenCounter([message]);

    if (
      nTokens >= maxTokensBeforeSummary! &&
      totalNTokens - nTokens <= maxRemainingTokens &&
      !shouldSummarize
    ) {
      nTokensToSummarize = nTokens;
      shouldSummarize = true;
      idx = i;
    }
  }

  let messagesToSummarize: BaseMessage[] = [];
  if (shouldSummarize) {
    messagesToSummarize = remainingMessages.slice(
      totalSummarizedMessages,
      idx + 1
    );
  }

  // Include subsequent tool messages if last message is AI with tool calls
  if (
    messagesToSummarize.length > 0 &&
    messagesToSummarize[messagesToSummarize.length - 1] instanceof AIMessage
  ) {
    const lastMsg = messagesToSummarize[
      messagesToSummarize.length - 1
    ] as AIMessage;
    if (lastMsg.tool_calls && lastMsg.tool_calls.length > 0) {
      for (const toolCall of lastMsg.tool_calls) {
        if (toolCall.id && toolCallIdToToolMessage.has(toolCall.id)) {
          const toolMessage = toolCallIdToToolMessage.get(toolCall.id)!;
          nTokensToSummarize += tokenCounter([toolMessage]);
          messagesToSummarize.push(toolMessage);
        }
      }
    }
  }

  return {
    messagesToSummarize,
    nTokensToSummarize,
    maxTokensToSummarize,
    totalSummarizedMessages,
    existingSystemMessage,
  };
}

function prepareSummarizationResult(options: {
  preprocessedMessages: PreprocessedMessages;
  messages: BaseMessage[];
  existingSummary?: RunningSummary;
  runningSummary?: RunningSummary;
}): SummarizationResult {
  const {
    preprocessedMessages,
    messages,
    existingSummary,
    runningSummary,
  } = options;

  const totalSummarizedMessages =
    preprocessedMessages.totalSummarizedMessages +
    preprocessedMessages.messagesToSummarize.length;

  if (runningSummary) {
    const includeSystemMessage =
      preprocessedMessages.existingSystemMessage &&
      !(
        existingSummary &&
        existingSummary.summary &&
        typeof preprocessedMessages.existingSystemMessage.content ===
          "string" &&
        preprocessedMessages.existingSystemMessage.content.includes(
          existingSummary.summary
        )
      );

    const updatedMessages: BaseMessage[] = [];
    if (includeSystemMessage) {
      updatedMessages.push(preprocessedMessages.existingSystemMessage!);
    }
    updatedMessages.push(
      new SystemMessage(
        `Summary of the conversation so far: ${runningSummary.summary}`
      )
    );
    updatedMessages.push(...messages.slice(totalSummarizedMessages));

    return { runningSummary, messages: updatedMessages };
  } else {
    const msgs = preprocessedMessages.existingSystemMessage
      ? [preprocessedMessages.existingSystemMessage, ...messages]
      : messages;
    return { runningSummary: undefined, messages: msgs };
  }
}

export async function summarizeMessages(
  messages: BaseMessage[],
  options: {
    runningSummary?: RunningSummary;
    model: BaseChatModel;
    maxTokens: number;
    maxTokensBeforeSummary?: number;
    maxSummaryTokens?: number;
    tokenCounter?: TokenCounter;
  }
): Promise<SummarizationResult> {
  const {
    runningSummary,
    model,
    maxTokens,
    maxTokensBeforeSummary,
    maxSummaryTokens = 256,
    tokenCounter = countTokensApproximately,
  } = options;

  const preprocessed = preprocessMessages({
    messages,
    runningSummary,
    maxTokens,
    maxTokensBeforeSummary,
    maxSummaryTokens,
    tokenCounter,
  });

  let remainingMessages = messages;
  if (preprocessed.existingSystemMessage) {
    remainingMessages = messages.slice(1);
  }

  if (!remainingMessages.length) {
    return {
      runningSummary,
      messages: preprocessed.existingSystemMessage
        ? [preprocessed.existingSystemMessage, ...remainingMessages]
        : remainingMessages,
    };
  }

  const existingSummary = runningSummary;
  let newRunningSummary = runningSummary;

  if (preprocessed.messagesToSummarize.length > 0) {
    // Trim messages if they exceed maxTokensToSummarize (to avoid exceeding context window)
    let messagesToSendToModel = preprocessed.messagesToSummarize;
    if (preprocessed.nTokensToSummarize > preprocessed.maxTokensToSummarize) {
      const trimmed = trimMessagesForSummarization(
        preprocessed.messagesToSummarize,
        preprocessed.maxTokensToSummarize,
        tokenCounter
      );
      if (trimmed.length > 0) {
        messagesToSendToModel = trimmed;
      }
    }

    // Build summary prompt
    const summaryMessages: BaseMessage[] = [...messagesToSendToModel];

    if (runningSummary) {
      summaryMessages.push(
        new HumanMessage(
          `This is summary of the conversation so far: ${runningSummary.summary}\n\nExtend this summary by taking into account the new messages above:`
        )
      );
    } else {
      summaryMessages.push(
        new HumanMessage("Create a summary of the conversation above:")
      );
    }

    const summaryResponse = await model.invoke(summaryMessages);
    const summaryContent =
      typeof summaryResponse.content === "string"
        ? summaryResponse.content
        : JSON.stringify(summaryResponse.content);

    const summarizedIds = runningSummary
      ? new Set(runningSummary.summarizedMessageIds)
      : new Set<string>();

    // Track ALL messagesToSummarize IDs (not just the trimmed subset)
    const newSummarizedIds = new Set([
      ...summarizedIds,
      ...preprocessed.messagesToSummarize
        .filter((m) => m.id)
        .map((m) => m.id!),
    ]);

    newRunningSummary = {
      summary: summaryContent,
      summarizedMessageIds: newSummarizedIds,
      lastSummarizedMessageId:
        preprocessed.messagesToSummarize[
          preprocessed.messagesToSummarize.length - 1
        ]?.id ?? null,
    };
  }

  return prepareSummarizationResult({
    preprocessedMessages: preprocessed,
    messages: remainingMessages,
    existingSummary,
    runningSummary: newRunningSummary,
  });
}

export interface SummarizationNodeOptions {
  model: BaseChatModel;
  maxTokens: number;
  maxTokensBeforeSummary?: number;
  maxSummaryTokens?: number;
  tokenCounter?: TokenCounter;
  inputMessagesKey?: string;
  outputMessagesKey?: string;
  name?: string;
}

export class SummarizationNode {
  private options: Required<
    Omit<SummarizationNodeOptions, "maxTokensBeforeSummary">
  > & {
    maxTokensBeforeSummary?: number;
  };

  constructor(options: SummarizationNodeOptions) {
    this.options = {
      model: options.model,
      maxTokens: options.maxTokens,
      maxTokensBeforeSummary: options.maxTokensBeforeSummary,
      maxSummaryTokens: options.maxSummaryTokens ?? 256,
      tokenCounter: options.tokenCounter ?? countTokensApproximately,
      inputMessagesKey: options.inputMessagesKey ?? "messages",
      outputMessagesKey: options.outputMessagesKey ?? "summarized_messages",
      name: options.name ?? "summarization",
    };
  }

  private parseInput(
    input: Record<string, unknown>
  ): [BaseMessage[], Record<string, unknown>] {
    const messages = input[
      this.options.inputMessagesKey
    ] as BaseMessage[];
    const context = (input.context ?? {}) as Record<string, unknown>;
    if (!messages) {
      throw new Error(
        `Missing required field \`${this.options.inputMessagesKey}\` in the input.`
      );
    }
    return [messages, context];
  }

  private prepareStateUpdate(
    context: Record<string, unknown>,
    result: SummarizationResult
  ): Record<string, unknown> {
    const update: Record<string, unknown> = {
      [this.options.outputMessagesKey]: result.messages,
    };
    if (result.runningSummary) {
      update.context = {
        ...context,
        running_summary: result.runningSummary,
      };
      // When input and output keys are the same, prepend a RemoveMessage to
      // clear the existing messages before writing the summarized ones
      if (this.options.inputMessagesKey === this.options.outputMessagesKey) {
        update[this.options.outputMessagesKey] = [
          new RemoveMessage({ id: REMOVE_ALL_MESSAGES }),
          ...(update[this.options.outputMessagesKey] as BaseMessage[]),
        ];
      }
    }
    return update;
  }

  async invoke(
    input: Record<string, unknown>,
    _config?: RunnableConfig
  ): Promise<Record<string, unknown>> {
    const [messages, context] = this.parseInput(input);
    const result = await summarizeMessages(messages, {
      runningSummary: (context.running_summary as RunningSummary) ?? undefined,
      model: this.options.model,
      maxTokens: this.options.maxTokens,
      maxTokensBeforeSummary: this.options.maxTokensBeforeSummary,
      maxSummaryTokens: this.options.maxSummaryTokens,
      tokenCounter: this.options.tokenCounter,
    });
    return this.prepareStateUpdate(context, result);
  }
}
