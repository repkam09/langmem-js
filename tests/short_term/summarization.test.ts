import { describe, it, expect } from "vitest";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  RemoveMessage,
} from "@langchain/core/messages";
import {
  summarizeMessages,
  SummarizationNode,
  countTokensApproximately,
} from "../../src/short_term/summarization";
import { FakeChatModel } from "../helpers/fake-chat-model";

// Simple token counter that counts messages (equivalent to Python's `len`)
const lenCounter = (msgs: Parameters<typeof countTokensApproximately>[0]) =>
  msgs.length;

describe("summarizeMessages", () => {
  it("test_empty_input", async () => {
    const model = new FakeChatModel([]);

    // Empty message list
    let result = await summarizeMessages([], {
      runningSummary: undefined,
      model,
      maxTokens: 10,
      maxSummaryTokens: 1,
    });

    expect(result.runningSummary).toBeUndefined();
    expect(result.messages).toEqual([]);
    expect(model.invokeCalls.length).toBe(0);

    // Only system message
    const systemMsg = new SystemMessage({ content: "You are a helpful assistant.", id: "sys" });
    result = await summarizeMessages([systemMsg], {
      runningSummary: undefined,
      model,
      maxTokens: 10,
      maxSummaryTokens: 1,
    });

    expect(result.runningSummary).toBeUndefined();
    expect(result.messages).toEqual([systemMsg]);
  });

  it("test_no_summarization_needed", async () => {
    const model = new FakeChatModel([]);

    const messages = [
      new HumanMessage({ content: "Message 1", id: "1" }),
      new AIMessage({ content: "Response 1", id: "2" }),
      new HumanMessage({ content: "Message 2", id: "3" }),
    ];

    // 3 tokens total, under the limit of 10
    const result = await summarizeMessages(messages, {
      runningSummary: undefined,
      model,
      tokenCounter: lenCounter,
      maxTokens: 10,
      maxSummaryTokens: 1,
    });

    expect(result.runningSummary).toBeUndefined();
    expect(result.messages).toEqual(messages);
    expect(model.invokeCalls.length).toBe(0);
  });

  it("test_summarize_first_time", async () => {
    const model = new FakeChatModel([
      new AIMessage("This is a summary of the conversation."),
    ]);

    const messages = [
      // these messages will be summarized
      new HumanMessage({ content: "Message 1", id: "1" }),
      new AIMessage({ content: "Response 1", id: "2" }),
      new HumanMessage({ content: "Message 2", id: "3" }),
      new AIMessage({ content: "Response 2", id: "4" }),
      new HumanMessage({ content: "Message 3", id: "5" }),
      new AIMessage({ content: "Response 3", id: "6" }),
      // these messages will be added to the result post-summarization
      new HumanMessage({ content: "Message 4", id: "7" }),
      new AIMessage({ content: "Response 4", id: "8" }),
      new HumanMessage({ content: "Latest message", id: "9" }),
    ];

    const maxSummaryTokens = 1;
    const result = await summarizeMessages(messages, {
      runningSummary: undefined,
      model,
      tokenCounter: lenCounter,
      maxTokens: 6,
      maxSummaryTokens,
    });

    expect(model.invokeCalls.length).toBe(1);
    expect(result.messages.length).toBe(4);
    expect(result.messages[0].getType()).toBe("system");
    expect(result.messages[0].content.toString().toLowerCase()).toContain("summary");
    expect(result.messages.slice(1)).toEqual(messages.slice(-3));

    const summaryValue = result.runningSummary!;
    expect(summaryValue).toBeDefined();
    expect(summaryValue.summary).toBe("This is a summary of the conversation.");
    expect(summaryValue.summarizedMessageIds).toEqual(
      new Set(messages.slice(0, 6).map((m) => m.id))
    );

    // Subsequent invocation — no new summary needed
    const result2 = await summarizeMessages(messages, {
      runningSummary: summaryValue,
      model,
      tokenCounter: lenCounter,
      maxTokens: 6,
      maxSummaryTokens,
    });
    expect(result2.messages.length).toBe(4);
    expect(result2.messages[0].getType()).toBe("system");
    expect(result2.messages[0].content).toBe(
      "Summary of the conversation so far: This is a summary of the conversation."
    );
    expect(result2.messages.slice(1)).toEqual(messages.slice(-3));
  });

  it("test_max_tokens_before_summary", async () => {
    const model = new FakeChatModel([
      new AIMessage("This is a summary of the conversation."),
    ]);

    const messages = [
      // these messages will be summarized
      new HumanMessage({ content: "Message 1", id: "1" }),
      new AIMessage({ content: "Response 1", id: "2" }),
      new HumanMessage({ content: "Message 2", id: "3" }),
      new AIMessage({ content: "Response 2", id: "4" }),
      new HumanMessage({ content: "Message 3", id: "5" }),
      new AIMessage({ content: "Response 3", id: "6" }),
      new HumanMessage({ content: "Message 4", id: "7" }),
      new AIMessage({ content: "Response 4", id: "8" }),
      // this message will be kept in the result post-summarization
      new HumanMessage({ content: "Latest message", id: "9" }),
    ];

    const maxSummaryTokens = 1;
    const result = await summarizeMessages(messages, {
      runningSummary: undefined,
      model,
      tokenCounter: lenCounter,
      maxTokens: 6,
      maxTokensBeforeSummary: 8,
      maxSummaryTokens,
    });

    expect(model.invokeCalls.length).toBe(1);
    expect(result.messages.length).toBe(2);
    expect(result.messages[0].getType()).toBe("system");
    expect(result.messages[0].content.toString().toLowerCase()).toContain("summary");
    expect(result.messages.slice(1)).toEqual(messages.slice(-1));

    const summaryValue = result.runningSummary!;
    expect(summaryValue).toBeDefined();
    expect(summaryValue.summary).toBe("This is a summary of the conversation.");
    expect(summaryValue.summarizedMessageIds).toEqual(
      new Set(messages.slice(0, 8).map((m) => m.id))
    );

    // Subsequent invocation — no new summary needed
    const result2 = await summarizeMessages(messages, {
      runningSummary: summaryValue,
      model,
      tokenCounter: lenCounter,
      maxTokens: 6,
      maxTokensBeforeSummary: 8,
      maxSummaryTokens,
    });
    expect(result2.messages.length).toBe(2);
    expect(result2.messages[0].content).toBe(
      "Summary of the conversation so far: This is a summary of the conversation."
    );
    expect(result2.messages.slice(1)).toEqual(messages.slice(-1));
  });

  it("test_with_system_message", async () => {
    const model = new FakeChatModel([
      new AIMessage("Summary with system message present."),
    ]);

    const messages = [
      // not summarized — preserved post-summarization
      new SystemMessage({ content: "You are a helpful assistant.", id: "0" }),
      // these will be summarized
      new HumanMessage({ content: "Message 1", id: "1" }),
      new AIMessage({ content: "Response 1", id: "2" }),
      new HumanMessage({ content: "Message 2", id: "3" }),
      new AIMessage({ content: "Response 2", id: "4" }),
      new HumanMessage({ content: "Message 3", id: "5" }),
      new AIMessage({ content: "Response 3", id: "6" }),
      // these will be kept
      new HumanMessage({ content: "Message 4", id: "7" }),
      new AIMessage({ content: "Response 4", id: "8" }),
      new HumanMessage({ content: "Latest message", id: "9" }),
    ];

    const result = await summarizeMessages(messages, {
      runningSummary: undefined,
      model,
      tokenCounter: lenCounter,
      maxTokens: 6,
      maxSummaryTokens: 1,
    });

    expect(model.invokeCalls.length).toBe(1);
    expect(model.invokeCalls[0]).toEqual([
      ...messages.slice(1, 7),
      new HumanMessage({ content: "Create a summary of the conversation above:" }),
    ]);

    expect(result.messages.length).toBe(5);
    expect(result.messages[0].getType()).toBe("system"); // original system message
    expect(result.messages[1].getType()).toBe("system"); // summary
    expect(result.messages[1].content.toString().toLowerCase()).toContain("summary");
    expect(result.messages.slice(2)).toEqual(messages.slice(-3));
  });

  it("test_approximate_token_counter", async () => {
    // countTokensApproximately: ceil(content.length / 4)
    // Token counts for these messages (TS implementation):
    //   "" → 0, "Response 1" → 3, "Message 2" → 3, "" → 0,
    //   "Message 3" → 3, "Response 3" → 3, "Message 4" → 3,
    //   "Response 4" → 3, "Latest message" → 4
    // Total = 22 tokens
    // With maxTokens=16 (defaults to maxTokensBeforeSummary), maxSummaryTokens=3:
    //   maxRemainingTokens=13; condition triggers at i=7 (nTokens=18>=16, 4<=13)
    //   messagesToSummarize = messages[0..7] (8 messages), kept = messages[8]
    const model = new FakeChatModel([
      new AIMessage("Summary with empty messages."),
    ]);

    const messages = [
      new HumanMessage({ content: "", id: "1" }),
      new AIMessage({ content: "Response 1", id: "2" }),
      new HumanMessage({ content: "Message 2", id: "3" }),
      new AIMessage({ content: "", id: "4" }),
      new HumanMessage({ content: "Message 3", id: "5" }),
      new AIMessage({ content: "Response 3", id: "6" }),
      new HumanMessage({ content: "Message 4", id: "7" }),
      new AIMessage({ content: "Response 4", id: "8" }),
      new HumanMessage({ content: "Latest message", id: "9" }),
    ];

    const result = await summarizeMessages(messages, {
      runningSummary: undefined,
      model,
      tokenCounter: countTokensApproximately,
      maxTokens: 16,
      maxSummaryTokens: 3,
    });

    expect(result.messages.length).toBe(2);
    expect(result.messages[0].getType()).toBe("system");
    expect(result.messages[0].content.toString().toLowerCase()).toContain("summary");
    expect(result.messages.slice(-1)).toEqual(messages.slice(-1));
  });

  it("test_large_number_of_messages", async () => {
    const model = new FakeChatModel([
      new AIMessage("Summary of many messages."),
    ]);

    const messages = [];
    for (let i = 0; i < 20; i++) {
      messages.push(new HumanMessage({ content: `Human message ${i}`, id: `h${i}` }));
      messages.push(new AIMessage({ content: `AI response ${i}`, id: `a${i}` }));
    }
    messages.push(new HumanMessage({ content: "Final message", id: `h${messages.length}` }));

    const result = await summarizeMessages(messages, {
      runningSummary: undefined,
      model,
      tokenCounter: lenCounter,
      maxTokens: 22,
      maxSummaryTokens: 0,
    });

    // summary (for the first 22 messages) + 19 remaining
    expect(result.messages.length).toBe(20);
    expect(result.messages[0].getType()).toBe("system");
    expect(result.messages[0].content.toString().toLowerCase()).toContain("summary");
    expect(result.messages.slice(1)).toEqual(messages.slice(22));
    expect(model.invokeCalls.length).toBe(1);
  });

  it("test_subsequent_summarization_with_new_messages", async () => {
    const model = new FakeChatModel([
      new AIMessage("First summary of the conversation."),
      new AIMessage("Updated summary including new messages."),
    ]);

    const messages1 = [
      // these will be summarized
      new HumanMessage({ content: "Message 1", id: "1" }),
      new AIMessage({ content: "Response 1", id: "2" }),
      new HumanMessage({ content: "Message 2", id: "3" }),
      new AIMessage({ content: "Response 2", id: "4" }),
      new HumanMessage({ content: "Message 3", id: "5" }),
      new AIMessage({ content: "Response 3", id: "6" }),
      // this will be propagated to the next summarization
      new HumanMessage({ content: "Latest message 1", id: "7" }),
    ];

    const maxTokens = 6;
    const maxSummaryTokens = 1;
    const result = await summarizeMessages(messages1, {
      runningSummary: undefined,
      model,
      tokenCounter: lenCounter,
      maxTokens,
      maxSummaryTokens,
    });

    expect(result.messages[0].getType()).toBe("system");
    expect(result.messages[0].content.toString().toLowerCase()).toContain("summary");
    expect(result.messages.length).toBe(2);
    expect(result.messages[result.messages.length - 1]).toEqual(messages1[messages1.length - 1]);
    expect(model.invokeCalls.length).toBe(1);

    const summaryValue = result.runningSummary!;
    expect(summaryValue.summary).toBe("First summary of the conversation.");
    expect(summaryValue.summarizedMessageIds.size).toBe(6);

    const newMessages = [
      new AIMessage({ content: "Response to latest 1", id: "8" }),
      new HumanMessage({ content: "Message 4", id: "9" }),
      new AIMessage({ content: "Response 4", id: "10" }),
      new HumanMessage({ content: "Message 5", id: "11" }),
      new AIMessage({ content: "Response 5", id: "12" }),
      // these will be kept in the final result
      new HumanMessage({ content: "Message 6", id: "13" }),
      new AIMessage({ content: "Response 6", id: "14" }),
      new HumanMessage({ content: "Latest message 2", id: "15" }),
    ];

    const messages2 = [...messages1, ...newMessages];

    const result2 = await summarizeMessages(messages2, {
      runningSummary: summaryValue,
      model,
      tokenCounter: lenCounter,
      maxTokens,
      maxSummaryTokens,
    });

    expect(model.invokeCalls.length).toBe(2);

    const secondCallMessages = model.invokeCalls[1];
    const promptMessage = secondCallMessages[secondCallMessages.length - 1];
    expect(promptMessage.content).toContain("First summary of the conversation");
    expect(promptMessage.content).toContain("Extend this summary");

    // Only new (non-summarized) messages are sent to model: 4 messages + prompt
    expect(secondCallMessages.length).toBe(5);
    expect(secondCallMessages.slice(0, -1).map((m) => m.content)).toEqual([
      "Message 4",
      "Response 4",
      "Message 5",
      "Response 5",
    ]);

    expect(result2.messages[0].getType()).toBe("system");
    expect(result2.messages[0].content.toString().toLowerCase()).toContain("summary");
    expect(result2.messages.length).toBe(4);
    expect(result2.messages.slice(-3)).toEqual(messages2.slice(-3));

    const updatedSummary = result2.runningSummary!;
    expect(updatedSummary.summary).toBe("Updated summary including new messages.");
    expect(updatedSummary.summarizedMessageIds.size).toBe(messages2.length - 3);
  });

  it("test_subsequent_summarization_with_new_messages_approximate_token_counter", async () => {
    // Token counts with countTokensApproximately (ceil(len/4)):
    //   "Message N" → 3, "Response N" → 3, "Latest message 1" → 4
    // messages1 total: 3*6 + 4 = 22 tokens
    // maxTokens=20, maxTokensBeforeSummary=18, maxSummaryTokens=5
    //   maxRemainingTokens=15; triggers at i=5 (nTokens=18>=18, 4<=15)
    //   messagesToSummarize = messages1[0..5], no trimming (18 ≤ 20)
    //
    // Second pass: "First summary of the conversation." → 9 tokens
    //   maxTokensToSummarize = 20-9 = 11
    //   triggers at i=10 (nTokens=18>=18, 13<=15)
    //   messagesToSummarize = 5 messages, nTokensToSummarize=18>11 → trimming
    //   trimmed to last 11 tokens: [HumanMessage("Message 4"), AIMessage("Response 4"), HumanMessage("Message 5")]
    const model = new FakeChatModel([
      new AIMessage("First summary of the conversation."),
      new AIMessage("Updated summary including new messages."),
    ]);

    const messages1 = [
      new HumanMessage({ content: "Message 1", id: "1" }),
      new AIMessage({ content: "Response 1", id: "2" }),
      new HumanMessage({ content: "Message 2", id: "3" }),
      new AIMessage({ content: "Response 2", id: "4" }),
      new HumanMessage({ content: "Message 3", id: "5" }),
      new AIMessage({ content: "Response 3", id: "6" }),
      new HumanMessage({ content: "Latest message 1", id: "7" }),
    ];

    const maxTokens = 20;
    const maxTokensBeforeSummary = 18;
    const maxSummaryTokens = 5;

    const result = await summarizeMessages(messages1, {
      runningSummary: undefined,
      model,
      tokenCounter: countTokensApproximately,
      maxTokens,
      maxTokensBeforeSummary,
      maxSummaryTokens,
    });

    expect(result.messages[0].getType()).toBe("system");
    expect(result.messages[0].content.toString().toLowerCase()).toContain("summary");
    expect(result.messages.length).toBe(2);
    expect(result.messages[result.messages.length - 1]).toEqual(messages1[messages1.length - 1]);
    expect(model.invokeCalls.length).toBe(1);

    const summaryValue = result.runningSummary!;
    expect(summaryValue.summary).toBe("First summary of the conversation.");
    expect(summaryValue.summarizedMessageIds.size).toBe(6);

    const newMessages = [
      new AIMessage({ content: "Response to latest 1", id: "8" }),
      new HumanMessage({ content: "Message 4", id: "9" }),
      new AIMessage({ content: "Response 4", id: "10" }),
      new HumanMessage({ content: "Message 5", id: "11" }),
      new AIMessage({ content: "Response 5", id: "12" }),
      new HumanMessage({ content: "Message 6", id: "13" }),
      new AIMessage({ content: "Response 6", id: "14" }),
      new HumanMessage({ content: "Latest message 2", id: "15" }),
    ];

    const messages2 = [...messages1, ...newMessages];

    const result2 = await summarizeMessages(messages2, {
      runningSummary: summaryValue,
      model,
      tokenCounter: countTokensApproximately,
      maxTokens,
      maxTokensBeforeSummary,
      maxSummaryTokens,
    });

    expect(model.invokeCalls.length).toBe(2);

    const secondCallMessages = model.invokeCalls[1];
    const promptMessage = secondCallMessages[secondCallMessages.length - 1];
    expect(promptMessage.content).toContain("First summary of the conversation");
    expect(promptMessage.content).toContain("Extend this summary");

    // Due to trimming, 3 messages from the non-summarized window are sent
    expect(secondCallMessages.length).toBe(4); // 3 trimmed messages + prompt
    expect(secondCallMessages.slice(0, -1).map((m) => m.content)).toEqual([
      "Message 4",
      "Response 4",
      "Message 5",
    ]);

    expect(result2.messages[0].getType()).toBe("system");
    expect(result2.messages[0].content.toString().toLowerCase()).toContain("summary");
    // 5 messages: summary + 4 remaining (messages2[11..14])
    expect(result2.messages.length).toBe(5);
    expect(result2.messages.slice(-4)).toEqual(messages2.slice(-4));

    const updatedSummary = result2.runningSummary!;
    expect(updatedSummary.summary).toBe("Updated summary including new messages.");
    // 6 from first pass + 5 newly summarized = 11
    expect(updatedSummary.summarizedMessageIds.size).toBe(11);
  });

  it("test_last_ai_with_tool_calls", async () => {
    const model = new FakeChatModel([
      new AIMessage("Summary without tool calls."),
    ]);

    const messages = [
      // these will be summarized
      new HumanMessage({ content: "Message 1", id: "1" }),
      new AIMessage({
        content: "",
        id: "2",
        tool_calls: [
          { name: "tool_1", id: "1", args: { arg1: "value1" } },
          { name: "tool_2", id: "2", args: { arg1: "value1" } },
        ],
      }),
      new ToolMessage({ content: "Call tool 1", tool_call_id: "1", name: "tool_1", id: "3" }),
      new ToolMessage({ content: "Call tool 2", tool_call_id: "2", name: "tool_2", id: "4" }),
      // these will be kept in the final result
      new AIMessage({ content: "Response 1", id: "5" }),
      new HumanMessage({ content: "Message 2", id: "6" }),
    ];

    const result = await summarizeMessages(messages, {
      runningSummary: undefined,
      model,
      tokenCounter: lenCounter,
      maxTokensBeforeSummary: 2,
      maxTokens: 6,
      maxSummaryTokens: 1,
    });

    // AI message with tool calls was summarized together with the tool messages
    expect(result.messages.length).toBe(3);
    expect(result.messages[0].getType()).toBe("system");
    expect(result.messages.slice(-2)).toEqual(messages.slice(-2));
    expect(result.runningSummary!.summarizedMessageIds).toEqual(
      new Set(messages.slice(0, -2).map((m) => m.id))
    );
  });

  it("test_missing_message_ids", async () => {
    const messages = [
      new HumanMessage({ content: "Message 1", id: "1" }),
      new AIMessage({ content: "Response" }), // No ID
    ];
    await expect(
      summarizeMessages(messages, {
        runningSummary: undefined,
        model: new FakeChatModel([]),
        maxTokens: 10,
        maxSummaryTokens: 1,
      })
    ).rejects.toThrow("Messages are required to have an ID field.");
  });

  it("test_duplicate_message_ids", async () => {
    const model = new FakeChatModel([new AIMessage("Summary")]);

    const messages1 = [
      new HumanMessage({ content: "Message 1", id: "1" }),
      new AIMessage({ content: "Response 1", id: "2" }),
      new HumanMessage({ content: "Message 2", id: "3" }),
    ];

    const result = await summarizeMessages(messages1, {
      runningSummary: undefined,
      model,
      tokenCounter: lenCounter,
      maxTokens: 2,
      maxSummaryTokens: 1,
    });

    const messages2 = [
      new AIMessage({ content: "Response 2", id: "4" }),
      new HumanMessage({ content: "Message 3", id: "1" }), // Duplicate ID
    ];

    await expect(
      summarizeMessages([...messages1, ...messages2], {
        runningSummary: result.runningSummary,
        model,
        tokenCounter: lenCounter,
        maxTokens: 5,
        maxSummaryTokens: 1,
      })
    ).rejects.toThrow("has already been summarized");
  });

  it("test_summarization_updated_messages", async () => {
    // Variant of test_subsequent_summarization_with_new_messages that uses
    // the updated (summarized) messages on the second turn
    const model = new FakeChatModel([
      new AIMessage("First summary of the conversation."),
      new AIMessage("Updated summary including new messages."),
    ]);

    const messages1 = [
      new HumanMessage({ content: "Message 1", id: "1" }),
      new AIMessage({ content: "Response 1", id: "2" }),
      new HumanMessage({ content: "Message 2", id: "3" }),
      new AIMessage({ content: "Response 2", id: "4" }),
      new HumanMessage({ content: "Message 3", id: "5" }),
      new AIMessage({ content: "Response 3", id: "6" }),
      new HumanMessage({ content: "Latest message 1", id: "7" }),
    ];

    const maxTokens = 6;
    const maxSummaryTokens = 1;
    const result = await summarizeMessages(messages1, {
      runningSummary: undefined,
      model,
      tokenCounter: lenCounter,
      maxTokens,
      maxSummaryTokens,
    });

    expect(result.messages[0].getType()).toBe("system");
    expect(result.messages.length).toBe(2);
    expect(result.messages[result.messages.length - 1]).toEqual(messages1[messages1.length - 1]);
    expect(model.invokeCalls.length).toBe(1);

    const summaryValue = result.runningSummary!;
    expect(summaryValue.summary).toBe("First summary of the conversation.");
    expect(summaryValue.summarizedMessageIds.size).toBe(6);

    const newMessages = [
      new AIMessage({ content: "Response to latest 1", id: "8" }),
      new HumanMessage({ content: "Message 4", id: "9" }),
      new AIMessage({ content: "Response 4", id: "10" }),
      new HumanMessage({ content: "Message 5", id: "11" }),
      new AIMessage({ content: "Response 5", id: "12" }),
      new HumanMessage({ content: "Message 6", id: "13" }),
      new AIMessage({ content: "Response 6", id: "14" }),
      new HumanMessage({ content: "Latest message 2", id: "15" }),
    ];

    // NOTE: use the updated messages (with summary), not the originals
    const messages2 = [...result.messages, ...newMessages];

    const result2 = await summarizeMessages(messages2, {
      runningSummary: summaryValue,
      model,
      tokenCounter: lenCounter,
      maxTokens,
      maxSummaryTokens,
    });

    expect(model.invokeCalls.length).toBe(2);

    const secondCallMessages = model.invokeCalls[1];
    const promptMessage = secondCallMessages[secondCallMessages.length - 1];
    expect(promptMessage.content).toContain("First summary of the conversation");
    expect(promptMessage.content).toContain("Extend this summary");

    expect(secondCallMessages.length).toBe(5); // 4 messages + prompt
    expect(secondCallMessages.slice(0, -1).map((m) => m.content)).toEqual([
      "Message 4",
      "Response 4",
      "Message 5",
      "Response 5",
    ]);

    expect(result2.messages[0].getType()).toBe("system");
    expect(result2.messages[0].content.toString().toLowerCase()).toContain("summary");
    expect(result2.messages.length).toBe(4);
    expect(result2.messages.slice(-3)).toEqual(messages2.slice(-3));

    const updatedSummary = result2.runningSummary!;
    expect(updatedSummary.summary).toBe("Updated summary including new messages.");
    expect(updatedSummary.summarizedMessageIds.size).toBe(12);
  });
});

describe("SummarizationNode", () => {
  it("test_summarization_node", async () => {
    const model = new FakeChatModel([
      new AIMessage("This is a summary of the conversation."),
    ]);

    const messages = [
      new HumanMessage({ content: "Message 1", id: "1" }),
      new AIMessage({ content: "Response 1", id: "2" }),
      new HumanMessage({ content: "Message 2", id: "3" }),
      new AIMessage({ content: "Response 2", id: "4" }),
      new HumanMessage({ content: "Message 3", id: "5" }),
      new AIMessage({ content: "Response 3", id: "6" }),
      new HumanMessage({ content: "Message 4", id: "7" }),
      new AIMessage({ content: "Response 4", id: "8" }),
      new HumanMessage({ content: "Latest message", id: "9" }),
    ];

    const maxSummaryTokens = 1;
    const node = new SummarizationNode({
      model,
      tokenCounter: lenCounter,
      maxTokens: 6,
      maxSummaryTokens,
    });

    const result = await node.invoke({ messages });

    expect(model.invokeCalls.length).toBe(1);
    expect(result["summarized_messages"].length).toBe(4);
    expect((result["summarized_messages"] as AIMessage[])[0].getType()).toBe("system");
    expect(
      (result["summarized_messages"] as AIMessage[])[0].content
        .toString()
        .toLowerCase()
    ).toContain("summary");
    expect((result["summarized_messages"] as AIMessage[]).slice(1)).toEqual(
      messages.slice(-3)
    );

    const summaryValue = (result["context"] as Record<string, unknown>)["running_summary"] as Parameters<typeof summarizeMessages>[1]["runningSummary"];
    expect(summaryValue).toBeDefined();
    expect(summaryValue!.summary).toBe("This is a summary of the conversation.");
    expect(summaryValue!.summarizedMessageIds).toEqual(
      new Set(messages.slice(0, 6).map((m) => m.id))
    );

    // Subsequent invocation — no new summary needed
    const result2 = await node.invoke({
      messages,
      context: { running_summary: summaryValue },
    });
    expect((result2["summarized_messages"] as AIMessage[]).length).toBe(4);
    expect((result2["summarized_messages"] as AIMessage[])[0].getType()).toBe("system");
    expect((result2["summarized_messages"] as AIMessage[])[0].content).toBe(
      "Summary of the conversation so far: This is a summary of the conversation."
    );
    expect((result2["summarized_messages"] as AIMessage[]).slice(1)).toEqual(
      messages.slice(-3)
    );
  });

  it("test_summarization_node_same_key", async () => {
    // Variant using the same key for input and output messages (realistic LangGraph use)
    const model = new FakeChatModel([
      new AIMessage("First summary of the conversation."),
      new AIMessage("Updated summary including new messages."),
    ]);

    const messages1 = [
      new HumanMessage({ content: "Message 1", id: "1" }),
      new AIMessage({ content: "Response 1", id: "2" }),
      new HumanMessage({ content: "Message 2", id: "3" }),
      new AIMessage({ content: "Response 2", id: "4" }),
      new HumanMessage({ content: "Message 3", id: "5" }),
      new AIMessage({ content: "Response 3", id: "6" }),
      new HumanMessage({ content: "Latest message 1", id: "7" }),
    ];

    const maxTokens = 6;
    const maxSummaryTokens = 1;
    const node = new SummarizationNode({
      model,
      tokenCounter: lenCounter,
      maxTokens,
      maxSummaryTokens,
      inputMessagesKey: "messages",
      outputMessagesKey: "messages",
    });

    const result = await node.invoke({ messages: messages1 });
    const resultMessages = result["messages"] as import("@langchain/core/messages").BaseMessage[];

    // When same key is used, first message must be RemoveMessage sentinel
    expect(resultMessages[0]).toBeInstanceOf(RemoveMessage);
    expect(resultMessages[1].getType()).toBe("system");
    expect(resultMessages[1].content.toString().toLowerCase()).toContain("summary");
    expect(resultMessages.length).toBe(3); // RemoveMessage + summary + messages1[6]
    expect(resultMessages[resultMessages.length - 1]).toEqual(messages1[messages1.length - 1]);
    expect(model.invokeCalls.length).toBe(1);

    const summaryValue = (result["context"] as Record<string, unknown>)["running_summary"] as Parameters<typeof summarizeMessages>[1]["runningSummary"];
    expect(summaryValue!.summary).toBe("First summary of the conversation.");
    expect(summaryValue!.summarizedMessageIds.size).toBe(6);

    const newMessages = [
      new AIMessage({ content: "Response to latest 1", id: "8" }),
      new HumanMessage({ content: "Message 4", id: "9" }),
      new AIMessage({ content: "Response 4", id: "10" }),
      new HumanMessage({ content: "Message 5", id: "11" }),
      new AIMessage({ content: "Response 5", id: "12" }),
      new HumanMessage({ content: "Message 6", id: "13" }),
      new AIMessage({ content: "Response 6", id: "14" }),
      new HumanMessage({ content: "Latest message 2", id: "15" }),
    ];

    // Use updated messages (skip RemoveMessage sentinel) for next turn
    const messages2 = [...resultMessages.slice(1), ...newMessages];

    const result2 = await node.invoke({
      messages: messages2,
      context: { running_summary: summaryValue },
    });
    const result2Messages = result2["messages"] as import("@langchain/core/messages").BaseMessage[];

    expect(model.invokeCalls.length).toBe(2);

    const secondCallMessages = model.invokeCalls[1];
    const promptMessage = secondCallMessages[secondCallMessages.length - 1];
    expect(promptMessage.content).toContain("First summary of the conversation");
    expect(promptMessage.content).toContain("Extend this summary");

    expect(secondCallMessages.length).toBe(5); // 4 messages + prompt
    expect(secondCallMessages.slice(0, -1).map((m) => m.content)).toEqual([
      "Message 4",
      "Response 4",
      "Message 5",
      "Response 5",
    ]);

    expect(result2Messages[0]).toBeInstanceOf(RemoveMessage);
    expect(result2Messages[1].getType()).toBe("system");
    expect(result2Messages[1].content.toString().toLowerCase()).toContain("summary");
    // RemoveMessage + summary + last 3 messages = 5
    expect(result2Messages.length).toBe(5);
    expect(result2Messages.slice(-3)).toEqual(messages2.slice(-3));

    const updatedSummary = (result2["context"] as Record<string, unknown>)["running_summary"] as Parameters<typeof summarizeMessages>[1]["runningSummary"];
    expect(updatedSummary!.summary).toBe("Updated summary including new messages.");
    expect(updatedSummary!.summarizedMessageIds.size).toBe(12);
  });
});
