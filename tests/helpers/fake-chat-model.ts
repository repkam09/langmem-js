import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";

/**
 * A fake chat model for testing that returns predefined responses in sequence
 * and tracks all messages passed to it.
 */
export class FakeChatModel extends BaseChatModel {
  invokeCalls: BaseMessage[][] = [];
  private responses: AIMessage[];
  private callIndex = 0;

  constructor(responses: AIMessage[]) {
    super({});
    this.responses =
      responses.length > 0
        ? responses
        : [new AIMessage("This is a mock summary.")];
  }

  _llmType(): string {
    return "fake";
  }

  async _generate(
    messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    this.invokeCalls.push(messages);
    const response = this.responses[this.callIndex % this.responses.length];
    this.callIndex++;
    return {
      generations: [
        {
          message: response,
          text:
            typeof response.content === "string" ? response.content : "",
        },
      ],
    };
  }
}
