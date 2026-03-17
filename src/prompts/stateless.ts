import { z } from "zod";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { BaseMessage } from "@langchain/core/messages";
import {
  INSTRUCTION_REFLECTION_PROMPT,
  INSTRUCTION_REFLECTION_MULTIPLE_PROMPT,
} from "./prompt.js";
import { getTrajectoryClean } from "./utils.js";
import { getVarHealer } from "../utils.js";
import type { OptimizerInput } from "./types.js";

const GeneralResponseSchema = z.object({
  logic: z.string(),
  update_prompt: z.boolean(),
  new_prompt: z.string(),
});

type GeneralResponse = z.infer<typeof GeneralResponseSchema>;

function formatPromptTemplate(
  template: string,
  vars: Record<string, string>
): string {
  let result = template;
  for (const [key, value] of Object.entries(vars)) {
    result = result.replace(new RegExp(`\\{${key}\\}`, "g"), value);
  }
  return result;
}

export class PromptMemory {
  private model: ReturnType<BaseChatModel["withStructuredOutput"]>;

  constructor(model: BaseChatModel) {
    this.model = model.withStructuredOutput(GeneralResponseSchema, {
      method: "jsonSchema",
    });
  }

  async invoke(
    input: {
      messages: (BaseMessage | Record<string, unknown>)[];
      current_prompt?: string;
      feedback?: string;
      instructions?: string;
    },
    _config?: RunnableConfig
  ): Promise<string> {
    const {
      messages,
      current_prompt = "",
      feedback = "",
      instructions = "",
    } = input;
    const trajectory = getTrajectoryClean(messages);
    const promptStr = formatPromptTemplate(INSTRUCTION_REFLECTION_PROMPT, {
      current_prompt,
      trajectory,
      feedback,
      instructions,
    });
    const output = (await this.model.invoke(promptStr)) as unknown as GeneralResponse;
    return output.new_prompt;
  }
}

export class PromptMemoryMultiple {
  private model: ReturnType<BaseChatModel["withStructuredOutput"]>;

  constructor(model: BaseChatModel) {
    this.model = model.withStructuredOutput(GeneralResponseSchema, {
      method: "jsonSchema",
    });
  }

  private static getData(
    trajectories: string | Array<[unknown, string]>
  ): string {
    if (typeof trajectories === "string") return trajectories;
    const pieces: string[] = [];
    for (let i = 0; i < trajectories.length; i++) {
      const [messages, feedback] = trajectories[i];
      const trajectory = getTrajectoryClean(messages as any);
      pieces.push(
        `<trajectory ${i}>\n${trajectory}\n</trajectory ${i}>\n<feedback ${i}>\n${feedback}\n</feedback ${i}>`
      );
    }
    return pieces.join("\n");
  }

  async invoke(
    input: OptimizerInput,
    _config?: RunnableConfig
  ): Promise<string> {
    const { trajectories, prompt: promptData } = input;
    const promptStr =
      typeof promptData === "string" ? promptData : promptData.prompt;
    const updateInstructions =
      typeof promptData === "string"
        ? ""
        : promptData.update_instructions ?? "";

    const dataStr = PromptMemoryMultiple.getData(trajectories as any);
    const healer = getVarHealer(promptStr);

    const promptInput = formatPromptTemplate(
      INSTRUCTION_REFLECTION_MULTIPLE_PROMPT,
      {
        current_prompt: promptStr,
        data: dataStr,
        instructions: updateInstructions ?? "",
      }
    );

    const output = (await this.model.invoke(promptInput)) as unknown as GeneralResponse;
    return healer(output.new_prompt);
  }
}
