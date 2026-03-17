import { z } from "zod";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { formatSessions, getPromptExtractionSchema } from "../utils.js";
import type { OptimizerInput } from "./types.js";

export const DEFAULT_MAX_REFLECTION_STEPS = 5;
export const DEFAULT_MIN_REFLECTION_STEPS = 1;

export const DEFAULT_METAPROMPT = `You are helping an AI assistant learn by optimizing its prompt.

## Background

Below is the current prompt:

<current_prompt>
{prompt}
</current_prompt>

The developer provided these instructions regarding when/how to update:

<update_instructions>
{update_instructions}
</update_instructions>

## Session Data
Analyze the session(s) (and any user feedback) below:

<trajectories>
{trajectories}
</trajectories>

## Instructions

1. Reflect on the agent's performance on the given session(s) and identify any real failure modes (e.g., style mismatch, unclear or incomplete instructions, flawed reasoning, etc.).
2. Recommend the minimal changes necessary to address any real failures. If the prompt performs perfectly, simply respond with the original prompt without making any changes.
3. Retain any template variables in the existing prompt exactly as they are (e.g. {variable_name}).

IFF changes are warranted, focus on actionable edits. Be concrete. Edits should be appropriate for the identified failure modes. For example, consider synthetic few-shot examples for style or clarifying decision boundaries, or adding or modifying explicit instructions for conditionals, rules, or logic fixes; or provide step-by-step reasoning guidelines for multi-step logic problems if the model is failing to reason appropriately.`;

export interface MetapromptOptimizerConfig {
  metaprompt?: string;
  max_reflection_steps?: number;
  min_reflection_steps?: number;
}

function formatTemplate(
  template: string,
  vars: Record<string, string>
): string {
  let result = template;
  for (const [key, value] of Object.entries(vars)) {
    result = result.replace(new RegExp(`\\{${key}\\}`, "g"), value ?? "");
  }
  return result;
}

const ThinkSchema = z.object({
  thought: z.string().describe("Think carefully about the problem."),
});

const CritiqueSchema = z.object({
  criticism: z.string().describe("Critique the current reasoning."),
});

export class MetaPromptOptimizer {
  private model: BaseChatModel;
  private finalConfig: Required<MetapromptOptimizerConfig>;

  constructor(model: BaseChatModel, config?: MetapromptOptimizerConfig) {
    this.model = model;
    this.finalConfig = {
      metaprompt: config?.metaprompt ?? DEFAULT_METAPROMPT,
      max_reflection_steps:
        config?.max_reflection_steps ?? DEFAULT_MAX_REFLECTION_STEPS,
      min_reflection_steps:
        config?.min_reflection_steps ?? DEFAULT_MIN_REFLECTION_STEPS,
    };
  }

  private processSessionsAndPrompt(input: OptimizerInput): {
    promptStr: string;
    updateInstructions: string;
    sessionsStr: string;
  } {
    const { prompt, trajectories } = input;
    const promptStr =
      typeof prompt === "string" ? prompt : prompt.prompt;
    const updateInstructions =
      typeof prompt === "string" ? "" : prompt.update_instructions ?? "";
    const sessionsStr =
      typeof trajectories === "string"
        ? trajectories
        : formatSessions(trajectories as any);
    return { promptStr, updateInstructions, sessionsStr };
  }

  private async reflectThenUpdate(
    sessionsStr: string,
    promptStr: string,
    updateInstructions: string
  ): Promise<unknown> {
    const schemaInfo = getPromptExtractionSchema(promptStr);
    const OptimizedSchema = z.object({
      analysis: z.string().describe("Analysis of the current results."),
      improved_prompt: z
        .string()
        .optional()
        .describe(
          `The full updated prompt. ${schemaInfo.promptDescription}`
        ),
    });

    const thinkTool = {
      name: "think",
      description: "Think carefully about the problem.",
      schema: ThinkSchema,
    };
    const critiqueTool = {
      name: "critique",
      description: "Critique the current reasoning.",
      schema: CritiqueSchema,
    };

    const messages: (Record<string, unknown> | AIMessage)[] = [
      {
        role: "user",
        content: formatTemplate(this.finalConfig.metaprompt, {
          prompt: promptStr,
          update_instructions: updateInstructions,
          trajectories: sessionsStr,
        }),
      },
    ];

    const maxSteps = this.finalConfig.max_reflection_steps;
    const minSteps = this.finalConfig.min_reflection_steps;

    for (let ix = 0; ix < maxSteps; ix++) {
      if (ix === maxSteps - 1) {
        // Final step: force output
        const finalModel = this.model.withStructuredOutput(OptimizedSchema);
        const result = await finalModel.invoke(messages as any);
        return result;
      }

      let tools;
      if (ix < minSteps - 1) {
        tools = [thinkTool, critiqueTool];
      } else {
        tools = [
          thinkTool,
          critiqueTool,
          {
            name: "OptimizedPromptOutput",
            description: `Output the optimized prompt. ${schemaInfo.promptDescription}`,
            schema: OptimizedSchema,
          },
        ];
      }

      const boundModel = this.model.bindTools!(
        tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.schema,
        }))
      );

      const response = await boundModel.invoke(messages as any);
      const aiMsg = response as AIMessage;

      // Check for final output tool call
      if (aiMsg.tool_calls && aiMsg.tool_calls.length > 0) {
        for (const tc of aiMsg.tool_calls) {
          if (tc.name === "OptimizedPromptOutput") {
            return OptimizedSchema.parse(tc.args);
          }
        }
        messages.push(aiMsg);
        for (const tc of aiMsg.tool_calls) {
          messages.push(
            new ToolMessage({ content: "", tool_call_id: tc.id ?? "" })
          );
        }
      } else {
        messages.push(aiMsg);
      }
    }

    throw new Error("Exceeded reflection steps without final output");
  }

  private processResult(
    resultObj: unknown,
    originalPrompt: string
  ): string {
    const schemaInfo = getPromptExtractionSchema(originalPrompt);
    const improvedPrompt = (resultObj as any)?.improved_prompt;
    if (
      !improvedPrompt ||
      improvedPrompt.trim().toLowerCase().startsWith("no recommend")
    ) {
      return originalPrompt;
    }
    return schemaInfo.pipeline(improvedPrompt);
  }

  async invoke(
    input: OptimizerInput,
    _config?: RunnableConfig
  ): Promise<string> {
    const { promptStr, updateInstructions, sessionsStr } =
      this.processSessionsAndPrompt(input);
    if (!sessionsStr) return promptStr;

    const resultObj = await this.reflectThenUpdate(
      sessionsStr,
      promptStr,
      updateInstructions
    );
    return this.processResult(resultObj, promptStr);
  }
}

export function createMetapromptOptimizer(
  model: BaseChatModel,
  config?: MetapromptOptimizerConfig
): MetaPromptOptimizer {
  return new MetaPromptOptimizer(model, config);
}
