import { z } from "zod";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import {
  formatSessions,
  getPromptExtractionSchema,
} from "../utils.js";
import type { OptimizerInput } from "./types.js";

export const DEFAULT_MAX_REFLECTION_STEPS = 5;
export const DEFAULT_MIN_REFLECTION_STEPS = 1;

export const DEFAULT_GRADIENT_PROMPT = `You are reviewing the performance of an AI assistant in a given interaction.

## Instructions

The current prompt that was used for the session is provided below.

<current_prompt>
{prompt}
</current_prompt>

The developer provided the following instructions around when and how to update the prompt:

<update_instructions>
{update_instructions}
</update_instructions>

## Session data

Analyze the following trajectories (and any associated user feedback) (either conversations with a user or other work that was performed by the assistant):

<trajectories>
{trajectories}
</trajectories>

## Task

Analyze the conversation, including the user's request and the assistant's response, and evaluate:
1. How effectively the assistant fulfilled the user's intent.
2. Where the assistant might have deviated from user expectations or the desired outcome.
3. Specific areas (correctness, completeness, style, tone, alignment, etc.) that need improvement.

If the prompt seems to do well, then no further action is needed. We ONLY recommend updates if there is evidence of failures.
When failures occur, we want to recommend the minimal required changes to fix the problem.

Focus on actionable changes and be concrete.

1. Summarize the key successes and failures in the assistant's response.
2. Identify which failure mode(s) best describe the issues (examples: style mismatch, unclear or incomplete instructions, flawed logic or reasoning, hallucination, etc.).
3. Based on these failure modes, recommend the most suitable edit strategy. For example, consider::
   - Use synthetic few-shot examples for style or clarifying decision boundaries.
   - Use explicit instruction updates for conditionals, rules, or logic fixes.
   - Provide step-by-step reasoning guidelines for multi-step logic problems.
4. Provide detailed, concrete suggestions for how to update the prompt accordingly.

But remember, the final updated prompt should only be changed if there is evidence of poor performance, and our recommendations should be minimally invasive.
Do not recommend generic changes that aren't clearly linked to failure modes.

First think through the conversation and critique the current behavior.
If you believe the prompt needs to further adapt to the target context, provide precise recommendations.
Otherwise, mark \`warrants_adjustment\` as False and respond with 'No recommendations.'`;

export const DEFAULT_GRADIENT_METAPROMPT = `You are optimizing a prompt to handle its target task more effectively.

<current_prompt>
{current_prompt}
</current_prompt>

We hypothesize the current prompt underperforms for these reasons:

<hypotheses>
{hypotheses}
</hypotheses>

Based on these hypotheses, we recommend the following adjustments:

<recommendations>
{recommendations}
</recommendations>

Respond with the updated prompt. Remember to ONLY make changes that are clearly necessary. Aim to be minimally invasive:`;

export interface GradientOptimizerConfig {
  gradient_prompt?: string;
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
  thought: z
    .string()
    .describe(
      "A reflection tool, used to reason over complexities and hypothesize fixes."
    ),
});

const CritiqueSchema = z.object({
  criticism: z
    .string()
    .describe("A critique tool for diagnosing flaws in reasoning."),
});

const RecommendSchema = z.object({
  warrants_adjustment: z
    .boolean()
    .describe("Whether the prompt warrants adjustment."),
  hypotheses: z
    .string()
    .optional()
    .describe("Hypotheses about why the prompt underperforms."),
  full_recommendations: z
    .string()
    .optional()
    .describe("Full recommendations for improving the prompt."),
});

export class GradientPromptOptimizer {
  private model: BaseChatModel;
  private config: Required<GradientOptimizerConfig>;

  constructor(model: BaseChatModel, config?: GradientOptimizerConfig) {
    this.model = model;
    this.config = {
      gradient_prompt:
        config?.gradient_prompt ?? DEFAULT_GRADIENT_PROMPT,
      metaprompt: config?.metaprompt ?? DEFAULT_GRADIENT_METAPROMPT,
      max_reflection_steps:
        config?.max_reflection_steps ?? DEFAULT_MAX_REFLECTION_STEPS,
      min_reflection_steps:
        config?.min_reflection_steps ?? DEFAULT_MIN_REFLECTION_STEPS,
    };
  }

  private async runReactAgent(
    inputs: string
  ): Promise<z.infer<typeof RecommendSchema>> {
    const tools = [
      {
        name: "think",
        description:
          "A reflection tool, used to reason over complexities and hypothesize fixes.",
        schema: ThinkSchema,
      },
      {
        name: "critique",
        description: "A critique tool for diagnosing flaws in reasoning.",
        schema: CritiqueSchema,
      },
      {
        name: "recommend",
        description: "Decides whether a prompt should be adjusted.",
        schema: RecommendSchema,
      },
    ];

    const maxSteps = this.config.max_reflection_steps;
    const minSteps = this.config.min_reflection_steps;

    const messages: (Record<string, unknown> | AIMessage)[] = [
      { role: "user", content: inputs },
    ];

    for (let ix = 0; ix < maxSteps; ix++) {
      // Choose which tools to allow
      let availableTools;
      if (ix === maxSteps - 1) {
        availableTools = [tools[2]]; // only recommend
      } else if (ix < minSteps) {
        availableTools = tools.slice(0, 2); // think and critique
      } else {
        availableTools = tools; // all
      }

      const boundModel = this.model.bindTools!(
        availableTools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.schema,
        }))
      );

      const response = await boundModel.invoke(messages as any);
      const aiMsg = response as AIMessage;

      // Check for a recommend tool call
      if (aiMsg.tool_calls && aiMsg.tool_calls.length > 0) {
        for (const tc of aiMsg.tool_calls) {
          if (tc.name === "recommend") {
            return RecommendSchema.parse(tc.args);
          }
        }

        // Continue the loop with tool responses
        messages.push(aiMsg);
        for (const tc of aiMsg.tool_calls) {
          messages.push(
            new ToolMessage({ content: "", tool_call_id: tc.id ?? "" })
          );
        }
      } else {
        // No tool calls - if this is the last step, return a no-adjustment recommendation
        if (ix === maxSteps - 1) {
          return { warrants_adjustment: false };
        }
        messages.push(aiMsg);
      }
    }

    throw new Error(
      `Failed to generate a final recommendation after ${maxSteps} attempts`
    );
  }

  private async updatePrompt(
    hypotheses: string,
    recommendations: string,
    currentPrompt: string,
    updateInstructions: string
  ): Promise<string> {
    const schemaInfo = getPromptExtractionSchema(currentPrompt);
    const OptimizedSchema = z.object({
      analysis: z
        .string()
        .describe(
          "First, analyze the current results and plan improvements to reconcile them."
        ),
      improved_prompt: z
        .string()
        .optional()
        .describe(
          `Finally, generate the full updated prompt to address the identified issues. ${schemaInfo.promptDescription}`
        ),
    });

    const boundModel = this.model.withStructuredOutput(OptimizedSchema);
    const promptInput = formatTemplate(this.config.metaprompt, {
      current_prompt: currentPrompt,
      recommendations,
      hypotheses,
      update_instructions: updateInstructions,
    });

    const result = await boundModel.invoke(promptInput);
    const improved = (result as any).improved_prompt;
    if (!improved) return currentPrompt;
    return schemaInfo.pipeline(improved);
  }

  private processInput(input: OptimizerInput): {
    promptStr: string;
    sessionsStr: string;
    feedback: string;
    updateInstructions: string;
  } {
    const promptData = input.prompt;
    const sessionsData = input.trajectories;

    let promptStr: string;
    let updateInstructions: string;

    if (typeof promptData === "string") {
      promptStr = promptData;
      updateInstructions = "";
    } else {
      promptStr = promptData.prompt ?? "";
      updateInstructions = promptData.update_instructions ?? "";
    }

    const sessionsStr =
      typeof sessionsData === "string"
        ? sessionsData
        : formatSessions(sessionsData as any);

    return { promptStr, sessionsStr, feedback: "", updateInstructions };
  }

  async invoke(
    input: OptimizerInput,
    _config?: RunnableConfig
  ): Promise<string> {
    const { promptStr, sessionsStr, feedback, updateInstructions } =
      this.processInput(input);
    if (!sessionsStr) return promptStr;

    const reflectionInput = formatTemplate(this.config.gradient_prompt, {
      trajectories: sessionsStr,
      feedback,
      prompt: promptStr,
      update_instructions: updateInstructions,
    });

    const finalResponse = await this.runReactAgent(reflectionInput);
    if (!finalResponse.warrants_adjustment) return promptStr;

    return this.updatePrompt(
      finalResponse.hypotheses ?? "",
      finalResponse.full_recommendations ?? "",
      promptStr,
      updateInstructions
    );
  }
}

export function createGradientPromptOptimizer(
  model: BaseChatModel,
  config?: GradientOptimizerConfig
): GradientPromptOptimizer {
  return new GradientPromptOptimizer(model, config);
}
