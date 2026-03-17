import { z } from "zod";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { RunnableConfig } from "@langchain/core/runnables";
import { formatSessions } from "../utils.js";
import type {
  OptimizerInput,
  MultiPromptOptimizerInput,
  Prompt,
} from "./types.js";
import {
  GradientPromptOptimizer,
  type GradientOptimizerConfig,
} from "./gradient.js";
import {
  MetaPromptOptimizer,
  type MetapromptOptimizerConfig,
} from "./metaprompt.js";
import { PromptMemoryMultiple } from "./stateless.js";

export type { Prompt };

export type OptimizerKind = "gradient" | "metaprompt" | "prompt_memory";

export interface PromptOptimizer {
  invoke(input: OptimizerInput, config?: RunnableConfig): Promise<string>;
}

function resolveModel(model: BaseChatModel): BaseChatModel {
  return model;
}

export function createPromptOptimizer(
  model: BaseChatModel,
  options?: {
    kind?: OptimizerKind;
    config?: GradientOptimizerConfig | MetapromptOptimizerConfig;
  }
): PromptOptimizer {
  const kind = options?.kind ?? "gradient";
  const chatModel = resolveModel(model);

  if (kind === "gradient") {
    return new GradientPromptOptimizer(
      chatModel,
      options?.config as GradientOptimizerConfig
    );
  } else if (kind === "metaprompt") {
    return new MetaPromptOptimizer(
      chatModel,
      options?.config as MetapromptOptimizerConfig
    );
  } else if (kind === "prompt_memory") {
    return new PromptMemoryMultiple(chatModel);
  } else {
    throw new Error(
      `Unsupported optimizer kind: ${kind}. Expected one of: gradient, metaprompt, prompt_memory`
    );
  }
}

export class MultiPromptOptimizer {
  private model: BaseChatModel;
  private kind: OptimizerKind;
  private config?: Record<string, unknown>;
  private optimizer: PromptOptimizer;

  constructor(
    model: BaseChatModel,
    options?: {
      kind?: OptimizerKind;
      config?: Record<string, unknown>;
    }
  ) {
    this.model = resolveModel(model);
    this.kind = options?.kind ?? "gradient";
    this.config = options?.config;
    this.optimizer = createPromptOptimizer(this.model, {
      kind: this.kind,
      config: this.config as any,
    });
  }

  async invoke(
    input: MultiPromptOptimizerInput,
    _config?: RunnableConfig
  ): Promise<Prompt[]> {
    const { trajectories, prompts } = input;

    const sessionsStr =
      typeof trajectories === "string"
        ? trajectories
        : formatSessions(trajectories as any);

    // If only one prompt and no when_to_update, just update it
    if (prompts.length === 1 && !prompts[0].when_to_update) {
      const updatedPrompt = await this.optimizer.invoke({
        trajectories,
        prompt: prompts[0],
      });
      return [{ ...prompts[0], prompt: updatedPrompt }];
    }

    // Classify which prompts to update
    const choices = prompts.map((p) => p.name);
    const promptJoinedContent = prompts
      .map((p) => `${p.name}: ${p.prompt}\n`)
      .join("");

    const classificationPrompt = `Analyze the following trajectories and decide which prompts
ought to be updated to improve the performance on future trajectories:

${sessionsStr}

Below are the prompts being optimized:
${promptJoinedContent}

Return a JSON object with "which": [...], listing the names of prompts that need updates. Only include names from: ${JSON.stringify(choices)}`;

    const ClassifySchema = z.object({
      reasoning: z
        .string()
        .describe("Reasoning for which prompts to update."),
      which: z
        .array(z.string())
        .describe(
          `List of prompt names that should be updated. Must be among ${JSON.stringify(choices)}`
        ),
    });

    const classifierModel = this.model.withStructuredOutput(ClassifySchema);
    const result = await classifierModel.invoke(classificationPrompt);
    const toUpdate = new Set(
      result.which.filter((name: string) => choices.includes(name))
    );

    const whichToUpdate = prompts.filter((p) => toUpdate.has(p.name));

    // Update each chosen prompt concurrently
    const updatedResults = await Promise.all(
      whichToUpdate.map((p) =>
        this.optimizer.invoke({ trajectories, prompt: p })
      )
    );

    const updatedMap = new Map<string, string>();
    for (let i = 0; i < whichToUpdate.length; i++) {
      updatedMap.set(whichToUpdate[i].name, updatedResults[i]);
    }

    return prompts.map((p) =>
      updatedMap.has(p.name)
        ? { ...p, prompt: updatedMap.get(p.name)! }
        : p
    );
  }
}

export function createMultiPromptOptimizer(
  model: BaseChatModel,
  options?: {
    kind?: OptimizerKind;
    config?: Record<string, unknown>;
  }
): MultiPromptOptimizer {
  return new MultiPromptOptimizer(model, options);
}
