import type { BaseMessage } from "@langchain/core/messages";

export interface Prompt {
  name: string;
  prompt: string;
  update_instructions?: string | null;
  when_to_update?: string | null;
}

export interface AnnotatedTrajectory {
  messages: (BaseMessage | Record<string, unknown>)[];
  feedback?: Record<string, string | number | boolean> | string | null;
}

export interface OptimizerInput {
  trajectories: AnnotatedTrajectory[] | string;
  prompt: string | Prompt;
}

export interface MultiPromptOptimizerInput {
  trajectories: AnnotatedTrajectory[] | string;
  prompts: Prompt[];
}
