import { z } from 'zod';

// Task related types
export type TaskStatus = 'pending' | 'reasoning' | 'responding' | 'complete' | 'error';

export interface TaskInfo {
  status: TaskStatus;
  prompt: string;
  showReasoning?: boolean;
  reasoning?: string;
  response?: string;
  error?: string;
  timestamp: number;
}

// Conversation related types
export interface ConversationEntry {
  timestamp: number;
  prompt: string;
  reasoning: string;
  response: string;
  model: string;
}

export interface ConversationContext {
  entries: ConversationEntry[];
  maxEntries: number;
}

export interface ClaudeContentPart {
  type: string;
  text: string;
}

export interface ClaudeMessage {
  role: 'user' | 'assistant';
  content: string | ClaudeContentPart[];
}

export interface UiMessage {
  ts: number;
  type: string;
  say?: string;
  ask?: string;
  text: string;
  conversationHistoryIndex: number;
}

// Schema definitions
export const GENERATE_RESPONSE_SCHEMA = z.object({
  prompt: z.string(),
  showReasoning: z.boolean().optional(),
  clearContext: z.boolean().optional(),
  includeHistory: z.boolean().optional()
});

export const CHECK_RESPONSE_STATUS_SCHEMA = z.object({
  taskId: z.string()
});

export type GenerateResponseArgs = z.infer<typeof GENERATE_RESPONSE_SCHEMA>;
export type CheckResponseStatusArgs = z.infer<typeof CHECK_RESPONSE_STATUS_SCHEMA>; 