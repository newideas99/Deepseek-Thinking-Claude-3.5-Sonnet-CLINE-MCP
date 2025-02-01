#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import { OpenAI } from 'openai';
import dotenv from 'dotenv';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs/promises';
import { v4 as uuidv4 } from 'uuid';

// Load environment variables
dotenv.config();

// Debug logging
const DEBUG = true;
const log = (...args: any[]) => {
  if (DEBUG) {
    console.error('[DEEPSEEK-CLAUDE MCP]', ...args);
  }
};

// Constants
const DEEPSEEK_MODEL = "deepseek/deepseek-r1";
const CLAUDE_MODEL = "anthropic/claude-3.5-sonnet:beta";

interface ConversationEntry {
  timestamp: number;
  prompt: string;
  reasoning: string;
  response: string;
  model: string;
}

interface ConversationContext {
  entries: ConversationEntry[];
  maxEntries: number;
}

interface GenerateResponseArgs {
  prompt: string;
  showReasoning?: boolean;
  clearContext?: boolean;
  includeHistory?: boolean;
}

interface CheckResponseStatusArgs {
  taskId: string;
}

interface TaskStatus {
  status: 'pending' | 'reasoning' | 'responding' | 'complete' | 'error';
  prompt: string;
  showReasoning?: boolean;
  reasoning?: string;
  response?: string;
  error?: string;
  timestamp: number;
}

const isValidCheckResponseStatusArgs = (args: any): args is CheckResponseStatusArgs =>
  typeof args === 'object' &&
  args !== null &&
  typeof args.taskId === 'string';

interface ClaudeMessage {
  role: 'user' | 'assistant';
  content: string | { type: string; text: string }[];
}

interface UiMessage {
  ts: number;
  type: string;
  say?: string;
  ask?: string;
  text: string;
  conversationHistoryIndex: number;
}

const isValidGenerateResponseArgs = (args: any): args is GenerateResponseArgs =>
  typeof args === 'object' &&
  args !== null &&
  typeof args.prompt === 'string' &&
  (args.showReasoning === undefined || typeof args.showReasoning === 'boolean') &&
  (args.clearContext === undefined || typeof args.clearContext === 'boolean') &&
  (args.includeHistory === undefined || typeof args.includeHistory === 'boolean');

function getClaudePath(): string {
  const homeDir = os.homedir();
  switch (process.platform) {
    case 'win32':
      return path.join(homeDir, 'AppData', 'Roaming', 'Code', 'User', 'globalStorage', 'saoudrizwan.claude-dev', 'tasks');
    case 'darwin':
      return path.join(homeDir, 'Library', 'Application Support', 'Code', 'User', 'globalStorage', 'saoudrizwan.claude-dev', 'tasks');
    default: // linux
      return path.join(homeDir, '.config', 'Code', 'User', 'globalStorage', 'saoudrizwan.claude-dev', 'tasks');
  }
}

async function findActiveConversation(): Promise<ClaudeMessage[] | null> {
  try {
    const tasksPath = getClaudePath();
    const dirs = await fs.readdir(tasksPath);
    
    // Get modification time for each api_conversation_history.json
    const dirStats = await Promise.all(
      dirs.map(async (dir) => {
        try {
          const historyPath = path.join(tasksPath, dir, 'api_conversation_history.json');
          const stats = await fs.stat(historyPath);
          const uiPath = path.join(tasksPath, dir, 'ui_messages.json');
          const uiContent = await fs.readFile(uiPath, 'utf8');
          const uiMessages: UiMessage[] = JSON.parse(uiContent);
          const hasEnded = uiMessages.some(m => m.type === 'conversation_ended');
          
          return {
            dir,
            mtime: stats.mtime.getTime(),
            hasEnded
          };
        } catch (error) {
          log('Error checking folder:', dir, error);
          return null;
        }
      })
    );

    // Filter out errors and ended conversations, then sort by modification time
    const sortedDirs = dirStats
      .filter((stat): stat is NonNullable<typeof stat> => 
        stat !== null && !stat.hasEnded
      )
      .sort((a, b) => b.mtime - a.mtime);

    // Use most recently modified active conversation
    const latest = sortedDirs[0]?.dir;
    if (!latest) {
      log('No active conversations found');
      return null;
    }
    
    const historyPath = path.join(tasksPath, latest, 'api_conversation_history.json');
    const history = await fs.readFile(historyPath, 'utf8');
    return JSON.parse(history);
  } catch (error) {
    log('Error finding active conversation:', error);
    return null;
  }
}

function formatHistoryForModel(history: ClaudeMessage[], isDeepSeek: boolean): string {
  const maxLength = isDeepSeek ? 50000 : 600000; // 50k chars for DeepSeek, 600k for Claude
  const formattedMessages = [];
  let totalLength = 0;
  
  // Process messages in reverse chronological order to get most recent first
  for (let i = history.length - 1; i >= 0; i--) {
    const msg = history[i];
    const content = Array.isArray(msg.content)
      ? msg.content.map(c => c.text).join('\n')
      : msg.content;
    
    const formattedMsg = `${msg.role === 'user' ? 'Human' : 'Assistant'}: ${content}`;
    const msgLength = formattedMsg.length;
    
    // Stop adding messages if we'd exceed the limit
    if (totalLength + msgLength > maxLength) {
      break;
    }
    
    formattedMessages.push(formattedMsg); // Add most recent messages first
    totalLength += msgLength;
  }
  
  // Reverse to get chronological order
  return formattedMessages.reverse().join('\n\n');
}

class DeepseekClaudeServer {
  private server: Server;
  private openrouterClient: OpenAI;
  private context: ConversationContext = {
    entries: [],
    maxEntries: 10
  };
  private activeTasks: Map<string, TaskStatus> = new Map();

  constructor() {
    log('Initializing API clients...');
    
    // Initialize OpenRouter client
    this.openrouterClient = new OpenAI({
      baseURL: "https://openrouter.ai/api/v1",
      apiKey: process.env.OPENROUTER_API_KEY
    });
    log('OpenRouter client initialized');

    // Initialize MCP server
    this.server = new Server(
      {
        name: 'deepseek-thinking-claude-mcp',
        version: '0.1.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
    
    // Error handling
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private addToContext(entry: ConversationEntry) {
    this.context.entries.push(entry);
    if (this.context.entries.length > this.context.maxEntries) {
      this.context.entries.shift();  // Remove oldest
    }
  }

  private formatContextForPrompt(): string {
    return this.context.entries
      .map(entry => `Question: ${entry.prompt}\nReasoning: ${entry.reasoning}\nAnswer: ${entry.response}`)
      .join('\n\n');
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'generate_response',
          description: 'Generate a response using DeepSeek\'s reasoning and Claude\'s response generation through OpenRouter.',
          inputSchema: {
            type: 'object',
            properties: {
              prompt: {
                type: 'string',
                description: 'The user\'s input prompt'
              },
              showReasoning: {
                type: 'boolean',
                description: 'Whether to include reasoning in response',
                default: false
              },
              clearContext: {
                type: 'boolean',
                description: 'Clear conversation history before this request',
                default: false
              },
              includeHistory: {
                type: 'boolean',
                description: 'Include Cline conversation history for context',
                default: true
              }
            },
            required: ['prompt']
          }
        },
        {
          name: 'check_response_status',
          description: 'Check the status of a response generation task',
          inputSchema: {
            type: 'object',
            properties: {
              taskId: {
                type: 'string',
                description: 'The task ID returned by generate_response'
              }
            },
            required: ['taskId']
          }
        }
      ]
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      if (request.params.name === 'generate_response') {
        if (!isValidGenerateResponseArgs(request.params.arguments)) {
          throw new McpError(
            ErrorCode.InvalidParams,
            'Invalid generate_response arguments'
          );
        }

        const taskId = uuidv4();
        const { prompt, showReasoning, clearContext, includeHistory } = request.params.arguments;

        // Initialize task status
        this.activeTasks.set(taskId, {
          status: 'pending',
          prompt,
          showReasoning,
          timestamp: Date.now()
        });

        // Start processing in background
        this.processTask(taskId, clearContext, includeHistory).catch(error => {
          log('Error processing task:', error);
          this.activeTasks.set(taskId, {
            ...this.activeTasks.get(taskId)!,
            status: 'error',
            error: error.message
          });
        });

        // Return task ID immediately
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({ taskId })
            }
          ]
        };
      } else if (request.params.name === 'check_response_status') {
        if (!isValidCheckResponseStatusArgs(request.params.arguments)) {
          throw new McpError(
            ErrorCode.InvalidParams,
            'Invalid check_response_status arguments'
          );
        }

        const taskId = request.params.arguments.taskId;
        const task = this.activeTasks.get(taskId);

        if (!task) {
          throw new McpError(
            ErrorCode.InvalidRequest,
            `No task found with ID: ${taskId}`
          );
        }

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                status: task.status,
                reasoning: task.showReasoning ? task.reasoning : undefined,
                response: task.status === 'complete' ? task.response : undefined,
                error: task.error
              })
            }
          ]
        };
      } else {
        throw new McpError(
          ErrorCode.MethodNotFound,
          `Unknown tool: ${request.params.name}`
        );
      }
    });
  }

  private async processTask(taskId: string, clearContext?: boolean, includeHistory?: boolean): Promise<void> {
    const task = this.activeTasks.get(taskId);
    if (!task) {
      throw new Error(`No task found with ID: ${taskId}`);
    }
    
    try {
      if (clearContext) {
        this.context.entries = [];
      }

      // Update status to reasoning
      this.activeTasks.set(taskId, {
        ...task,
        status: 'reasoning'
      });

      // Get Cline conversation history if requested
      let history: ClaudeMessage[] | null = null;
      if (includeHistory !== false) {
        history = await findActiveConversation();
      }

      // Get DeepSeek reasoning with limited history
      const reasoningHistory = history ? formatHistoryForModel(history, true) : '';
      const reasoningPrompt = reasoningHistory 
        ? `${reasoningHistory}\n\nNew question: ${task.prompt}`
        : task.prompt;
      const reasoning = await this.getDeepseekReasoning(reasoningPrompt);

      // Update status with reasoning
      this.activeTasks.set(taskId, {
        ...task,
        status: 'responding',
        reasoning
      });

      // Get final response with full history
      const responseHistory = history ? formatHistoryForModel(history, false) : '';
      const fullPrompt = responseHistory 
        ? `${responseHistory}\n\nCurrent task: ${task.prompt}`
        : task.prompt;
      const response = await this.getFinalResponse(fullPrompt, reasoning);

      // Add to context after successful response
      this.addToContext({
        timestamp: Date.now(),
        prompt: task.prompt,
        reasoning,
        response,
        model: CLAUDE_MODEL
      });

      // Update status to complete
      this.activeTasks.set(taskId, {
        ...task,
        status: 'complete',
        reasoning,
        response,
        timestamp: Date.now()
      });
    } catch (error) {
      // Update status to error
      this.activeTasks.set(taskId, {
        ...task,
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: Date.now()
      });
      throw error;
    }
  }

  private async getDeepseekReasoning(prompt: string): Promise<string> {
    const contextPrompt = this.context.entries.length > 0
      ? `Previous conversation:\n${this.formatContextForPrompt()}\n\nNew question: ${prompt}`
      : prompt;

    try {
      // Get reasoning from DeepSeek
      const response = await this.openrouterClient.chat.completions.create({
        model: DEEPSEEK_MODEL,
        messages: [{ 
          role: "user", 
          content: contextPrompt
        }],
        include_reasoning: true,
        temperature: 0.7,
        top_p: 1
      } as any);

      // Get reasoning from response
      const responseData = response as any;
      if (!responseData.choices?.[0]?.message?.reasoning) {
        throw new Error('No reasoning received from DeepSeek');
      }
      return responseData.choices[0].message.reasoning;
    } catch (error) {
      log('Error in getDeepseekReasoning:', error);
      throw error;
    }
  }

  private async getFinalResponse(prompt: string, reasoning: string): Promise<string> {
    try {
      // Create messages array with proper structure
      const messages = [
        // First the user's question
        {
          role: "user" as const,
          content: prompt
        },
        // Then the reasoning as assistant's thoughts
        {
          role: "assistant" as const,
          content: `<thinking>${reasoning}</thinking>`
        }
      ];

      // If we have context, prepend it as previous turns
      if (this.context.entries.length > 0) {
        const contextMessages = this.context.entries.flatMap(entry => [
          {
            role: "user" as const,
            content: entry.prompt
          },
          {
            role: "assistant" as const,
            content: entry.response
          }
        ]);
        messages.unshift(...contextMessages);
      }

      const response = await this.openrouterClient.chat.completions.create({
        model: CLAUDE_MODEL,
        messages: messages,
        temperature: 0.7,
        top_p: 1,
        repetition_penalty: 1
      } as any);
      
      return response.choices[0].message.content || "Error: No response content";
    } catch (error) {
      log('Error in getFinalResponse:', error);
      throw error;
    }
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('DeepSeek-Claude MCP server running on stdio');
  }
}

const server = new DeepseekClaudeServer();
server.run().catch(console.error);
