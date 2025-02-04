import { ConversationContext, ConversationEntry, ClaudeMessage, UiMessage, ClaudeContentPart } from './types.js';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs/promises';

export class ConversationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConversationError';
  }
}

interface ConversationStats {
  dir: string;
  mtime: number;
  hasEnded: boolean;
}

export class ConversationManager {
  private readonly context: ConversationContext;
  private static readonly DEFAULT_MAX_ENTRIES = 10;
  private static readonly MODEL_LIMITS = {
    DEEPSEEK_MAX_LENGTH: 50000,
    CLAUDE_MAX_LENGTH: 600000
  } as const;

  constructor(maxEntries: number = ConversationManager.DEFAULT_MAX_ENTRIES) {
    this.validateMaxEntries(maxEntries);
    this.context = {
      entries: [],
      maxEntries
    };
  }

  addEntry(entry: ConversationEntry): void {
    this.validateEntry(entry);
    this.context.entries.push(entry);
    if (this.context.entries.length > this.context.maxEntries) {
      this.context.entries.shift();
    }
  }

  clearContext(): void {
    this.context.entries = [];
  }

  getEntries(): readonly ConversationEntry[] {
    return Object.freeze([...this.context.entries]);
  }

  formatContextForPrompt(): string {
    if (this.context.entries.length === 0) {
      return '';
    }

    return this.context.entries
      .map(entry => this.formatEntry(entry))
      .join('\n\n');
  }

  async findActiveConversation(): Promise<ClaudeMessage[] | null> {
    try {
      const tasksPath = this.getClaudePath();
      await this.validateDirectoryExists(tasksPath);
      
      const dirs = await fs.readdir(tasksPath);
      const dirStats = await this.getConversationStats(tasksPath, dirs);
      const activeConversation = await this.findMostRecentActiveConversation(tasksPath, dirStats);
      
      return activeConversation;
    } catch (error) {
      if (error instanceof ConversationError) {
        throw error;
      }
      throw new ConversationError(error instanceof Error ? error.message : 'Unknown error occurred');
    }
  }

  formatHistoryForModel(history: ClaudeMessage[], isDeepSeek: boolean): string {
    this.validateHistory(history);
    
    const maxLength = isDeepSeek ? 
      ConversationManager.MODEL_LIMITS.DEEPSEEK_MAX_LENGTH : 
      ConversationManager.MODEL_LIMITS.CLAUDE_MAX_LENGTH;

    const formattedMessages = this.formatMessagesWithinLimit(history, maxLength);
    return formattedMessages.join('\n\n');
  }

  private formatEntry(entry: ConversationEntry): string {
    return `Question: ${entry.prompt}\nReasoning: ${entry.reasoning}\nAnswer: ${entry.response}`;
  }

  private async getConversationStats(tasksPath: string, dirs: string[]): Promise<ConversationStats[]> {
    const stats = await Promise.all(
      dirs.map(async (dir) => {
        try {
          const historyPath = path.join(tasksPath, dir, 'api_conversation_history.json');
          const stats = await fs.stat(historyPath);
          const uiPath = path.join(tasksPath, dir, 'ui_messages.json');
          const uiContent = await fs.readFile(uiPath, 'utf8');
          const uiMessages = this.parseUiMessages(uiContent);
          
          return {
            dir,
            mtime: stats.mtime.getTime(),
            hasEnded: uiMessages.some(m => m.type === 'conversation_ended')
          };
        } catch (error) {
          return null;
        }
      })
    );

    return stats.filter((stat): stat is NonNullable<typeof stat> => 
      stat !== null && !stat.hasEnded
    ).sort((a, b) => b.mtime - a.mtime);
  }

  private async findMostRecentActiveConversation(
    tasksPath: string, 
    dirStats: ConversationStats[]
  ): Promise<ClaudeMessage[] | null> {
    const latest = dirStats[0]?.dir;
    if (!latest) {
      return null;
    }

    const historyPath = path.join(tasksPath, latest, 'api_conversation_history.json');
    const history = await fs.readFile(historyPath, 'utf8');
    return this.parseClaudeMessages(history);
  }

  private formatMessagesWithinLimit(history: ClaudeMessage[], maxLength: number): string[] {
    const formattedMessages: string[] = [];
    let totalLength = 0;

    for (let i = history.length - 1; i >= 0; i--) {
      const msg = history[i];
      const formattedMsg = this.formatMessage(msg);
      
      if (totalLength + formattedMsg.length > maxLength) {
        break;
      }
      
      formattedMessages.push(formattedMsg);
      totalLength += formattedMsg.length;
    }

    return formattedMessages.reverse();
  }

  private formatMessage(msg: ClaudeMessage): string {
    const content = Array.isArray(msg.content)
      ? msg.content.map(c => c.text).join('\n')
      : msg.content;
    
    return `${msg.role === 'user' ? 'Human' : 'Assistant'}: ${content}`;
  }

  private getClaudePath(): string {
    const homeDir = os.homedir();
    const platform = process.platform;
    
    const paths = {
      'win32': ['AppData', 'Roaming'],
      'darwin': ['Library', 'Application Support'],
      'linux': ['.config']
    };

    const basePath = paths[platform as keyof typeof paths] || ['.config'];
    return path.join(
      homeDir,
      ...basePath,
      'Code',
      'User',
      'globalStorage',
      'saoudrizwan.claude-dev',
      'tasks'
    );
  }

  private async validateDirectoryExists(dir: string): Promise<void> {
    try {
      await fs.access(dir);
    } catch {
      throw new ConversationError(`Directory not found: ${dir}`);
    }
  }

  private validateMaxEntries(maxEntries: number): void {
    if (!Number.isInteger(maxEntries) || maxEntries <= 0) {
      throw new ConversationError('Max entries must be a positive integer');
    }
  }

  private validateEntry(entry: ConversationEntry): void {
    if (!entry || typeof entry !== 'object') {
      throw new ConversationError('Invalid conversation entry');
    }

    const requiredFields: (keyof ConversationEntry)[] = ['timestamp', 'prompt', 'reasoning', 'response', 'model'];
    for (const field of requiredFields) {
      if (!(field in entry)) {
        throw new ConversationError(`Missing required field: ${field}`);
      }
    }
  }

  private validateHistory(history: ClaudeMessage[]): void {
    if (!Array.isArray(history)) {
      throw new ConversationError('History must be an array');
    }

    if (history.some(msg => !this.isValidClaudeMessage(msg))) {
      throw new ConversationError('Invalid message format in history');
    }
  }

  private isValidClaudeMessage(msg: any): msg is ClaudeMessage {
    return msg 
      && (msg.role === 'user' || msg.role === 'assistant')
      && (typeof msg.content === 'string' || 
          (Array.isArray(msg.content) && msg.content.every((c: any): c is ClaudeContentPart => 
            c && typeof c.type === 'string' && typeof c.text === 'string'
          ))
      );
  }

  private parseUiMessages(content: string): UiMessage[] {
    try {
      const messages = JSON.parse(content);
      if (!Array.isArray(messages)) {
        throw new ConversationError('UI messages must be an array');
      }
      return messages;
    } catch {
      throw new ConversationError('Failed to parse UI messages');
    }
  }

  private parseClaudeMessages(content: string): ClaudeMessage[] {
    try {
      const messages = JSON.parse(content);
      this.validateHistory(messages);
      return messages;
    } catch {
      throw new ConversationError('Failed to parse Claude messages');
    }
  }
} 