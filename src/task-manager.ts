import { TaskInfo, TaskStatus } from './types.js';
import { v4 as uuidv4 } from 'uuid';

export class TaskError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TaskError';
  }
}

export class TaskManager {
  private readonly tasks: Map<string, TaskInfo> = new Map();
  private static readonly POLL_INTERVAL = 100; // ms

  createTask(prompt: string, showReasoning?: boolean): string {
    if (!prompt.trim()) {
      throw new TaskError('Prompt cannot be empty');
    }

    const taskId = uuidv4();
    this.tasks.set(taskId, {
      status: 'pending',
      prompt,
      showReasoning,
      timestamp: Date.now()
    });
    return taskId;
  }

  getTask(taskId: string): TaskInfo | undefined {
    this.validateTaskId(taskId);
    return this.tasks.get(taskId);
  }

  updateTask(taskId: string, updates: Partial<TaskInfo>): void {
    this.validateTaskId(taskId);
    const task = this.tasks.get(taskId);
    
    if (!task) {
      throw new TaskError(`Task ${taskId} not found`);
    }

    this.tasks.set(taskId, { 
      ...task, 
      ...updates,
      timestamp: Date.now() 
    });
  }

  async waitForTaskUpdate(taskId: string, timeout: number = 10000): Promise<TaskInfo> {
    this.validateTaskId(taskId);
    this.validateTimeout(timeout);

    const task = this.tasks.get(taskId);
    if (!task) {
      throw new TaskError(`Task ${taskId} not found`);
    }

    if (this.isTaskComplete(task.status)) {
      return task;
    }

    const initialStatus = task.status;
    
    try {
      await new Promise((resolve, reject) => {
        const checkInterval = setInterval(() => {
          const currentTask = this.tasks.get(taskId);
          if (!currentTask) {
            clearInterval(checkInterval);
            reject(new TaskError(`Task ${taskId} no longer exists`));
            return;
          }

          if (currentTask.status !== initialStatus || this.isTaskComplete(currentTask.status)) {
            clearInterval(checkInterval);
            resolve(true);
          }
        }, TaskManager.POLL_INTERVAL);

        setTimeout(() => {
          clearInterval(checkInterval);
          resolve(false);
        }, timeout);
      });
    } catch (error) {
      if (error instanceof TaskError) {
        throw error;
      }
      throw new TaskError(error instanceof Error ? error.message : 'Unknown error occurred');
    }

    const finalTask = this.tasks.get(taskId);
    if (!finalTask) {
      throw new TaskError(`Task ${taskId} no longer exists`);
    }

    return finalTask;
  }

  setTaskStatus(taskId: string, status: TaskStatus, additionalInfo: Partial<TaskInfo> = {}): void {
    this.validateTaskId(taskId);
    const task = this.tasks.get(taskId);
    
    if (!task) {
      throw new TaskError(`Task ${taskId} not found`);
    }

    this.tasks.set(taskId, { 
      ...task, 
      ...additionalInfo,
      status,
      timestamp: Date.now()
    });
  }

  private validateTaskId(taskId: string): void {
    if (!taskId || typeof taskId !== 'string') {
      throw new TaskError('Invalid task ID');
    }
  }

  private validateTimeout(timeout: number): void {
    if (!Number.isFinite(timeout) || timeout <= 0) {
      throw new TaskError('Timeout must be a positive number');
    }
  }

  private isTaskComplete(status: TaskStatus): boolean {
    return ['complete', 'error'].includes(status);
  }
} 