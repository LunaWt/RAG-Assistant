export type Hit = { title: string; href?: string; snippet?: string };

export type ToolResult =
  | { status: 'ok' | 'empty'; hits: Hit[] }
  | { status: 'error'; hits: Hit[]; error: string };

export type ChatEvent =
  | { type: 'thought_delta'; text: string }
  | { type: 'text_delta'; text: string }
  | { type: 'tool_start'; name: string[]; args: Record<string, unknown>[] }
  | { type: 'tool_result'; index: number; result: ToolResult }
  | { type: 'stream_reset' }
  | { type: 'done' }
  | { type: 'error'; message: string };

/** One tool round. results[i] belongs to names[i] and stays null until that call reports. */
export type ToolBlock = {
  type: 'tool';
  names: string[];
  args: Record<string, unknown>[];
  results: (ToolResult | null)[];
};

export type Block =
  | { type: 'thought'; text: string }
  | { type: 'answer'; text: string }
  | ToolBlock;

export type ChatMessage =
  | { role: 'user'; content: string }
  | { role: 'assistant'; blocks: Block[] };
