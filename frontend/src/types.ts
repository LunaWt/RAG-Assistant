export type Hit = { title: string; href: string };

export type ChatEvent =
  | { type: 'thought_delta'; text: string }
  | { type: 'text_delta'; text: string }
  | { type: 'tool_start'; name: string[]; args: Record<string, unknown>[] }
  | { type: 'tool_hits'; query: string; hits: Hit[] }
  | { type: 'stream_reset' }
  | { type: 'done' }
  | { type: 'error'; message: string };

export type ToolResult = { query: string; hits: Hit[] };

export type ToolBlock = {
  type: 'tool';
  names: string[];
  args: Record<string, unknown>[];
  running: boolean;
  results: ToolResult[];
};

export type Block =
  | { type: 'thought'; text: string }
  | { type: 'answer'; text: string }
  | ToolBlock;

export type ChatMessage =
  | { role: 'user'; content: string }
  | { role: 'assistant'; blocks: Block[] };
