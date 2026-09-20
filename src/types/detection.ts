export interface DetectResponse {
  is_synthetic: boolean;
  confidence: number;
  engine?: string;
  latency_ms?: number;
  call_id?: string;
  detail?: string;
  [key: string]: unknown;
}

export interface EngineConfig {
  id: 'baseline' | 'multimodal';
  prefix: 'e1' | 'e2';
  titleTag: string;
  title: string;
  description: string;
  endpoint: string;
  chips: {
    icon: string;
    label: string;
    highlight?: boolean;
  }[];
  exampleResponse: {
    is_synthetic: boolean;
    confidence: number;
  };
  docsUrl: string;
}
