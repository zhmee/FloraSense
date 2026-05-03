export interface KeywordUsed {
  keyword: string
  category: string
  score: number
  explanation?: string
  explanation_source?: 'llm' | 'local' | ''
}

export interface MatchedKeyword {
  keyword: string
  category: string
  score: number
}

export interface FlowerSuggestion {
  name: string
  scientific_name: string
  colors: string[]
  plant_types: string[]
  maintenance: string[]
  meanings: string[]
  occasions: string[]
  score: number
  query_fit_explanation?: string
  query_fit_occasion_summary?: string
  ir_summary?: string
  ir_summary_source?: 'llm' | 'local' | 'csv' | ''
  rag_summary?: string
  rag_occasion_summary?: string
  rag_source?: 'llm' | 'local' | ''
  rag_occasion_source?: 'llm' | 'local' | ''
  explanation_source?: 'llm' | 'local'
  occasion_summary_source?: 'llm' | 'local' | 'csv' | ''
  matched_keywords: MatchedKeyword[]
  latent_radar_chart: string | null
  latent_radar_axes: string[]
  image_url?: string
}

export interface RagContextDocument {
  rank: number
  name: string
  scientific_name: string
  score?: number
  colors: string[]
  maintenance: string[]
  plant_types: string[]
  meanings: string[]
  occasions: string[]
  matched_keywords: Array<{
    keyword: string
    category: string
  }>
}

export interface RagResponse {
  user_query: string
  retrieval_query: string
  query_transform_source: 'llm' | 'local'
  query_transform_rationale: string
  answer: string
  answer_source: 'llm' | 'local' | ''
  context_documents: RagContextDocument[]
}

export interface RecommendationResponse {
  query: string
  keywords_used: KeywordUsed[]
  score_scale?: 'unit' | 'percent'
  query_latent_radar_chart: string | null
  query_latent_radar_axes: string[]
  suggestions: FlowerSuggestion[]
  rag?: RagResponse
}

export interface AutocompleteResponse {
  query: string
  suggestions: string[]
}

export interface VisualizerFlower {
  id: string
  name: string
  scientific_name: string
  image_url?: string
  colors: string[]
  plant_types: string[]
  maintenance: string[]
  meanings: string[]
  occasions: string[]
  primary_color: string
  primary_meaning: string
  primary_occasion: string
  latent_axes: string[]
  latent_position: {
    x: number
    y: number
    z: number
  }
  summary: string[]
}

export interface VisualizerFlowersResponse {
  flowers: VisualizerFlower[]
}

export interface BouquetMeaning {
  label: string
  score: number
}

export interface BouquetInsightsResponse {
  scientific_names: string[]
  meanings: BouquetMeaning[]
  recommendations: FlowerSuggestion[]
}
