export interface KeywordUsed {
  keyword: string
  category: string
  score: number
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
  matched_keywords: MatchedKeyword[]
  latent_radar_chart: string | null
  latent_radar_axes: string[]
  image_url?: string
}

export interface RecommendationResponse {
  query: string
  keywords_used: KeywordUsed[]
  query_latent_radar_chart: string | null
  query_latent_radar_axes: string[]
  suggestions: FlowerSuggestion[]
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
