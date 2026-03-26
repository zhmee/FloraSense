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
