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
}

export interface RecommendationResponse {
  query: string
  keywords_used: KeywordUsed[]
  suggestions: FlowerSuggestion[]
}