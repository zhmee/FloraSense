export interface KeywordMatch {
  keyword: string;
  category: string;
  score: number;
}

export interface FlowerSuggestion {
  name: string;
  scientific_name: string;
  colors: string[];
  plant_types: string[];
  maintenance: string[];
  meanings: string[];
  score: number;
  matched_keywords: KeywordMatch[];
}

export interface RecommendationResponse {
  query: string;
  keywords_used: KeywordMatch[];
  suggestions: FlowerSuggestion[];
}
