use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct ImageDocument<'a> {
    pub id: i64,
    pub image_url: &'a str, 
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingResult {
    pub id: i64,
    pub embedding: Vec<f32>
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingFailure {
    pub id: i64,
    pub err: String
}





