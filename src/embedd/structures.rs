#[derive(Debug, Deserialize, Clone)]
pub struct ImageDocument {
    pub id: i64,
    pub image_url: &str, 
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingResult {
    pub id: i64,
    pub embedding: Vec<f32>
}





