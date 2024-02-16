use ndarray::{s, Array, Dim};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Value, tensor::DynOrtTensor, tensor::FromArray, tensor::InputTensor, tensor::OrtOwnedTensor};
use super::structures::{EmbeddingResult, ImageDocument};
use super::image_processor::ImageProcessor;
use std::sync::Arc;
use std::error::Error;


pub struct ImageEmbedder {
    session: Session,
    image_dimensions: i32, 
    preprocesser: Arc<ImageProcessor>,
}

struct ClipVector {
    id: i64, 
    clip_vector: Array<f32, Dim<[usize; 3]>>
}


impl ImageEmbedder {
    pub fn new(
        model_path: &str
    ) -> anyhow::Result<Self, Box<dyn Error + Send + Sync>> {
        let session = create_session(model_path)?;
        let clip_image_processor = ImageProcessor::create().expect("unable to create the img processor");
        Ok(Self {
            session,
            image_dimensions: 224,
            preprocesser: Arc::new(clip_image_processor)
        })
    }

    pub fn encode_image_batch(&self, image_documents: &[ImageDocument]) -> anyhow::Result<Vec<EmbeddingResult>, Box<dyn Error + Send + Sync>> {
        let uri_vecs: anyhow::Result<Vec<ClipVector>> = image_documents
            .iter()
            .map(|uri| {
                ClipVector {
                    id: uri.id,
                    clip_vector: self.preprocesser.uri_to_clip_vector(uri.image_url, self.image_dimensions) 
                }
            })
            .collect();

        // expected format:  num_uris, 3, onnx_model.image_size, onnx_model.image_size
        let md_vecs = uri_vecs?;
        let mut a = Array::<f32, _>::zeros((
            md_vecs.len() as usize,
            3 as usize,
            img_dims as usize,
            img_dims as usize,
        ));

        let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> = self
            .session
            .run([InputTensor::from_array(a.into_dyn())])?;

        let embedding: OrtOwnedTensor<f32, _> = outputs[outputs.len() - 1].try_extract().unwrap();
        let embedding = embedding.view().to_owned();
        let mut result: Vec<EmbeddingResult> = Vec::with_capacity(md_vecs.len());
        for idx in 0..md_vecs.len() {
            result.push(
                EmbeddingResult::new(md_vecs[idx].id, embedding
                    .slice(s![idx..idx + 1, ..])
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>())
            );
        }
        Ok(result)
    } 
} 


fn create_session(model_path: &str) -> Result<Session, Box<dyn Error + Send + Sync>> {
    let environment = Environment::builder()
        .with_name("embed-rs")
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .build()?
        .into_arc();
    let num_cpus = num_cpus::get();
    let session = SessionBuilder::new(&environment)?
        .with_parallel_execution(true)?
        .with_intra_threads(num_cpus as i16)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file(model_path)?;
    Ok(session)
}
