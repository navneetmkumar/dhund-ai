use ndarray::{s, Array, Dim};
use ort::{inputs, CUDAExecutionProvider, CPUExecutionProvider, Session, SessionOutputs};
use super::structures::{EmbeddingFailure, EmbeddingResult, ImageDocument};
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
        ort::init()
		.with_execution_providers([CUDAExecutionProvider::default().build(), CPUExecutionProvider::default().build()])
		.commit()?;
        let session = Session::builder()?.with_model_from_file(model_path)?;
        let clip_image_processor = ImageProcessor::create().expect("unable to create the img processor");
        Ok(Self {
            session,
            image_dimensions: 224,
            preprocesser: Arc::new(clip_image_processor)
        })
    }

    pub fn encode_image_batch(&self, image_documents: &[ImageDocument]) -> anyhow::Result<(Vec<EmbeddingResult>, Vec<EmbeddingFailure>), Box<dyn Error + Send + Sync>> {
        let mut embedding_failures = vec![];
        let mut uri_vecs = vec![];

        for image_doc in image_documents {
            match self.preprocesser.uri_to_clip_vector(image_doc.image_url, self.image_dimensions) {
                Ok(preprocessed) => {
                    let clip_vector = ClipVector { id: image_doc.id, clip_vector: preprocessed};
                    uri_vecs.push(clip_vector);
                }
                Err(_err) => {
                    let embedding_failure = EmbeddingFailure {id: image_doc.id, err: _err.to_string()};
                    embedding_failures.push(embedding_failure);
                }
            }
        }

        // expected format:  num_uris, 3, onnx_model.image_size, onnx_model.image_size
        let md_vecs = uri_vecs;
        let mut a = Array::<f32, _>::zeros((
            md_vecs.len() as usize,
            3 as usize,
            self.image_dimensions as usize,
            self.image_dimensions as usize,
        ));

        let outputs = self
            .session
            .run(inputs![a.view()]?)?;

        let embedding = outputs[outputs.len() - 1].extract_tensor::<f32>().unwrap().view().t().into_owned();
        let mut result: Vec<EmbeddingResult> = Vec::with_capacity(md_vecs.len());
        for idx in 0..md_vecs.len() {
            result.push(
                EmbeddingResult {
                    id: md_vecs[idx].id, 
                    embedding: embedding
                    .slice(s![idx..idx + 1, ..])
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                }
            )
        }
        Ok((result, embedding_failures))
    } 
} 
