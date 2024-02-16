use std::io::Cursor;
use std::sync::Arc;
use image::io::Reader as ImageReader;
use image::GenericImageView;
use std::thread::available_parallelism;
use libvips::ops::{jpegsave_buffer, resize, smartcrop_with_opts, Interesting};
use libvips::{VipsApp, VipsImage};
use tracing::warn;
use ndarray::{Array, Dim};
use tokio::{self, runtime::Handle};

use crate::utils::{download_image, download_image_sync};


pub struct ImageProcessor {
    libvips_app: Arc<VipsApp>, // Shared reference to the VipsApp
}

impl ImageProcessor {
    pub fn create() -> anyhow::Result<Self> {
        let libvips_app = Arc::new(VipsApp::new("image-processor", false).unwrap());
        let mut num_threads = 4;
        match available_parallelism() {
            Ok(ap) => num_threads = ap.get().try_into().unwrap(),
            Err(_err) => {
                warn!("unable to determine available_parallelism - defaulting to num_threads: 4");
            }
        }

        libvips_app.concurrency_set(num_threads);
        Ok(ImageProcessor {
            libvips_app : libvips_app
        })
    }

    // heavily inspired by:
    // https://github.com/openai/CLIP. MIT License, Copyright (c) 2021 OpenAI
    pub fn uri_to_clip_vector(&self, uri: &str, dimensions: i32) -> anyhow::Result<Array<f32, Dim<[usize; 3]>>> {
        let res;

        match Handle::try_current() {
            Ok(handle) => {
                let _guard = handle.enter();
                res = tokio::task::block_in_place(|| handle.block_on(download_image(uri)))?;
            }
            Err(_err) => {
                res = download_image_sync(uri)?;
            }
        };

        let maybe_processed_image = VipsImage::new_from_buffer(&res, "")
                                                        .and_then(|image| {
                                                            let scale = dimensions as f64 / i32::min(image.get_height(), image.get_width()) as f64;
                                                            resize(&image, scale)
                                                        })
                                                        .and_then(|image_scaled| smartcrop_with_opts(&image_scaled, dimensions, dimensions, &libvips::ops::SmartcropOptions {
                                                            interesting: Interesting::Centre,
                                                            attention_x: 0,
                                                            attention_y: 0,
                                                            premultiplied: false,
                                                        },));
        if let Err(err) = maybe_processed_image {
            anyhow::bail!("failed to create image and process from buffer due to {:?}", err)
        }        

        let processed_image = maybe_processed_image.unwrap();
        
        // Saving image
        let maybe_formatted = jpegsave_buffer(&processed_image);

        if let Err(format_error) = maybe_formatted {
            anyhow::bail!("saving image to jpeg buffer failed due to {:?}", format_error)
        }

        let formatted = maybe_formatted.unwrap();
        
        // getpoint(..) is *super* slow in libvips - so we are going to
        // use ImageReader
        let maybe_guessed_format = ImageReader::new(Cursor::new(formatted)).with_guessed_format();
        
        if let Err(format_guessing_error) = maybe_guessed_format {
            anyhow::bail!("Guessing format errored due to {:?}", format_guessing_error)
        }

        let guessed_format = maybe_guessed_format.unwrap();
        let maybe_decoded = guessed_format.decode();

        if let Err(decoding_error) = maybe_decoded {
            anyhow::bail!("Decoding errored due to {:?}", decoding_error)
        }

        let decoded = maybe_decoded.unwrap();
        let mut a = Array::<f32, _>::zeros((3, dimensions as usize, dimensions as usize));
        for i in 0..(dimensions as usize) {
            for j in 0..(dimensions as usize) {
                let p = decoded.get_pixel(i.try_into().unwrap(), j.try_into().unwrap());
                a[[0, i as usize, j as usize]] = p[0] as f32 / 255.0;
                a[[1, i as usize, j as usize]] = p[1] as f32 / 255.0;
                a[[2, i as usize, j as usize]] = p[2] as f32 / 255.0;
                a[[0, i as usize, j as usize]] =
                    (a[[0, i as usize, j as usize]] - 0.48145466) / 0.26862954;
                a[[1, i as usize, j as usize]] =
                    (a[[1, i as usize, j as usize]] - 0.4578275) / 0.26130258;
                a[[2, i as usize, j as usize]] =
                    (a[[2, i as usize, j as usize]] - 0.40821073) / 0.27577711;
            }
        }
        Ok(a)


    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_clip_resizing() {
        let processor = ImageProcessor::create().expect("unable to create the img processor");
        let uri = "https://images.unsplash.com/photo-1481349518771-20055b2a7b24?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2439&q=80";
        assert!(
            processor.uri_to_clip_vector(uri, 224).is_ok(),
            "unable to download and create multi-dimensional arr"
        )
    }
}