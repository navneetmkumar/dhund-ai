use bytes::Bytes;
use reqwest::header::HeaderMap;

pub async fn download_image(uri: &str) -> anyhow::Result<Bytes> {
    let client = reqwest::Client::new();
    let mut headers = HeaderMap::new();
    headers.insert("User-Agent", "Mozilla/5.0".parse().unwrap());
    let res = client.get(uri).headers(headers).send().await.unwrap();
    Ok(res.bytes().await.unwrap())
}

pub fn download_image_sync(uri: &str) -> anyhow::Result<Bytes> {
    let client = reqwest::blocking::Client::new();
    let mut headers = HeaderMap::new();
    headers.insert("User-Agent", "Mozilla/5.0".parse().unwrap());
    let res = client.get(uri).headers(headers).send().unwrap();
    Ok(res.bytes().unwrap())
}

