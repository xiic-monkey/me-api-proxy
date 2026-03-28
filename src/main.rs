use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Router,
};
use reqwest::Client;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;
use tracing;

const DEFAULT_CONFIG: &str = r#"{
  "newapis": "https://newapis.xyz",
  "default": "http://demo.com"
}"#;

const HOP_BY_HOP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
];

#[derive(Clone)]
struct AppState {
    client: Client,
    routes: Arc<RwLock<HashMap<String, String>>>,
}

fn get_config_path() -> PathBuf {
    let home = dirs::home_dir().expect("Cannot find home directory");
    home.join(".me-api-proxy").join("apis.json")
}

fn load_routes() -> HashMap<String, String> {
    let config_path = get_config_path();

    // Create directory if not exists
    if let Some(parent) = config_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).expect("Failed to create config directory");
            tracing::info!("Created config directory: {:?}", parent);
        }
    }

    // Create file with default content if not exists
    if !config_path.exists() {
        std::fs::write(&config_path, DEFAULT_CONFIG).expect("Failed to create config file");
        tracing::info!("Created config file: {:?}", config_path);
    }

    // Read and parse config
    let content = std::fs::read_to_string(&config_path).expect("Failed to read config file");
    let json: Map<String, Value> =
        serde_json::from_str(&content).expect("Failed to parse config JSON");

    let mut routes = HashMap::new();
    for (key, value) in json {
        if let Some(url) = value.as_str() {
            routes.insert(key, url.to_string());
        }
    }

    tracing::info!("Loaded {} routes from {:?}", routes.len(), config_path);
    routes
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "me_api_proxy=info,tower_http=info".into()),
        )
        .init();

    let config_path = get_config_path();
    let routes = load_routes();
    let state = AppState {
        client: Client::new(),
        routes: Arc::new(RwLock::new(routes)),
    };

    // Start config hot-reload task
    let routes_clone = state.routes.clone();
    let config_path_clone = config_path.clone();
    tokio::spawn(async move {
        let mut last_modified: Option<SystemTime> = None;
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            if let Ok(meta) = std::fs::metadata(&config_path_clone) {
                if let Ok(modified) = meta.modified() {
                    if last_modified.is_none() || last_modified.unwrap() != modified {
                        last_modified = Some(modified);
                        // Reload config
                        let new_routes = load_routes();
                        *routes_clone.write().await = new_routes;
                        tracing::info!("Config reloaded from {:?}", config_path_clone);
                    }
                }
            }
        }
    });

    let app = Router::new()
        .fallback(proxy_handler)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = "0.0.0.0:8080";
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn proxy_handler(State(state): State<AppState>, req: Request) -> Response {
    let path = req.uri().path().to_string();
    let query = req.uri().query().map(|q| format!("?{q}")).unwrap_or_default();

    // Parse first segment as route name
    let segments: Vec<&str> = path.trim_start_matches('/').splitn(2, '/').collect();
    let route_name = segments.first().unwrap_or(&"");
    let remaining_path = if segments.len() > 1 {
        format!("/{}", segments[1])
    } else {
        String::from("/")
    };

    // Resolve upstream
    let upstream_base = {
        let routes = state.routes.read().await;
        routes.get(*route_name).cloned()
    };

    let upstream_base = match upstream_base {
        Some(url) => url,
        None => {
            return (
                StatusCode::NOT_FOUND,
                format!("Route '{}' not found. Available routes in ~/.me-api-proxy/apis.json", route_name),
            )
                .into_response();
        }
    };

    let full_url = format!(
        "{}{}{}",
        upstream_base.trim_end_matches('/'),
        remaining_path,
        query
    );

    tracing::info!("{} {} -> {}", req.method(), path, full_url);

    // Decompose incoming request
    let (parts, body) = req.into_parts();

    // Build outgoing request
    let mut outgoing = state.client.request(parts.method.clone(), &full_url);

    // Forward all headers except hop-by-hop
    for (key, value) in parts.headers.iter() {
        if !HOP_BY_HOP_HEADERS.contains(&key.as_str().to_lowercase().as_str()) {
            outgoing = outgoing.header(key.clone(), value.clone());
        }
    }

    // Stream request body
    let outgoing = outgoing.body(reqwest::Body::wrap_stream(body.into_data_stream()));

    // Send to upstream
    let upstream = match outgoing.send().await {
        Ok(res) => res,
        Err(e) => {
            tracing::error!("Upstream error: {}", e);
            return (StatusCode::BAD_GATEWAY, format!("Upstream error: {}", e)).into_response();
        }
    };

    // Build streaming response
    let mut builder = Response::builder().status(upstream.status());

    if let Some(headers) = builder.headers_mut() {
        for (k, v) in upstream.headers() {
            if !HOP_BY_HOP_HEADERS.contains(&k.as_str().to_lowercase().as_str()) {
                headers.insert(k.clone(), v.clone());
            }
        }
    }

    // Stream response body without buffering
    builder
        .body(Body::from_stream(upstream.bytes_stream()))
        .unwrap_or_else(|e| {
            tracing::error!("Failed to build response: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        })
}
