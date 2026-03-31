use axum::{
    body::Body,
    extract::{
        ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade},
        FromRequestParts, Request, State,
    },
    http::{header, HeaderMap, HeaderValue, Response, StatusCode},
    response::IntoResponse,
    Router,
};
use futures_util::{sink::SinkExt, stream::StreamExt};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tokio_tungstenite::tungstenite::{
    client::IntoClientRequest,
    protocol::{frame::coding::CloseCode as WsCloseCode, CloseFrame as WsCloseFrame, Message as WsMessage},
};
use tower_http::trace::TraceLayer;
use tracing::{error, info};
use url::Url;

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
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
];

const WS_HANDSHAKE_HEADERS: &[&str] = &[
    "host",
    "connection",
    "upgrade",
    "sec-websocket-key",
    "sec-websocket-version",
    "sec-websocket-extensions",
];

#[derive(Clone)]
struct AppState {
    routes: Arc<RwLock<HashMap<String, String>>>,
    client: reqwest::Client,
}

#[derive(Clone)]
struct RouteMatch {
    target: String,
    new_path: String,
}

fn get_config_path() -> PathBuf {
    let home = dirs::home_dir().expect("Cannot find home directory");
    home.join(".me-api-proxy").join("apis.json")
}

fn ensure_default_config() {
    let config_path = get_config_path();

    if let Some(parent) = config_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).expect("Failed to create config directory");
            info!("Created config directory: {:?}", parent);
        }
    }

    if !config_path.exists() {
        std::fs::write(&config_path, DEFAULT_CONFIG).expect("Failed to create config file");
        info!("Created config file: {:?}", config_path);
    }
}

fn load_routes() -> HashMap<String, String> {
    ensure_default_config();
    let config_path = get_config_path();
    let content = std::fs::read_to_string(&config_path).expect("Failed to read config file");
    let json: Map<String, Value> =
        serde_json::from_str(&content).expect("Failed to parse config JSON");

    let mut routes = HashMap::new();
    for (key, value) in json {
        if let Some(url) = value.as_str() {
            routes.insert(key, url.to_string());
        }
    }

    info!("Loaded {} routes from {:?}", routes.len(), config_path);
    routes
}

fn match_prefix(path: &str, prefix: &str) -> bool {
    path == prefix || path.starts_with(&format!("{prefix}/"))
}

fn find_target(path: &str, routes: &HashMap<String, String>) -> Option<RouteMatch> {
    let mut best: Option<(usize, RouteMatch)> = None;

    for (key, url) in routes {
        let prefix = format!("/{}", key.trim_matches('/'));
        if !match_prefix(path, &prefix) {
            continue;
        }

        let suffix = &path[prefix.len()..];
        let new_path = if suffix.is_empty() {
            "/".to_string()
        } else {
            suffix.to_string()
        };

        let candidate = RouteMatch {
            target: url.clone(),
            new_path,
        };

        match &best {
            Some((len, _)) if *len >= prefix.len() => {}
            _ => best = Some((prefix.len(), candidate)),
        }
    }

    best.map(|(_, value)| value)
}

fn build_upstream_url(target: &str, new_path: &str, query: Option<&str>) -> Result<Url, String> {
    let mut parsed = Url::parse(target).map_err(|e| format!("Invalid upstream URL: {e}"))?;
    let base_path = parsed.path().trim_end_matches('/');
    let suffix = if new_path.starts_with('/') {
        new_path.to_string()
    } else {
        format!("/{new_path}")
    };
    let combined_path = if base_path.is_empty() {
        suffix
    } else {
        format!("{base_path}{suffix}")
    };

    parsed.set_path(if combined_path.is_empty() {
        "/"
    } else {
        &combined_path
    });
    parsed.set_query(query);
    Ok(parsed)
}

fn should_skip_header(name: &str, skipped: &[&str]) -> bool {
    let lower = name.to_ascii_lowercase();
    skipped.contains(&lower.as_str())
}

fn set_host_header(headers: &mut HeaderMap, authority: &str) -> Result<(), String> {
    let value =
        HeaderValue::from_str(authority).map_err(|e| format!("Invalid host header '{authority}': {e}"))?;
    headers.insert(header::HOST, value);
    Ok(())
}

fn filtered_headers(source: &HeaderMap, excluded: &[&str]) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (name, value) in source {
        if !should_skip_header(name.as_str(), excluded) {
            headers.append(name.clone(), value.clone());
        }
    }
    headers
}

fn extract_ws_protocols(headers: &HeaderMap) -> Vec<String> {
    headers
        .get_all("sec-websocket-protocol")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn is_websocket_request(headers: &HeaderMap) -> bool {
    headers
        .get(header::UPGRADE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.eq_ignore_ascii_case("websocket"))
        .unwrap_or(false)
}

fn websocket_url(target: &str, new_path: &str, query: Option<&str>) -> Result<Url, String> {
    let mut url = build_upstream_url(target, new_path, query)?;
    match url.scheme() {
        "http" => url.set_scheme("ws").map_err(|_| "Failed to switch ws scheme".to_string())?,
        "https" => url
            .set_scheme("wss")
            .map_err(|_| "Failed to switch wss scheme".to_string())?,
        "ws" | "wss" => {}
        scheme => return Err(format!("Unsupported websocket target scheme: {scheme}")),
    }
    Ok(url)
}

fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to build reqwest client")
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
    let state = AppState {
        routes: Arc::new(RwLock::new(load_routes())),
        client: build_client(),
    };

    let routes = state.routes.clone();
    tokio::spawn(async move {
        let mut last_modified: Option<SystemTime> = None;
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            if let Ok(meta) = std::fs::metadata(&config_path) {
                if let Ok(modified) = meta.modified() {
                    if last_modified != Some(modified) {
                        last_modified = Some(modified);
                        *routes.write().await = load_routes();
                        info!("Config reloaded from {:?}", config_path);
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
    info!("Listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn proxy_handler(State(state): State<AppState>, req: Request) -> Response<Body> {
    let path = req.uri().path().to_string();
    let query = req.uri().query().map(ToOwned::to_owned);

    let route_match = {
        let routes = state.routes.read().await;
        find_target(&path, &routes)
    };

    let Some(route_match) = route_match else {
        return (StatusCode::NOT_FOUND, "No target found").into_response();
    };

    if is_websocket_request(req.headers()) {
        return proxy_websocket_request(state, req, route_match, query.as_deref()).await;
    }

    proxy_http_request(state, req, route_match, query.as_deref()).await
}

async fn proxy_http_request(
    state: AppState,
    req: Request,
    route_match: RouteMatch,
    query: Option<&str>,
) -> Response<Body> {
    let upstream_url = match build_upstream_url(&route_match.target, &route_match.new_path, query) {
        Ok(url) => url,
        Err(err) => return (StatusCode::BAD_GATEWAY, err).into_response(),
    };

    info!("{} {} -> {}", req.method(), req.uri().path(), upstream_url);

    let authority = upstream_url.authority().to_string();

    let (parts, body) = req.into_parts();
    let mut headers = filtered_headers(&parts.headers, HOP_BY_HOP_HEADERS);
    if let Err(err) = set_host_header(&mut headers, &authority) {
        return (StatusCode::BAD_GATEWAY, err).into_response();
    }

    let mut builder = state.client.request(parts.method, upstream_url);
    builder = builder.headers(headers);
    builder = builder.body(reqwest::Body::wrap_stream(body.into_data_stream()));

    let upstream = match builder.send().await {
        Ok(response) => response,
        Err(err) if err.is_timeout() => {
            return (StatusCode::GATEWAY_TIMEOUT, "Gateway Timeout").into_response();
        }
        Err(err) => {
            error!("HTTP proxy error: {err}");
            return (StatusCode::BAD_GATEWAY, format!("Bad Gateway: {err}")).into_response();
        }
    };

    let mut response = Response::builder().status(upstream.status());
    if let Some(headers_mut) = response.headers_mut() {
        for (name, value) in upstream.headers() {
            if !should_skip_header(name.as_str(), HOP_BY_HOP_HEADERS) {
                headers_mut.append(name.clone(), value.clone());
            }
        }
    }

    response
        .body(Body::from_stream(upstream.bytes_stream()))
        .unwrap_or_else(|err| {
            error!("Failed to build upstream response: {err}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        })
}

async fn proxy_websocket_request(
    state: AppState,
    req: Request,
    route_match: RouteMatch,
    query: Option<&str>,
) -> Response<Body> {
    let ws_url = match websocket_url(&route_match.target, &route_match.new_path, query) {
        Ok(url) => url,
        Err(err) => return (StatusCode::BAD_GATEWAY, err).into_response(),
    };

    let authority = ws_url.authority().to_string();

    let protocols = extract_ws_protocols(req.headers());
    let (mut parts, _) = req.into_parts();
    let upgrade = match WebSocketUpgrade::from_request_parts(&mut parts, &state).await {
        Ok(value) => value,
        Err(err) => return (StatusCode::BAD_REQUEST, err.to_string()).into_response(),
    };

    let response_protocols = protocols.clone();
    upgrade
        .protocols(response_protocols)
        .on_upgrade(move |socket| async move {
            if let Err(err) = proxy_websocket(socket, ws_url, authority, parts.headers, protocols).await
            {
                error!("WebSocket proxy error: {err}");
            }
        })
}

async fn proxy_websocket(
    downstream: WebSocket,
    ws_url: Url,
    authority: String,
    request_headers: HeaderMap,
    protocols: Vec<String>,
) -> Result<(), String> {
    let mut client_request = ws_url
        .as_str()
        .into_client_request()
        .map_err(|e| format!("Failed to build websocket request: {e}"))?;

    {
        let headers = client_request.headers_mut();
        for (name, value) in filtered_headers(&request_headers, WS_HANDSHAKE_HEADERS).iter() {
            headers.append(name.clone(), value.clone());
        }
        set_host_header(headers, &authority)?;
        if !protocols.is_empty() {
            let value = HeaderValue::from_str(&protocols.join(", "))
                .map_err(|e| format!("Invalid Sec-WebSocket-Protocol header: {e}"))?;
            headers.insert("Sec-WebSocket-Protocol", value);
        }
    }

    let (upstream, _) = tokio_tungstenite::connect_async(client_request)
        .await
        .map_err(|e| format!("Upstream websocket connect failed: {e}"))?;

    let (mut downstream_sink, mut downstream_stream) = downstream.split();
    let (mut upstream_sink, mut upstream_stream) = upstream.split();

    let client_to_upstream = async {
        while let Some(message) = downstream_stream.next().await {
            let message = message.map_err(|e| e.to_string())?;
            match message {
                Message::Text(text) => upstream_sink
                    .send(WsMessage::Text(text.to_string().into()))
                    .await
                    .map_err(|e| e.to_string())?,
                Message::Binary(data) => upstream_sink
                    .send(WsMessage::Binary(data))
                    .await
                    .map_err(|e| e.to_string())?,
                Message::Ping(data) => upstream_sink
                    .send(WsMessage::Ping(data))
                    .await
                    .map_err(|e| e.to_string())?,
                Message::Pong(data) => upstream_sink
                    .send(WsMessage::Pong(data))
                    .await
                    .map_err(|e| e.to_string())?,
                Message::Close(frame) => {
                    let frame = frame.map(|frame| WsCloseFrame {
                        code: WsCloseCode::from(u16::from(frame.code)),
                        reason: frame.reason.to_string().into(),
                    });
                    upstream_sink
                        .send(WsMessage::Close(frame))
                        .await
                        .map_err(|e| e.to_string())?;
                    break;
                }
            }
        }
        Ok::<(), String>(())
    };

    let upstream_to_client = async {
        while let Some(message) = upstream_stream.next().await {
            let message = message.map_err(|e| e.to_string())?;
            match message {
                WsMessage::Text(text) => downstream_sink
                    .send(Message::Text(text.to_string().into()))
                    .await
                    .map_err(|e| e.to_string())?,
                WsMessage::Binary(data) => downstream_sink
                    .send(Message::Binary(data))
                    .await
                    .map_err(|e| e.to_string())?,
                WsMessage::Ping(data) => downstream_sink
                    .send(Message::Ping(data))
                    .await
                    .map_err(|e| e.to_string())?,
                WsMessage::Pong(data) => downstream_sink
                    .send(Message::Pong(data))
                    .await
                    .map_err(|e| e.to_string())?,
                WsMessage::Close(frame) => {
                    let frame = frame.map(|frame| CloseFrame {
                        code: u16::from(frame.code).into(),
                        reason: frame.reason.to_string().into(),
                    });
                    downstream_sink
                        .send(Message::Close(frame))
                        .await
                        .map_err(|e| e.to_string())?;
                    break;
                }
                WsMessage::Frame(_) => {}
            }
        }
        Ok::<(), String>(())
    };

    let _ = tokio::join!(client_to_upstream, upstream_to_client);
    Ok(())
}
