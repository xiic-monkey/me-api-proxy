use axum::{
    body::{to_bytes, Body},
    extract::{Json, Request, State},
    http::{header, HeaderMap, HeaderValue, Method, Response, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse},
    routing::{get, put},
    Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;
use tracing::{error, info};
use url::Url;

const ADMIN_HTML: &str = include_str!("admin.html");
const MAX_REQUEST_BODY_BYTES: usize = 20 * 1024 * 1024;
const CHAT_PATH: &str = "/v1/chat/completions";
const RESPONSES_PATH: &str = "/v1/responses";

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

#[derive(Clone)]
struct AppState {
    config: Arc<RwLock<OpenAiConfig>>,
    client: reqwest::Client,
    db_path: PathBuf,
    admin: AdminCredentials,
}

#[derive(Clone)]
struct AdminCredentials {
    username: String,
    password: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ProtocolMode {
    Chat,
    Responses,
    Both,
}

impl ProtocolMode {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Responses => "responses",
            Self::Both => "both",
        }
    }

    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "chat" => Ok(Self::Chat),
            "responses" => Ok(Self::Responses),
            "both" => Ok(Self::Both),
            _ => Err("protocol_mode must be chat, responses, or both".to_string()),
        }
    }

    fn supports(&self, path: &str) -> bool {
        matches!(
            (self, path),
            (Self::Chat, CHAT_PATH)
                | (Self::Responses, RESPONSES_PATH)
                | (Self::Both, CHAT_PATH)
                | (Self::Both, RESPONSES_PATH)
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct OpenAiConfig {
    base_url: String,
    api_key: String,
    protocol_mode: ProtocolMode,
    enabled: bool,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: String::new(),
            protocol_mode: ProtocolMode::Both,
            enabled: false,
        }
    }
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

fn get_config_dir() -> PathBuf {
    let home = dirs::home_dir().expect("Cannot find home directory");
    home.join(".me-api-proxy")
}

fn get_db_path() -> PathBuf {
    get_config_dir().join("config.db")
}

fn ensure_config_dir() {
    let config_dir = get_config_dir();
    if !config_dir.exists() {
        std::fs::create_dir_all(&config_dir).expect("Failed to create config directory");
        info!("Created config directory: {:?}", config_dir);
    }
}

fn open_db(path: &PathBuf) -> Result<Connection, String> {
    let conn = Connection::open(path).map_err(|err| format!("Failed to open database: {err}"))?;
    conn.execute_batch("PRAGMA foreign_keys = ON;")
        .map_err(|err| format!("Failed to enable foreign keys: {err}"))?;
    Ok(conn)
}

fn init_database(path: &PathBuf) -> Result<(), String> {
    let conn = open_db(path)?;
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS openai_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            base_url TEXT NOT NULL DEFAULT '',
            api_key TEXT NOT NULL DEFAULT '',
            protocol_mode TEXT NOT NULL DEFAULT 'both',
            enabled INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
        );

        INSERT OR IGNORE INTO openai_config (id, base_url, api_key, protocol_mode, enabled)
        VALUES (1, '', '', 'both', 0);
        ",
    )
    .map_err(|err| format!("Failed to initialize database: {err}"))?;
    Ok(())
}

fn load_config(path: &PathBuf) -> Result<OpenAiConfig, String> {
    let conn = open_db(path)?;
    let row = conn
        .query_row(
            "SELECT base_url, api_key, protocol_mode, enabled FROM openai_config WHERE id = 1",
            [],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)? != 0,
                ))
            },
        )
        .optional()
        .map_err(|err| format!("Failed to load config: {err}"))?;

    let Some((base_url, api_key, protocol_mode, enabled)) = row else {
        return Ok(OpenAiConfig::default());
    };

    Ok(OpenAiConfig {
        base_url,
        api_key,
        protocol_mode: ProtocolMode::from_str(&protocol_mode)?,
        enabled,
    })
}

fn save_config(path: &PathBuf, config: &OpenAiConfig) -> Result<(), String> {
    let conn = open_db(path)?;
    conn.execute(
        "UPDATE openai_config
         SET base_url = ?1,
             api_key = ?2,
             protocol_mode = ?3,
             enabled = ?4,
             updated_at = strftime('%s', 'now')
         WHERE id = 1",
        params![
            config.base_url,
            config.api_key,
            config.protocol_mode.as_str(),
            if config.enabled { 1 } else { 0 }
        ],
    )
    .map_err(|err| format!("Failed to save config: {err}"))?;
    Ok(())
}

fn validate_config(config: OpenAiConfig) -> Result<OpenAiConfig, String> {
    let base_url = config.base_url.trim().trim_end_matches('/').to_string();
    let api_key = config.api_key.trim().to_string();

    if base_url.is_empty() {
        return Err("base_url must not be empty".to_string());
    }
    if api_key.is_empty() {
        return Err("api_key must not be empty".to_string());
    }

    let parsed = Url::parse(&base_url).map_err(|err| format!("invalid base_url: {err}"))?;
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => return Err(format!("unsupported base_url scheme: {scheme}")),
    }

    Ok(OpenAiConfig {
        base_url,
        api_key,
        protocol_mode: config.protocol_mode,
        enabled: config.enabled,
    })
}

fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("Failed to build reqwest client")
}

fn load_admin_credentials() -> Result<AdminCredentials, String> {
    let username =
        std::env::var("ADMIN_USERNAME").map_err(|_| "ADMIN_USERNAME is required".to_string())?;
    let password =
        std::env::var("ADMIN_PASSWORD").map_err(|_| "ADMIN_PASSWORD is required".to_string())?;

    if username.is_empty() || password.is_empty() {
        return Err("ADMIN_USERNAME and ADMIN_PASSWORD must not be empty".to_string());
    }

    Ok(AdminCredentials { username, password })
}

fn parse_basic_auth(headers: &HeaderMap) -> Option<(String, String)> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let encoded = value.strip_prefix("Basic ")?;
    let decoded = BASE64.decode(encoded).ok()?;
    let decoded = String::from_utf8(decoded).ok()?;
    let (username, password) = decoded.split_once(':')?;
    Some((username.to_string(), password.to_string()))
}

fn unauthorized_response() -> Response<Body> {
    let mut response = (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static(r#"Basic realm="me-api-proxy-admin""#),
    );
    response
}

async fn admin_auth(State(state): State<AppState>, req: Request, next: Next) -> Response<Body> {
    let Some((username, password)) = parse_basic_auth(req.headers()) else {
        return unauthorized_response();
    };

    if username != state.admin.username || password != state.admin.password {
        return unauthorized_response();
    }

    next.run(req).await
}

fn json_error(status: StatusCode, message: impl Into<String>) -> Response<Body> {
    (
        status,
        Json(ErrorResponse {
            error: message.into(),
        }),
    )
        .into_response()
}

fn should_skip_header(name: &str, skipped: &[&str]) -> bool {
    let lower = name.to_ascii_lowercase();
    skipped.contains(&lower.as_str())
}

fn set_host_header(headers: &mut HeaderMap, authority: &str) -> Result<(), String> {
    let value = HeaderValue::from_str(authority)
        .map_err(|e| format!("Invalid host header '{authority}': {e}"))?;
    headers.insert(header::HOST, value);
    Ok(())
}

fn build_upstream_headers(
    source: &HeaderMap,
    authority: &str,
    api_key: &str,
) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::new();
    for (name, value) in source {
        if !should_skip_header(name.as_str(), HOP_BY_HOP_HEADERS)
            && !name.as_str().eq_ignore_ascii_case("host")
            && !name.as_str().eq_ignore_ascii_case("authorization")
        {
            headers.append(name.clone(), value.clone());
        }
    }

    set_host_header(&mut headers, authority)?;
    let auth = HeaderValue::from_str(&format!("Bearer {api_key}"))
        .map_err(|err| format!("Invalid upstream authorization header: {err}"))?;
    headers.insert(header::AUTHORIZATION, auth);
    Ok(headers)
}

fn build_upstream_url(base_url: &str, path: &str, query: Option<&str>) -> Result<Url, String> {
    let mut parsed = Url::parse(base_url).map_err(|err| format!("Invalid base_url: {err}"))?;
    let base_path = parsed.path().trim_end_matches('/');
    let combined_path = if base_path.is_empty() {
        path.to_string()
    } else {
        format!("{base_path}{path}")
    };

    parsed.set_path(&combined_path);
    parsed.set_query(query);
    Ok(parsed)
}

async fn admin_page() -> Html<&'static str> {
    Html(ADMIN_HTML)
}

async fn get_config(State(state): State<AppState>) -> Json<OpenAiConfig> {
    Json(state.config.read().await.clone())
}

async fn update_config(
    State(state): State<AppState>,
    Json(payload): Json<OpenAiConfig>,
) -> Response<Body> {
    let config = match validate_config(payload) {
        Ok(config) => config,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };

    if let Err(err) = save_config(&state.db_path, &config) {
        error!("Failed to save config: {err}");
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }

    *state.config.write().await = config;
    Json(json!({ "ok": true })).into_response()
}

fn protocol_name(path: &str) -> &'static str {
    match path {
        CHAT_PATH => "chat",
        RESPONSES_PATH => "responses",
        _ => "unknown",
    }
}

async fn proxy_openai(State(state): State<AppState>, req: Request) -> Response<Body> {
    if req.method() != Method::POST {
        return (StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed").into_response();
    }

    let path = req.uri().path().to_string();
    if path != CHAT_PATH && path != RESPONSES_PATH {
        return (StatusCode::NOT_FOUND, "Not Found").into_response();
    }

    let config = state.config.read().await.clone();
    if !config.enabled {
        return (StatusCode::SERVICE_UNAVAILABLE, "OpenAI proxy is disabled").into_response();
    }
    if config.base_url.is_empty() || config.api_key.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "OpenAI proxy is not configured",
        )
            .into_response();
    }
    if !config.protocol_mode.supports(&path) {
        return (
            StatusCode::BAD_REQUEST,
            format!("{} protocol is not enabled", protocol_name(&path)),
        )
            .into_response();
    }

    let query = req.uri().query().map(ToOwned::to_owned);
    let upstream_url = match build_upstream_url(&config.base_url, &path, query.as_deref()) {
        Ok(url) => url,
        Err(err) => return (StatusCode::BAD_GATEWAY, err).into_response(),
    };

    let authority = upstream_url.authority().to_string();
    let (parts, body) = req.into_parts();
    let buffered_body = match to_bytes(body, MAX_REQUEST_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("Request body exceeds {MAX_REQUEST_BODY_BYTES} bytes"),
            )
                .into_response();
        }
    };

    let headers = match build_upstream_headers(&parts.headers, &authority, &config.api_key) {
        Ok(headers) => headers,
        Err(err) => return (StatusCode::BAD_GATEWAY, err).into_response(),
    };

    info!("{} {} -> {}", parts.method, path, upstream_url);

    let upstream = match state
        .client
        .request(parts.method, upstream_url)
        .headers(headers)
        .body(buffered_body)
        .send()
        .await
    {
        Ok(response) => response,
        Err(err) if err.is_timeout() => {
            return (StatusCode::GATEWAY_TIMEOUT, "Gateway Timeout").into_response();
        }
        Err(err) => {
            error!("OpenAI proxy error: {err}");
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

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "me_api_proxy=info,tower_http=info".into()),
        )
        .init();

    ensure_config_dir();
    let db_path = get_db_path();
    init_database(&db_path).expect("Failed to initialize SQLite database");

    let admin = load_admin_credentials().expect("Failed to load admin credentials");
    let config = load_config(&db_path).expect("Failed to load OpenAI config");
    let state = AppState {
        config: Arc::new(RwLock::new(config)),
        client: build_client(),
        db_path,
        admin,
    };

    let admin_routes = Router::new()
        .route("/admin", get(admin_page))
        .route("/admin/api/config", get(get_config).put(update_config))
        .layer(middleware::from_fn_with_state(state.clone(), admin_auth));

    let app = Router::new()
        .merge(admin_routes)
        .route(CHAT_PATH, put(proxy_openai).post(proxy_openai))
        .route(RESPONSES_PATH, put(proxy_openai).post(proxy_openai))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = "0.0.0.0:8080";
    info!("Listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn protocol_mode_supports_expected_paths() {
        assert!(ProtocolMode::Chat.supports(CHAT_PATH));
        assert!(!ProtocolMode::Chat.supports(RESPONSES_PATH));
        assert!(ProtocolMode::Responses.supports(RESPONSES_PATH));
        assert!(ProtocolMode::Both.supports(CHAT_PATH));
        assert!(ProtocolMode::Both.supports(RESPONSES_PATH));
    }

    #[test]
    fn validate_config_normalizes_base_url() {
        let config = validate_config(OpenAiConfig {
            base_url: " https://api.example.com/ ".to_string(),
            api_key: " sk-test ".to_string(),
            protocol_mode: ProtocolMode::Both,
            enabled: true,
        })
        .unwrap();

        assert_eq!(config.base_url, "https://api.example.com");
        assert_eq!(config.api_key, "sk-test");
    }

    #[test]
    fn validate_config_rejects_bad_values() {
        assert!(validate_config(OpenAiConfig {
            base_url: "".to_string(),
            api_key: "sk".to_string(),
            protocol_mode: ProtocolMode::Both,
            enabled: true,
        })
        .is_err());

        assert!(validate_config(OpenAiConfig {
            base_url: "ftp://example.com".to_string(),
            api_key: "sk".to_string(),
            protocol_mode: ProtocolMode::Both,
            enabled: true,
        })
        .is_err());

        assert!(validate_config(OpenAiConfig {
            base_url: "https://example.com".to_string(),
            api_key: "".to_string(),
            protocol_mode: ProtocolMode::Both,
            enabled: true,
        })
        .is_err());
    }

    #[test]
    fn build_upstream_url_preserves_base_path_and_query() {
        let url = build_upstream_url(
            "https://gateway.example.com/openai",
            CHAT_PATH,
            Some("stream=true"),
        )
        .unwrap();

        assert_eq!(
            url.as_str(),
            "https://gateway.example.com/openai/v1/chat/completions?stream=true"
        );
    }

    #[test]
    fn build_upstream_headers_replaces_authorization() {
        let mut source = HeaderMap::new();
        source.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer client-key"),
        );
        source.insert(header::HOST, HeaderValue::from_static("proxy.local"));
        source.insert("x-request-id", HeaderValue::from_static("req-1"));

        let headers = build_upstream_headers(&source, "api.example.com", "upstream-key").unwrap();

        assert_eq!(
            headers.get(header::AUTHORIZATION).unwrap(),
            "Bearer upstream-key"
        );
        assert_eq!(headers.get(header::HOST).unwrap(), "api.example.com");
        assert_eq!(headers.get("x-request-id").unwrap(), "req-1");
    }

    #[test]
    fn init_database_creates_default_openai_config() {
        let path = std::env::temp_dir().join(format!(
            "me-api-proxy-openai-test-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        init_database(&path).unwrap();
        let config = load_config(&path).unwrap();

        assert_eq!(config, OpenAiConfig::default());
        let _ = fs::remove_file(path);
    }
}
