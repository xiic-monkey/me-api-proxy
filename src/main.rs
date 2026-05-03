use axum::{
    body::{to_bytes, Body, Bytes},
    extract::{Json, Path, Request, State},
    http::{header, HeaderMap, HeaderValue, Method, Response, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse},
    routing::{get, post},
    Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use chrono::{DateTime, Duration as ChronoDuration, Local, TimeZone};
use futures_util::{stream::BoxStream, StreamExt};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::io;
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
const TEMP_BAN_FAILS: u32 = 3;
const TEMP_BAN_MINUTES: i64 = 5;
const DAILY_BAN_FAILS: u32 = 5;

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
    key_state: Arc<RwLock<KeyRuntimeState>>,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpenAiEndpoint {
    Chat,
    Responses,
}

impl OpenAiEndpoint {
    fn from_path(path: &str) -> Option<Self> {
        match path {
            CHAT_PATH => Some(Self::Chat),
            RESPONSES_PATH => Some(Self::Responses),
            _ => None,
        }
    }

    fn path(self) -> &'static str {
        match self {
            Self::Chat => CHAT_PATH,
            Self::Responses => RESPONSES_PATH,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConversionMode {
    Direct,
    ChatToResponses,
    ResponsesToChat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UpstreamPlan {
    upstream_endpoint: OpenAiEndpoint,
    conversion: ConversionMode,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct OpenAiConfig {
    base_url: String,
    api_keys: Vec<String>,
    protocol_mode: ProtocolMode,
    enabled: bool,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_keys: Vec::new(),
            protocol_mode: ProtocolMode::Both,
            enabled: false,
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiConfigPayload {
    base_url: String,
    api_keys: Vec<String>,
    protocol_mode: ProtocolMode,
    enabled: bool,
}

#[derive(Debug, Serialize)]
struct OpenAiConfigResponse {
    base_url: String,
    api_keys: Vec<String>,
    protocol_mode: ProtocolMode,
    enabled: bool,
    key_statuses: Vec<KeyStatus>,
}

#[derive(Clone, Debug, Serialize)]
struct KeyStatus {
    index: usize,
    masked_key: String,
    fail_count: u32,
    banned: bool,
    ban_until: Option<i64>,
}

#[derive(Clone, Debug, Default)]
struct ApiKeyRuntime {
    fail_count: u32,
    ban_until: Option<DateTime<Local>>,
}

#[derive(Clone, Debug)]
struct KeyRuntimeState {
    last_reset_date: chrono::NaiveDate,
    keys: HashMap<String, ApiKeyRuntime>,
}

impl Default for KeyRuntimeState {
    fn default() -> Self {
        Self {
            last_reset_date: Local::now().date_naive(),
            keys: HashMap::new(),
        }
    }
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Clone)]
struct ProxyAttemptError {
    status: StatusCode,
    body: String,
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

fn table_columns(conn: &Connection, table: &str) -> Result<HashSet<String>, String> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(|err| format!("Failed to inspect {table} table: {err}"))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|err| format!("Failed to query {table} table info: {err}"))?;

    let mut columns = HashSet::new();
    for row in rows {
        columns.insert(row.map_err(|err| format!("Failed to decode {table} table info: {err}"))?);
    }
    Ok(columns)
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

        CREATE TABLE IF NOT EXISTS openai_api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 0,
            UNIQUE(api_key)
        );

        INSERT OR IGNORE INTO openai_config (id, base_url, api_key, protocol_mode, enabled)
        VALUES (1, '', '', 'both', 0);
        ",
    )
    .map_err(|err| format!("Failed to initialize database: {err}"))?;

    let columns = table_columns(&conn, "openai_config")?;
    if !columns.contains("api_key") {
        conn.execute(
            "ALTER TABLE openai_config ADD COLUMN api_key TEXT NOT NULL DEFAULT ''",
            [],
        )
        .map_err(|err| format!("Failed to add legacy api_key column: {err}"))?;
    }

    migrate_legacy_api_key(&conn)?;
    Ok(())
}

fn migrate_legacy_api_key(conn: &Connection) -> Result<(), String> {
    let existing_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM openai_api_keys", [], |row| row.get(0))
        .map_err(|err| format!("Failed to query API key count: {err}"))?;
    if existing_count > 0 {
        return Ok(());
    }

    let legacy_key: Option<String> = conn
        .query_row(
            "SELECT api_key FROM openai_config WHERE id = 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(|err| format!("Failed to load legacy API key: {err}"))?;

    if let Some(key) = legacy_key
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())
    {
        conn.execute(
            "INSERT OR IGNORE INTO openai_api_keys (api_key, sort_order) VALUES (?1, 0)",
            params![key],
        )
        .map_err(|err| format!("Failed to migrate legacy API key: {err}"))?;
    }

    Ok(())
}

fn load_api_keys(conn: &Connection) -> Result<Vec<String>, String> {
    let mut stmt = conn
        .prepare("SELECT api_key FROM openai_api_keys ORDER BY sort_order ASC, id ASC")
        .map_err(|err| format!("Failed to prepare API key query: {err}"))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|err| format!("Failed to query API keys: {err}"))?;

    let mut keys = Vec::new();
    for row in rows {
        keys.push(row.map_err(|err| format!("Failed to decode API key row: {err}"))?);
    }
    Ok(keys)
}

fn replace_api_keys(conn: &Connection, api_keys: &[String]) -> Result<(), String> {
    conn.execute("DELETE FROM openai_api_keys", [])
        .map_err(|err| format!("Failed to clear API keys: {err}"))?;

    let mut stmt = conn
        .prepare("INSERT INTO openai_api_keys (api_key, sort_order) VALUES (?1, ?2)")
        .map_err(|err| format!("Failed to prepare API key insert: {err}"))?;

    for (index, key) in api_keys.iter().enumerate() {
        stmt.execute(params![key, index as i64])
            .map_err(|err| format!("Failed to save API key: {err}"))?;
    }

    Ok(())
}

fn load_config(path: &PathBuf) -> Result<OpenAiConfig, String> {
    let conn = open_db(path)?;
    let row = conn
        .query_row(
            "SELECT base_url, protocol_mode, enabled FROM openai_config WHERE id = 1",
            [],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)? != 0,
                ))
            },
        )
        .optional()
        .map_err(|err| format!("Failed to load config: {err}"))?;

    let Some((base_url, protocol_mode, enabled)) = row else {
        return Ok(OpenAiConfig::default());
    };

    Ok(OpenAiConfig {
        base_url,
        api_keys: load_api_keys(&conn)?,
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
            config.api_keys.first().cloned().unwrap_or_default(),
            config.protocol_mode.as_str(),
            if config.enabled { 1 } else { 0 }
        ],
    )
    .map_err(|err| format!("Failed to save config: {err}"))?;
    replace_api_keys(&conn, &config.api_keys)?;
    Ok(())
}

fn normalize_api_keys(api_keys: &[String]) -> Result<Vec<String>, String> {
    let mut normalized = Vec::new();
    let mut seen = HashSet::new();

    for key in api_keys {
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        if seen.insert(key.to_string()) {
            normalized.push(key.to_string());
        }
    }

    if normalized.is_empty() {
        return Err("api_keys must not be empty".to_string());
    }

    Ok(normalized)
}

fn validate_config(payload: OpenAiConfigPayload) -> Result<OpenAiConfig, String> {
    let base_url = payload.base_url.trim().trim_end_matches('/').to_string();
    let api_keys = normalize_api_keys(&payload.api_keys)?;

    if base_url.is_empty() {
        return Err("base_url must not be empty".to_string());
    }

    let parsed = Url::parse(&base_url).map_err(|err| format!("invalid base_url: {err}"))?;
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => return Err(format!("unsupported base_url scheme: {scheme}")),
    }

    Ok(OpenAiConfig {
        base_url,
        api_keys,
        protocol_mode: payload.protocol_mode,
        enabled: payload.enabled,
    })
}

fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(300))
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
            && !name.as_str().eq_ignore_ascii_case("content-length")
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
    let base_path = parsed
        .path()
        .trim_end_matches('/')
        .strip_suffix("/v1")
        .unwrap_or_else(|| parsed.path().trim_end_matches('/'));
    let combined_path = if base_path.is_empty() {
        path.to_string()
    } else {
        format!("{base_path}{path}")
    };

    parsed.set_path(&combined_path);
    parsed.set_query(query);
    Ok(parsed)
}

fn upstream_plan(inbound: OpenAiEndpoint, upstream_mode: &ProtocolMode) -> UpstreamPlan {
    match (inbound, upstream_mode) {
        (OpenAiEndpoint::Chat, ProtocolMode::Both | ProtocolMode::Chat) => UpstreamPlan {
            upstream_endpoint: OpenAiEndpoint::Chat,
            conversion: ConversionMode::Direct,
        },
        (OpenAiEndpoint::Responses, ProtocolMode::Both | ProtocolMode::Responses) => UpstreamPlan {
            upstream_endpoint: OpenAiEndpoint::Responses,
            conversion: ConversionMode::Direct,
        },
        (OpenAiEndpoint::Chat, ProtocolMode::Responses) => UpstreamPlan {
            upstream_endpoint: OpenAiEndpoint::Responses,
            conversion: ConversionMode::ChatToResponses,
        },
        (OpenAiEndpoint::Responses, ProtocolMode::Chat) => UpstreamPlan {
            upstream_endpoint: OpenAiEndpoint::Chat,
            conversion: ConversionMode::ResponsesToChat,
        },
    }
}

fn is_stream_request(body: &[u8]) -> bool {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| value.get("stream").and_then(Value::as_bool).or(Some(false)))
        .unwrap_or(false)
}

fn copy_field(source: &Map<String, Value>, target: &mut Map<String, Value>, name: &str) {
    if let Some(value) = source.get(name) {
        target.insert(name.to_string(), value.clone());
    }
}

fn copy_renamed_field(
    source: &Map<String, Value>,
    target: &mut Map<String, Value>,
    from: &str,
    to: &str,
) {
    if let Some(value) = source.get(from) {
        target.insert(to.to_string(), value.clone());
    }
}

fn chat_request_to_responses(body: &[u8]) -> Result<Vec<u8>, String> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|err| format!("invalid Chat Completions request JSON: {err}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "Chat Completions request must be a JSON object".to_string())?;

    let messages = object
        .get("messages")
        .cloned()
        .ok_or_else(|| "Chat Completions request is missing messages".to_string())?;

    let mut target = Map::new();
    target.insert("input".to_string(), messages);

    for field in [
        "model",
        "stream",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "metadata",
        "user",
        "store",
    ] {
        copy_field(object, &mut target, field);
    }

    copy_field(object, &mut target, "instructions");
    copy_field(object, &mut target, "stop");
    if object.contains_key("max_completion_tokens") {
        copy_renamed_field(
            object,
            &mut target,
            "max_completion_tokens",
            "max_output_tokens",
        );
    } else {
        copy_renamed_field(object, &mut target, "max_tokens", "max_output_tokens");
    }

    serde_json::to_vec(&Value::Object(target))
        .map_err(|err| format!("failed to encode Responses request: {err}"))
}

fn has_responses_only_tool(value: &Value) -> bool {
    value
        .as_array()
        .map(|tools| {
            tools.iter().any(|tool| {
                tool.get("type")
                    .and_then(Value::as_str)
                    .map(|kind| kind != "function")
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn responses_content_to_chat_content(content: &Value) -> Value {
    let Some(parts) = content.as_array() else {
        return content.clone();
    };

    let mut converted = Vec::new();
    for part in parts {
        let Some(part_object) = part.as_object() else {
            converted.push(part.clone());
            continue;
        };
        match part_object.get("type").and_then(Value::as_str) {
            Some("input_text") | Some("output_text") => {
                converted.push(json!({
                    "type": "text",
                    "text": part_object.get("text").cloned().unwrap_or(Value::String(String::new()))
                }));
            }
            Some("input_image") => {
                converted.push(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": part_object
                            .get("image_url")
                            .or_else(|| part_object.get("file_id"))
                            .cloned()
                            .unwrap_or(Value::String(String::new()))
                    }
                }));
            }
            _ => converted.push(part.clone()),
        }
    }
    Value::Array(converted)
}

fn responses_input_to_chat_messages(input: &Value) -> Vec<Value> {
    match input {
        Value::String(text) => vec![json!({ "role": "user", "content": text })],
        Value::Array(items) => items
            .iter()
            .map(|item| {
                if let Some(object) = item.as_object() {
                    if let Some(role) = object.get("role").and_then(Value::as_str) {
                        return json!({
                            "role": role,
                            "content": object
                                .get("content")
                                .map(responses_content_to_chat_content)
                                .unwrap_or(Value::String(String::new()))
                        });
                    }
                }
                json!({ "role": "user", "content": item })
            })
            .collect(),
        _ => vec![json!({ "role": "user", "content": input })],
    }
}

fn responses_request_to_chat(body: &[u8]) -> Result<Vec<u8>, String> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|err| format!("invalid Responses request JSON: {err}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "Responses request must be a JSON object".to_string())?;

    if object
        .get("tools")
        .map(has_responses_only_tool)
        .unwrap_or(false)
    {
        return Err(
            "Responses built-in tools cannot be converted to a Chat Completions upstream"
                .to_string(),
        );
    }

    let input = object
        .get("input")
        .ok_or_else(|| "Responses request is missing input".to_string())?;
    let mut messages = Vec::new();
    if let Some(instructions) = object.get("instructions") {
        messages.push(json!({ "role": "system", "content": instructions }));
    }
    messages.extend(responses_input_to_chat_messages(input));

    let mut target = Map::new();
    target.insert("messages".to_string(), Value::Array(messages));

    for field in [
        "model",
        "stream",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "metadata",
        "user",
        "stop",
    ] {
        copy_field(object, &mut target, field);
    }
    copy_renamed_field(object, &mut target, "max_output_tokens", "max_tokens");

    serde_json::to_vec(&Value::Object(target))
        .map_err(|err| format!("failed to encode Chat Completions request: {err}"))
}

fn transform_request_body(body: &[u8], conversion: ConversionMode) -> Result<Vec<u8>, String> {
    match conversion {
        ConversionMode::Direct => Ok(body.to_vec()),
        ConversionMode::ChatToResponses => chat_request_to_responses(body),
        ConversionMode::ResponsesToChat => responses_request_to_chat(body),
    }
}

fn extract_chat_content(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.as_str())
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn chat_usage_to_responses_usage(usage: Option<&Value>) -> Value {
    let Some(usage) = usage else {
        return Value::Null;
    };
    json!({
        "input_tokens": usage.get("prompt_tokens").cloned().unwrap_or(Value::Number(0.into())),
        "output_tokens": usage.get("completion_tokens").cloned().unwrap_or(Value::Number(0.into())),
        "total_tokens": usage.get("total_tokens").cloned().unwrap_or(Value::Number(0.into()))
    })
}

fn chat_response_to_responses(body: &[u8]) -> Result<Vec<u8>, String> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|err| format!("invalid Chat Completions response JSON: {err}"))?;
    let first_choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let message = first_choice
        .get("message")
        .cloned()
        .unwrap_or_else(|| json!({ "role": "assistant", "content": "" }));
    let text = extract_chat_content(&message);

    let mut output = Vec::new();
    if !text.is_empty() {
        output.push(json!({
            "id": "msg_0",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": text, "annotations": [] }]
        }));
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            output.push(json!({
                "type": "function_call",
                "id": tool_call.get("id").cloned().unwrap_or(Value::Null),
                "call_id": tool_call.get("id").cloned().unwrap_or(Value::Null),
                "name": tool_call.pointer("/function/name").cloned().unwrap_or(Value::Null),
                "arguments": tool_call.pointer("/function/arguments").cloned().unwrap_or(Value::String(String::new()))
            }));
        }
    }

    serde_json::to_vec(&json!({
        "id": value.get("id").cloned().unwrap_or_else(|| Value::String("resp_proxy".to_string())),
        "object": "response",
        "created_at": value.get("created").cloned().unwrap_or(Value::Null),
        "model": value.get("model").cloned().unwrap_or(Value::Null),
        "status": "completed",
        "output": output,
        "output_text": text,
        "usage": chat_usage_to_responses_usage(value.get("usage"))
    }))
    .map_err(|err| format!("failed to encode Responses response: {err}"))
}

fn responses_output_text(value: &Value) -> String {
    if let Some(text) = value.get("output_text").and_then(Value::as_str) {
        return text.to_string();
    }

    let mut text = String::new();
    if let Some(output) = value.get("output").and_then(Value::as_array) {
        for item in output {
            if let Some(content) = item.get("content").and_then(Value::as_array) {
                for part in content {
                    if matches!(
                        part.get("type").and_then(Value::as_str),
                        Some("output_text")
                    ) {
                        if let Some(part_text) = part.get("text").and_then(Value::as_str) {
                            text.push_str(part_text);
                        }
                    }
                }
            }
        }
    }
    text
}

fn responses_usage_to_chat_usage(usage: Option<&Value>) -> Value {
    let Some(usage) = usage else {
        return Value::Null;
    };
    json!({
        "prompt_tokens": usage.get("input_tokens").cloned().unwrap_or(Value::Number(0.into())),
        "completion_tokens": usage.get("output_tokens").cloned().unwrap_or(Value::Number(0.into())),
        "total_tokens": usage.get("total_tokens").cloned().unwrap_or(Value::Number(0.into()))
    })
}

fn responses_response_to_chat(body: &[u8]) -> Result<Vec<u8>, String> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|err| format!("invalid Responses response JSON: {err}"))?;
    let text = responses_output_text(&value);

    serde_json::to_vec(&json!({
        "id": value.get("id").cloned().unwrap_or_else(|| Value::String("chatcmpl_proxy".to_string())),
        "object": "chat.completion",
        "created": value.get("created_at").cloned().unwrap_or(Value::Null),
        "model": value.get("model").cloned().unwrap_or(Value::Null),
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop"
        }],
        "usage": responses_usage_to_chat_usage(value.get("usage"))
    }))
    .map_err(|err| format!("failed to encode Chat Completions response: {err}"))
}

fn transform_response_body(body: &[u8], conversion: ConversionMode) -> Result<Vec<u8>, String> {
    match conversion {
        ConversionMode::Direct => Ok(body.to_vec()),
        ConversionMode::ChatToResponses => responses_response_to_chat(body),
        ConversionMode::ResponsesToChat => chat_response_to_responses(body),
    }
}

fn response_headers(
    source: &HeaderMap,
    body_was_transformed: bool,
    content_type: Option<&'static str>,
) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::new();
    for (name, value) in source {
        if should_skip_header(name.as_str(), HOP_BY_HOP_HEADERS) {
            continue;
        }
        if body_was_transformed
            && (name.as_str().eq_ignore_ascii_case("content-length")
                || name.as_str().eq_ignore_ascii_case("content-encoding"))
        {
            continue;
        }
        headers.append(name.clone(), value.clone());
    }
    if let Some(content_type) = content_type {
        headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    }
    Ok(headers)
}

struct SseTransformState {
    stream: BoxStream<'static, Result<Bytes, reqwest::Error>>,
    buffer: String,
    conversion: ConversionMode,
    completed_sent: bool,
}

fn pop_sse_event(buffer: &mut String) -> Option<String> {
    let pos = buffer.find("\n\n")?;
    let raw = buffer.drain(..pos + 2).collect::<String>();
    Some(raw.trim().to_string())
}

fn sse_data_payload(event: &str) -> Option<String> {
    let mut lines = Vec::new();
    for line in event.lines() {
        let line = line.trim_end();
        if let Some(data) = line.strip_prefix("data:") {
            lines.push(data.trim_start());
        }
    }
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

fn sse_event(name: &str, data: Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

fn chat_chunk(id: &str, model: &str, delta: Value, finish_reason: Value) -> String {
    format!(
        "data: {}\n\n",
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": chrono::Utc::now().timestamp(),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        })
    )
}

fn chat_sse_to_responses_event(data: &str, completed_sent: &mut bool) -> Option<String> {
    if data == "[DONE]" {
        if *completed_sent {
            return None;
        }
        *completed_sent = true;
        return Some(sse_event(
            "response.completed",
            json!({
                "type": "response.completed",
                "response": { "id": "resp_proxy", "object": "response", "status": "completed" }
            }),
        ));
    }

    let value: Value = serde_json::from_str(data).ok()?;
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())?;
    let delta = choice.get("delta").cloned().unwrap_or_else(|| json!({}));
    let mut out = String::new();

    if let Some(content) = delta.get("content").and_then(Value::as_str) {
        if !content.is_empty() {
            out.push_str(&sse_event(
                "response.output_text.delta",
                json!({
                    "type": "response.output_text.delta",
                    "item_id": "msg_0",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": content
                }),
            ));
        }
    }

    if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            if let Some(arguments) = tool_call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
            {
                out.push_str(&sse_event(
                    "response.function_call_arguments.delta",
                    json!({
                        "type": "response.function_call_arguments.delta",
                        "item_id": tool_call.get("id").cloned().unwrap_or(Value::String("call_0".to_string())),
                        "output_index": tool_call.get("index").cloned().unwrap_or(Value::Number(0.into())),
                        "delta": arguments
                    }),
                ));
            }
        }
    }

    if choice
        .get("finish_reason")
        .is_some_and(|reason| !reason.is_null())
        && !*completed_sent
    {
        *completed_sent = true;
        out.push_str(&sse_event(
            "response.completed",
            json!({
                "type": "response.completed",
                "response": {
                    "id": value.get("id").cloned().unwrap_or(Value::String("resp_proxy".to_string())),
                    "object": "response",
                    "status": "completed",
                    "model": value.get("model").cloned().unwrap_or(Value::Null)
                }
            }),
        ));
    }

    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn responses_sse_to_chat_event(data: &str, completed_sent: &mut bool) -> Option<String> {
    if data == "[DONE]" {
        if *completed_sent {
            return None;
        }
        *completed_sent = true;
        return Some("data: [DONE]\n\n".to_string());
    }

    let value: Value = serde_json::from_str(data).ok()?;
    let event_type = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match event_type {
        "response.output_text.delta" => {
            let delta = value
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default();
            Some(chat_chunk(
                "chatcmpl_proxy",
                "",
                json!({ "content": delta }),
                Value::Null,
            ))
        }
        "response.function_call_arguments.delta" => {
            let delta = value
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default();
            Some(chat_chunk(
                "chatcmpl_proxy",
                "",
                json!({
                    "tool_calls": [{
                        "index": value.get("output_index").cloned().unwrap_or(Value::Number(0.into())),
                        "id": value.get("item_id").cloned().unwrap_or(Value::String("call_0".to_string())),
                        "type": "function",
                        "function": { "arguments": delta }
                    }]
                }),
                Value::Null,
            ))
        }
        "response.completed" | "response.failed" | "response.cancelled" | "response.incomplete" => {
            if *completed_sent {
                None
            } else {
                *completed_sent = true;
                Some(format!(
                    "{}data: [DONE]\n\n",
                    chat_chunk(
                        "chatcmpl_proxy",
                        "",
                        json!({}),
                        Value::String("stop".to_string())
                    )
                ))
            }
        }
        _ => None,
    }
}

fn transform_sse_event(
    event: &str,
    conversion: ConversionMode,
    completed_sent: &mut bool,
) -> Option<String> {
    let data = sse_data_payload(event)?;
    match conversion {
        ConversionMode::Direct => Some(format!("{event}\n\n")),
        ConversionMode::ChatToResponses => responses_sse_to_chat_event(&data, completed_sent),
        ConversionMode::ResponsesToChat => chat_sse_to_responses_event(&data, completed_sent),
    }
}

fn transformed_sse_body(response: reqwest::Response, conversion: ConversionMode) -> Body {
    let state = SseTransformState {
        stream: response.bytes_stream().boxed(),
        buffer: String::new(),
        conversion,
        completed_sent: false,
    };

    let stream = futures_util::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event) = pop_sse_event(&mut state.buffer) {
                if let Some(output) =
                    transform_sse_event(&event, state.conversion, &mut state.completed_sent)
                {
                    return Some((Ok::<Bytes, io::Error>(Bytes::from(output)), state));
                }
                continue;
            }

            match state.stream.next().await {
                Some(Ok(bytes)) => {
                    let chunk = String::from_utf8_lossy(&bytes).replace("\r\n", "\n");
                    state.buffer.push_str(&chunk);
                }
                Some(Err(err)) => {
                    return Some((
                        Err(io::Error::new(io::ErrorKind::Other, err.to_string())),
                        state,
                    ));
                }
                None => {
                    if !state.buffer.trim().is_empty() {
                        let event = std::mem::take(&mut state.buffer);
                        if let Some(output) =
                            transform_sse_event(&event, state.conversion, &mut state.completed_sent)
                        {
                            return Some((Ok(Bytes::from(output)), state));
                        }
                    }
                    return None;
                }
            }
        }
    });

    Body::from_stream(stream)
}

fn next_day_zero(now: DateTime<Local>) -> DateTime<Local> {
    let tomorrow = now
        .date_naive()
        .succ_opt()
        .unwrap_or_else(|| now.date_naive());
    let naive = tomorrow.and_hms_opt(0, 0, 0).unwrap();
    Local
        .from_local_datetime(&naive)
        .single()
        .unwrap_or_else(|| Local.from_utc_datetime(&naive))
}

fn ensure_runtime_day(state: &mut KeyRuntimeState, now: DateTime<Local>) {
    let today = now.date_naive();
    if state.last_reset_date != today {
        state.keys.clear();
        state.last_reset_date = today;
    }
}

fn mark_key_fail_in_state(api_key: &str, state: &mut KeyRuntimeState, now: DateTime<Local>) {
    ensure_runtime_day(state, now);
    let entry = state.keys.entry(api_key.to_string()).or_default();
    entry.fail_count += 1;
    if entry.fail_count >= DAILY_BAN_FAILS {
        entry.ban_until = Some(next_day_zero(now));
    } else if entry.fail_count >= TEMP_BAN_FAILS {
        entry.ban_until = Some(now + ChronoDuration::minutes(TEMP_BAN_MINUTES));
    }
}

fn mark_key_success_in_state(api_key: &str, state: &mut KeyRuntimeState, now: DateTime<Local>) {
    ensure_runtime_day(state, now);
    state.keys.remove(api_key);
}

fn available_api_keys_in_state(
    api_keys: &[String],
    state: &mut KeyRuntimeState,
    now: DateTime<Local>,
) -> Vec<String> {
    ensure_runtime_day(state, now);
    let mut available = Vec::new();

    for key in api_keys {
        let expired = state
            .keys
            .get(key)
            .and_then(|entry| entry.ban_until)
            .map(|until| until <= now)
            .unwrap_or(false);
        if expired {
            state.keys.remove(key);
        }

        let banned = state
            .keys
            .get(key)
            .and_then(|entry| entry.ban_until)
            .map(|until| until > now)
            .unwrap_or(false);
        if !banned {
            available.push(key.clone());
        }
    }

    available
}

fn mask_key(api_key: &str) -> String {
    let chars: Vec<char> = api_key.chars().collect();
    if chars.len() <= 10 {
        return api_key.to_string();
    }
    let start: String = chars.iter().take(6).collect();
    let end: String = chars.iter().skip(chars.len() - 4).collect();
    format!("{start}...{end}")
}

fn key_statuses_for_config(
    config: &OpenAiConfig,
    state: &mut KeyRuntimeState,
    now: DateTime<Local>,
) -> Vec<KeyStatus> {
    ensure_runtime_day(state, now);
    config
        .api_keys
        .iter()
        .enumerate()
        .map(|(index, key)| {
            let entry = state.keys.get(key);
            let ban_until = entry.and_then(|entry| entry.ban_until);
            KeyStatus {
                index,
                masked_key: mask_key(key),
                fail_count: entry.map(|entry| entry.fail_count).unwrap_or(0),
                banned: ban_until.map(|until| until > now).unwrap_or(false),
                ban_until: ban_until.map(|until| until.timestamp()),
            }
        })
        .collect()
}

async fn available_api_keys(state: &AppState, config: &OpenAiConfig) -> Vec<String> {
    let mut runtime = state.key_state.write().await;
    available_api_keys_in_state(&config.api_keys, &mut runtime, Local::now())
}

async fn mark_key_fail(state: &AppState, api_key: &str) {
    let mut runtime = state.key_state.write().await;
    mark_key_fail_in_state(api_key, &mut runtime, Local::now());
}

async fn mark_key_success(state: &AppState, api_key: &str) {
    let mut runtime = state.key_state.write().await;
    mark_key_success_in_state(api_key, &mut runtime, Local::now());
}

async fn config_response(state: &AppState) -> OpenAiConfigResponse {
    let config = state.config.read().await.clone();
    let mut runtime = state.key_state.write().await;
    let key_statuses = key_statuses_for_config(&config, &mut runtime, Local::now());
    OpenAiConfigResponse {
        base_url: config.base_url,
        api_keys: config.api_keys,
        protocol_mode: config.protocol_mode,
        enabled: config.enabled,
        key_statuses,
    }
}

async fn admin_page() -> Html<&'static str> {
    Html(ADMIN_HTML)
}

async fn get_config(State(state): State<AppState>) -> Json<OpenAiConfigResponse> {
    Json(config_response(&state).await)
}

async fn update_config(
    State(state): State<AppState>,
    Json(payload): Json<OpenAiConfigPayload>,
) -> Response<Body> {
    let config = match validate_config(payload) {
        Ok(config) => config,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };

    if let Err(err) = save_config(&state.db_path, &config) {
        error!("Failed to save config: {err}");
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }

    {
        let mut runtime = state.key_state.write().await;
        runtime
            .keys
            .retain(|key, _| config.api_keys.iter().any(|api_key| api_key == key));
    }
    *state.config.write().await = config;
    Json(json!({ "ok": true })).into_response()
}

async fn unban_key(State(state): State<AppState>, Path(index): Path<usize>) -> Response<Body> {
    let config = state.config.read().await.clone();
    let Some(key) = config.api_keys.get(index) else {
        return json_error(StatusCode::NOT_FOUND, "key index not found");
    };

    state.key_state.write().await.keys.remove(key);
    Json(json!({ "ok": true })).into_response()
}

async fn unban_all_keys(State(state): State<AppState>) -> Response<Body> {
    state.key_state.write().await.keys.clear();
    Json(json!({ "ok": true })).into_response()
}

fn proxy_attempt_error(status: StatusCode, body: impl Into<String>) -> ProxyAttemptError {
    ProxyAttemptError {
        status,
        body: body.into(),
    }
}

async fn proxy_openai(State(state): State<AppState>, req: Request) -> Response<Body> {
    if req.method() != Method::POST {
        return (StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed").into_response();
    }

    let path = req.uri().path().to_string();
    let Some(inbound_endpoint) = OpenAiEndpoint::from_path(&path) else {
        return (StatusCode::NOT_FOUND, "Not Found").into_response();
    };

    let config = state.config.read().await.clone();
    if !config.enabled {
        return (StatusCode::SERVICE_UNAVAILABLE, "OpenAI proxy is disabled").into_response();
    }
    if config.base_url.is_empty() || config.api_keys.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "OpenAI proxy is not configured",
        )
            .into_response();
    }
    let upstream_plan = upstream_plan(inbound_endpoint, &config.protocol_mode);

    let query = req.uri().query().map(ToOwned::to_owned);
    let upstream_url = match build_upstream_url(
        &config.base_url,
        upstream_plan.upstream_endpoint.path(),
        query.as_deref(),
    ) {
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
    let is_streaming = is_stream_request(&buffered_body);
    let upstream_body = match transform_request_body(&buffered_body, upstream_plan.conversion) {
        Ok(body) => body,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };

    let available_keys = available_api_keys(&state, &config).await;
    if available_keys.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "All API keys are currently banned",
        )
            .into_response();
    }

    let mut last_error: Option<ProxyAttemptError> = None;
    for api_key in available_keys {
        let headers = match build_upstream_headers(&parts.headers, &authority, &api_key) {
            Ok(headers) => headers,
            Err(err) => return (StatusCode::BAD_GATEWAY, err).into_response(),
        };

        info!(
            "{} {} -> {}",
            parts.method,
            path,
            upstream_plan.upstream_endpoint.path()
        );
        let upstream = state
            .client
            .request(parts.method.clone(), upstream_url.clone())
            .headers(headers)
            .body(upstream_body.clone())
            .send()
            .await;

        match upstream {
            Ok(response)
                if response.status().is_client_error() || response.status().is_server_error() =>
            {
                let status = response.status();
                mark_key_fail(&state, &api_key).await;
                last_error = Some(proxy_attempt_error(
                    status,
                    format!("Upstream returned {}", status.as_u16()),
                ));
            }
            Ok(response) => {
                mark_key_success(&state, &api_key).await;
                let mut downstream = Response::builder().status(response.status());
                if let Some(headers_mut) = downstream.headers_mut() {
                    match response_headers(
                        response.headers(),
                        upstream_plan.conversion != ConversionMode::Direct,
                        if upstream_plan.conversion != ConversionMode::Direct {
                            Some(if is_streaming {
                                "text/event-stream"
                            } else {
                                "application/json"
                            })
                        } else {
                            None
                        },
                    ) {
                        Ok(headers) => headers_mut.extend(headers),
                        Err(err) => return (StatusCode::BAD_GATEWAY, err).into_response(),
                    }
                }

                if upstream_plan.conversion != ConversionMode::Direct && is_streaming {
                    return downstream
                        .body(transformed_sse_body(response, upstream_plan.conversion))
                        .unwrap_or_else(|err| {
                            error!("Failed to build transformed stream response: {err}");
                            StatusCode::INTERNAL_SERVER_ERROR.into_response()
                        });
                }

                if upstream_plan.conversion != ConversionMode::Direct {
                    let body = match response.bytes().await {
                        Ok(bytes) => bytes,
                        Err(err) => {
                            return (
                                StatusCode::BAD_GATEWAY,
                                format!("Failed to read upstream response: {err}"),
                            )
                                .into_response();
                        }
                    };
                    let body = match transform_response_body(&body, upstream_plan.conversion) {
                        Ok(body) => body,
                        Err(err) => return (StatusCode::BAD_GATEWAY, err).into_response(),
                    };
                    return downstream.body(Body::from(body)).unwrap_or_else(|err| {
                        error!("Failed to build transformed upstream response: {err}");
                        StatusCode::INTERNAL_SERVER_ERROR.into_response()
                    });
                }

                return downstream
                    .body(Body::from_stream(response.bytes_stream()))
                    .unwrap_or_else(|err| {
                        error!("Failed to build upstream response: {err}");
                        StatusCode::INTERNAL_SERVER_ERROR.into_response()
                    });
            }
            Err(err) if err.is_timeout() => {
                mark_key_fail(&state, &api_key).await;
                last_error = Some(proxy_attempt_error(
                    StatusCode::GATEWAY_TIMEOUT,
                    "Gateway Timeout",
                ));
            }
            Err(err) => {
                mark_key_fail(&state, &api_key).await;
                last_error = Some(proxy_attempt_error(
                    StatusCode::BAD_GATEWAY,
                    format!("Bad Gateway: {err}"),
                ));
            }
        }
    }

    let remaining = available_api_keys(&state, &config).await;
    if remaining.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "All API keys are banned, please try again later",
        )
            .into_response();
    }

    if let Some(error) = last_error {
        return (error.status, error.body).into_response();
    }

    (
        StatusCode::SERVICE_UNAVAILABLE,
        "All API keys failed".to_string(),
    )
        .into_response()
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
        key_state: Arc::new(RwLock::new(KeyRuntimeState::default())),
        client: build_client(),
        db_path,
        admin,
    };

    let admin_routes = Router::new()
        .route("/admin", get(admin_page))
        .route("/admin/api/config", get(get_config).put(update_config))
        .route("/admin/api/keys/{index}/unban", post(unban_key))
        .route("/admin/api/keys/unban-all", post(unban_all_keys))
        .layer(middleware::from_fn_with_state(state.clone(), admin_auth));

    let app = Router::new()
        .merge(admin_routes)
        .route(CHAT_PATH, post(proxy_openai))
        .route(RESPONSES_PATH, post(proxy_openai))
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
    fn protocol_mode_selects_upstream_endpoint() {
        assert_eq!(
            upstream_plan(OpenAiEndpoint::Chat, &ProtocolMode::Both),
            UpstreamPlan {
                upstream_endpoint: OpenAiEndpoint::Chat,
                conversion: ConversionMode::Direct
            }
        );
        assert_eq!(
            upstream_plan(OpenAiEndpoint::Responses, &ProtocolMode::Chat),
            UpstreamPlan {
                upstream_endpoint: OpenAiEndpoint::Chat,
                conversion: ConversionMode::ResponsesToChat
            }
        );
        assert_eq!(
            upstream_plan(OpenAiEndpoint::Chat, &ProtocolMode::Responses),
            UpstreamPlan {
                upstream_endpoint: OpenAiEndpoint::Responses,
                conversion: ConversionMode::ChatToResponses
            }
        );
    }

    #[test]
    fn validate_config_normalizes_base_url_and_keys() {
        let config = validate_config(OpenAiConfigPayload {
            base_url: " https://api.example.com/ ".to_string(),
            api_keys: vec![
                " sk-one ".to_string(),
                "".to_string(),
                "sk-one".to_string(),
                "sk-two".to_string(),
            ],
            protocol_mode: ProtocolMode::Both,
            enabled: true,
        })
        .unwrap();

        assert_eq!(config.base_url, "https://api.example.com");
        assert_eq!(
            config.api_keys,
            vec!["sk-one".to_string(), "sk-two".to_string()]
        );
    }

    #[test]
    fn validate_config_rejects_bad_values() {
        assert!(validate_config(OpenAiConfigPayload {
            base_url: "".to_string(),
            api_keys: vec!["sk".to_string()],
            protocol_mode: ProtocolMode::Both,
            enabled: true,
        })
        .is_err());

        assert!(validate_config(OpenAiConfigPayload {
            base_url: "ftp://example.com".to_string(),
            api_keys: vec!["sk".to_string()],
            protocol_mode: ProtocolMode::Both,
            enabled: true,
        })
        .is_err());

        assert!(validate_config(OpenAiConfigPayload {
            base_url: "https://example.com".to_string(),
            api_keys: vec!["".to_string()],
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
    fn build_upstream_url_deduplicates_trailing_v1() {
        let root_url =
            build_upstream_url("https://api.example.com/v1", RESPONSES_PATH, None).unwrap();
        let gateway_url =
            build_upstream_url("https://gateway.example.com/openai/v1", CHAT_PATH, None).unwrap();

        assert_eq!(root_url.as_str(), "https://api.example.com/v1/responses");
        assert_eq!(
            gateway_url.as_str(),
            "https://gateway.example.com/openai/v1/chat/completions"
        );
    }

    #[test]
    fn converts_chat_request_to_responses_request() {
        let body = br#"{
            "model":"gpt-test",
            "messages":[{"role":"user","content":"hi"}],
            "max_tokens":128,
            "stream":true
        }"#;

        let converted: Value =
            serde_json::from_slice(&chat_request_to_responses(body).unwrap()).unwrap();

        assert_eq!(converted["model"], "gpt-test");
        assert_eq!(converted["input"][0]["role"], "user");
        assert_eq!(converted["max_output_tokens"], 128);
        assert_eq!(converted["stream"], true);
    }

    #[test]
    fn converts_responses_request_to_chat_request() {
        let body = br#"{
            "model":"gpt-test",
            "instructions":"be brief",
            "input":"hello",
            "max_output_tokens":64
        }"#;

        let converted: Value =
            serde_json::from_slice(&responses_request_to_chat(body).unwrap()).unwrap();

        assert_eq!(converted["model"], "gpt-test");
        assert_eq!(converted["messages"][0]["role"], "system");
        assert_eq!(converted["messages"][1]["role"], "user");
        assert_eq!(converted["messages"][1]["content"], "hello");
        assert_eq!(converted["max_tokens"], 64);
    }

    #[test]
    fn responses_to_chat_rejects_builtin_tools() {
        let body = br#"{
            "model":"gpt-test",
            "input":"hello",
            "tools":[{"type":"web_search_preview"}]
        }"#;

        assert!(responses_request_to_chat(body).is_err());
    }

    #[test]
    fn converts_chat_response_to_responses_response() {
        let body = br#"{
            "id":"chatcmpl_1",
            "created":123,
            "model":"gpt-test",
            "choices":[{"message":{"role":"assistant","content":"hello"}}],
            "usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
        }"#;

        let converted: Value =
            serde_json::from_slice(&chat_response_to_responses(body).unwrap()).unwrap();

        assert_eq!(converted["object"], "response");
        assert_eq!(converted["output_text"], "hello");
        assert_eq!(converted["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(converted["usage"]["input_tokens"], 2);
    }

    #[test]
    fn converts_responses_response_to_chat_response() {
        let body = br#"{
            "id":"resp_1",
            "created_at":123,
            "model":"gpt-test",
            "output_text":"hello",
            "usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}
        }"#;

        let converted: Value =
            serde_json::from_slice(&responses_response_to_chat(body).unwrap()).unwrap();

        assert_eq!(converted["object"], "chat.completion");
        assert_eq!(converted["choices"][0]["message"]["content"], "hello");
        assert_eq!(converted["usage"]["prompt_tokens"], 2);
    }

    #[test]
    fn converts_sse_events_between_protocols() {
        let mut completed = false;
        let responses_event = responses_sse_to_chat_event(
            r#"{"type":"response.output_text.delta","delta":"hi"}"#,
            &mut completed,
        )
        .unwrap();
        assert!(responses_event.contains(r#""content":"hi""#));

        let chat_event = chat_sse_to_responses_event(
            r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
            &mut completed,
        )
        .unwrap();
        assert!(chat_event.contains("response.output_text.delta"));
        assert!(chat_event.contains(r#""delta":"hi""#));
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
    fn key_failures_trigger_bans_and_success_clears_state() {
        let mut state = KeyRuntimeState::default();
        let now = Local::now();

        mark_key_fail_in_state("sk-one", &mut state, now);
        mark_key_fail_in_state("sk-one", &mut state, now);
        mark_key_fail_in_state("sk-one", &mut state, now);
        let entry = state.keys.get("sk-one").unwrap();
        assert!(entry.ban_until.unwrap() > now);

        mark_key_success_in_state("sk-one", &mut state, now);
        assert!(!state.keys.contains_key("sk-one"));
    }

    #[test]
    fn available_api_keys_skip_banned_keys() {
        let mut state = KeyRuntimeState::default();
        let now = Local::now();
        mark_key_fail_in_state("sk-one", &mut state, now);
        mark_key_fail_in_state("sk-one", &mut state, now);
        mark_key_fail_in_state("sk-one", &mut state, now);

        let keys = available_api_keys_in_state(
            &["sk-one".to_string(), "sk-two".to_string()],
            &mut state,
            now,
        );
        assert_eq!(keys, vec!["sk-two".to_string()]);
    }

    #[test]
    fn init_database_migrates_legacy_single_key() {
        let path = std::env::temp_dir().join(format!(
            "me-api-proxy-openai-test-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "
                CREATE TABLE openai_config (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    base_url TEXT NOT NULL DEFAULT '',
                    api_key TEXT NOT NULL DEFAULT '',
                    protocol_mode TEXT NOT NULL DEFAULT 'both',
                    enabled INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL DEFAULT 0
                );
                INSERT INTO openai_config (id, base_url, api_key, protocol_mode, enabled)
                VALUES (1, 'https://api.example.com', 'sk-legacy', 'both', 1);
                ",
            )
            .unwrap();
        }

        init_database(&path).unwrap();
        let config = load_config(&path).unwrap();

        assert_eq!(config.api_keys, vec!["sk-legacy".to_string()]);
        let _ = fs::remove_file(path);
    }
}
