use axum::{
    body::{to_bytes, Body, Bytes},
    extract::{Json, Path, Request, State},
    http::{header, HeaderMap, HeaderValue, Method, Response, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse},
    routing::{any, get, post, put},
    Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use chrono::{DateTime, Duration as ChronoDuration, Local, TimeZone};
use futures_util::{stream::BoxStream, StreamExt};
use rand::RngCore;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;
use tracing::{error, info};
use url::Url;

const ADMIN_HTML: &str = include_str!("admin.html");
const ACCESS_KEYS_HTML: &str = include_str!("access_keys.html");
const MAX_REQUEST_BODY_BYTES: usize = 20 * 1024 * 1024;
const CHAT_PATH: &str = "/v1/chat/completions";
const RESPONSES_PATH: &str = "/v1/responses";
const MODELS_PATH: &str = "/v1/models";
const IMAGES_GENERATIONS_PATH: &str = "/v1/images/generations";
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
    pools: Arc<RwLock<Vec<PoolConfig>>>,
    access_keys: Arc<RwLock<Vec<AccessKeyConfig>>>,
    runtime: Arc<RwLock<KeyRuntimeState>>,
    usage: Arc<RwLock<UsageRuntimeState>>,
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TestProtocol {
    Chat,
    Responses,
}

impl TestProtocol {
    fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Responses => "responses",
        }
    }

    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "chat" => Ok(Self::Chat),
            "responses" => Ok(Self::Responses),
            _ => Err("protocol must be chat or responses".to_string()),
        }
    }

    fn from_saved(value: &str) -> Option<Self> {
        Self::from_str(value).ok()
    }

    fn endpoint(self) -> OpenAiEndpoint {
        match self {
            Self::Chat => OpenAiEndpoint::Chat,
            Self::Responses => OpenAiEndpoint::Responses,
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

#[derive(Clone, Debug, PartialEq, Eq)]
enum ResponsesToolKind {
    Function,
    Namespace,
    Custom,
    ToolSearch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResponsesToolSpec {
    kind: ResponsesToolKind,
    name: String,
    namespace: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct ResponsesToolContext {
    chat_tools: Vec<Value>,
    chat_name_to_spec: HashMap<String, ResponsesToolSpec>,
    namespace_name_to_chat_name: HashMap<(String, String), String>,
    seen_chat_names: HashSet<String>,
}

#[derive(Clone, Debug, Default)]
struct BridgeContext {
    responses_tool_context: Option<ResponsesToolContext>,
}

#[derive(Clone, Debug)]
struct RequestTransformResult {
    body: Vec<u8>,
    bridge_context: BridgeContext,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PoolConfig {
    id: i64,
    name: String,
    note: String,
    is_active: bool,
    base_urls: Vec<BaseUrlConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct BaseUrlConfig {
    id: i64,
    pool_id: i64,
    name: String,
    base_url: String,
    protocol_mode: ProtocolMode,
    override_model: String,
    sort_order: i64,
    api_keys: Vec<ApiKeyConfig>,
    image_keys: Vec<ApiKeyConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct ApiKeyConfig {
    id: i64,
    base_url_id: i64,
    api_key: String,
    sort_order: i64,
    manually_disabled: bool,
    test_model: String,
    test_protocol: Option<TestProtocol>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct AccessKeyConfig {
    id: i64,
    name: String,
    access_key: String,
    proxy_id: i64,
    image_proxy_id: Option<i64>,
    override_model: String,
    created_at: i64,
    updated_at: i64,
}

#[derive(Clone, Debug, Default)]
struct ApiKeyRuntime {
    fail_count: u32,
    ban_until: Option<DateTime<Local>>,
}

#[derive(Clone, Debug)]
struct KeyRuntimeState {
    last_reset_date: chrono::NaiveDate,
    keys: HashMap<i64, ApiKeyRuntime>,
    image_keys: HashMap<i64, ApiKeyRuntime>,
}

impl Default for KeyRuntimeState {
    fn default() -> Self {
        Self {
            last_reset_date: Local::now().date_naive(),
            keys: HashMap::new(),
            image_keys: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AccessKeyUsage {
    access_key_id: i64,
    proxy_id: i64,
    supplier_id: i64,
    key_id: i64,
    used_at: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct KeyUsage {
    access_key_id: i64,
    supplier_id: i64,
    used_at: i64,
}

#[derive(Clone, Debug, Default)]
struct UsageRuntimeState {
    by_access_key: HashMap<i64, AccessKeyUsage>,
    by_key: HashMap<i64, KeyUsage>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Serialize)]
struct OkResponse {
    ok: bool,
}

#[derive(Debug, Serialize)]
struct IdResponse {
    ok: bool,
    id: i64,
}

#[derive(Debug, Serialize)]
struct KeyTestResponse {
    ok: bool,
    message: String,
    status_code: Option<u16>,
}

#[derive(Debug, Serialize)]
struct KeyModelsResponse {
    models: Vec<String>,
    selected_model: Option<String>,
    selected_protocol: Option<TestProtocol>,
    supplier_protocol: ProtocolMode,
}

#[derive(Debug, Serialize)]
struct SupplierModelsResponse {
    models: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct KeyTestRequest {
    model: String,
    protocol: Option<TestProtocol>,
}

#[derive(Debug, Clone)]
struct ProxyAttemptError {
    body: String,
}

#[derive(Debug, Deserialize)]
struct PoolPayload {
    name: String,
    note: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BaseUrlPayload {
    #[serde(default)]
    name: String,
    base_url: String,
    protocol: ProtocolMode,
    #[serde(default)]
    override_model: String,
    sort_order: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ApiKeyPayload {
    api_key: String,
    sort_order: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ProxySavePayload {
    name: String,
    #[serde(default)]
    note: String,
    #[serde(default)]
    suppliers: Vec<SupplierSavePayload>,
}

#[derive(Debug, Deserialize)]
struct SupplierSavePayload {
    id: Option<i64>,
    #[serde(default)]
    name: String,
    base_url: String,
    protocol: ProtocolMode,
    #[serde(default)]
    override_model: String,
    #[serde(default)]
    keys: Vec<ApiKeySavePayload>,
    #[serde(default)]
    image_keys: Vec<ApiKeySavePayload>,
}

#[derive(Debug, Deserialize)]
struct ApiKeySavePayload {
    id: Option<i64>,
    api_key: String,
}

#[derive(Debug)]
struct PreparedProxySave {
    name: String,
    note: String,
    suppliers: Vec<PreparedSupplierSave>,
}

#[derive(Debug)]
struct PreparedSupplierSave {
    id: Option<i64>,
    name: String,
    base_url: String,
    protocol: ProtocolMode,
    override_model: String,
    keys: Vec<PreparedApiKeySave>,
    image_keys: Vec<PreparedApiKeySave>,
}

#[derive(Debug)]
struct PreparedApiKeySave {
    id: Option<i64>,
    api_key: String,
}

#[derive(Debug, Deserialize)]
struct AccessKeyPayload {
    name: String,
    proxy_id: i64,
    #[serde(default)]
    image_proxy_id: Option<i64>,
    #[serde(default)]
    override_model: String,
}

#[derive(Debug, Serialize)]
struct ProxiesResponse {
    active_proxy_id: Option<i64>,
    proxies: Vec<ProxyView>,
}

#[derive(Debug, Serialize)]
struct ProxyView {
    id: i64,
    name: String,
    note: String,
    is_active: bool,
    access_key_count: usize,
    in_use: bool,
    available_supplier_count: usize,
    total_supplier_count: usize,
    suppliers: Vec<SupplierView>,
}

#[derive(Debug, Serialize)]
struct SupplierView {
    id: i64,
    proxy_id: i64,
    name: String,
    base_url: String,
    protocol: ProtocolMode,
    override_model: String,
    sort_order: i64,
    available_key_count: usize,
    total_key_count: usize,
    schedulable: bool,
    last_used_at: Option<i64>,
    last_used_by_access_key_name: Option<String>,
    keys: Vec<ApiKeyView>,
    image_keys: Vec<ApiKeyView>,
}

#[derive(Debug, Serialize)]
struct ApiKeyView {
    id: i64,
    api_key: String,
    masked_key: String,
    sort_order: i64,
    manually_disabled: bool,
    test_model: Option<String>,
    test_protocol: Option<TestProtocol>,
    fail_count: u32,
    banned: bool,
    ban_until: Option<i64>,
    last_used: bool,
    last_used_at: Option<i64>,
    last_used_by_access_key_name: Option<String>,
}

#[derive(Debug, Serialize)]
struct AccessKeysResponse {
    access_keys: Vec<AccessKeyView>,
    proxies: Vec<ProxyOptionView>,
}

#[derive(Debug, Serialize)]
struct AccessKeyView {
    id: i64,
    name: String,
    access_key: String,
    masked_key: String,
    proxy_id: i64,
    proxy_name: String,
    image_proxy_id: Option<i64>,
    image_proxy_name: Option<String>,
    override_model: String,
    created_at: i64,
    updated_at: i64,
    last_used_supplier_name: Option<String>,
    last_used_key_masked: Option<String>,
    last_used_at: Option<i64>,
}

#[derive(Debug, Serialize)]
struct ProxyOptionView {
    id: i64,
    name: String,
    suppliers: Vec<SupplierOptionView>,
}

#[derive(Debug, Serialize)]
struct SupplierOptionView {
    id: i64,
    name: String,
    base_url: String,
}

#[derive(Clone, Copy, Debug)]
struct KeyAvailability {
    fail_count: u32,
    banned: bool,
    ban_until: Option<i64>,
    schedulable: bool,
}

struct SseTransformState {
    stream: BoxStream<'static, Result<Bytes, reqwest::Error>>,
    buffer: String,
    conversion: ConversionMode,
    bridge_context: BridgeContext,
    completed_sent: bool,
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

fn open_db(path: &PathBuf) -> Result<Connection, String> {
    let conn = Connection::open(path).map_err(|err| format!("Failed to open database: {err}"))?;
    conn.execute_batch("PRAGMA foreign_keys = ON;")
        .map_err(|err| format!("Failed to enable foreign keys: {err}"))?;
    Ok(conn)
}

fn table_exists(conn: &Connection, table: &str) -> Result<bool, String> {
    let exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1 LIMIT 1",
            params![table],
            |_| Ok(()),
        )
        .optional()
        .map_err(|err| format!("Failed to inspect table {table}: {err}"))?
        .is_some();
    Ok(exists)
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
        CREATE TABLE IF NOT EXISTS pools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            is_active INTEGER NOT NULL DEFAULT 0,
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
        );

        CREATE TABLE IF NOT EXISTS pool_base_urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pool_id INTEGER NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            base_url TEXT NOT NULL,
            protocol_mode TEXT NOT NULL DEFAULT 'both',
            override_model TEXT NOT NULL DEFAULT '',
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            FOREIGN KEY (pool_id) REFERENCES pools(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS pool_api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            base_url_id INTEGER NOT NULL,
            api_key TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 0,
            manually_disabled INTEGER NOT NULL DEFAULT 0,
            test_model TEXT NOT NULL DEFAULT '',
            test_protocol TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            FOREIGN KEY (base_url_id) REFERENCES pool_base_urls(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS pool_image_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            base_url_id INTEGER NOT NULL,
            api_key TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 0,
            manually_disabled INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            FOREIGN KEY (base_url_id) REFERENCES pool_base_urls(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS access_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            access_key TEXT NOT NULL UNIQUE,
            proxy_id INTEGER NOT NULL,
            image_proxy_id INTEGER,
            override_model TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            FOREIGN KEY (proxy_id) REFERENCES pools(id) ON DELETE CASCADE
        );
        ",
    )
    .map_err(|err| format!("Failed to initialize database: {err}"))?;

    let pool_columns = table_columns(&conn, "pools")?;
    if !pool_columns.contains("note") {
        conn.execute(
            "ALTER TABLE pools ADD COLUMN note TEXT NOT NULL DEFAULT ''",
            [],
        )
        .map_err(|err| format!("Failed to add proxy note column: {err}"))?;
    }
    if !pool_columns.contains("sort_order") {
        conn.execute(
            "ALTER TABLE pools ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0",
            [],
        )
        .map_err(|err| format!("Failed to add proxy sort_order column: {err}"))?;
        // 为现有数据设置默认排序
        conn.execute(
            "UPDATE pools SET sort_order = id - 1 WHERE sort_order = 0",
            [],
        )
        .map_err(|err| format!("Failed to initialize proxy sort_order: {err}"))?;
    }

    let base_url_columns = table_columns(&conn, "pool_base_urls")?;
    if !base_url_columns.contains("name") {
        conn.execute(
            "ALTER TABLE pool_base_urls ADD COLUMN name TEXT NOT NULL DEFAULT ''",
            [],
        )
        .map_err(|err| format!("Failed to add supplier name column: {err}"))?;
    }
    if !base_url_columns.contains("protocol_mode") {
        conn.execute(
            "ALTER TABLE pool_base_urls ADD COLUMN protocol_mode TEXT NOT NULL DEFAULT 'both'",
            [],
        )
        .map_err(|err| format!("Failed to add supplier protocol column: {err}"))?;
    }
    if !base_url_columns.contains("override_model") {
        conn.execute(
            "ALTER TABLE pool_base_urls ADD COLUMN override_model TEXT NOT NULL DEFAULT ''",
            [],
        )
        .map_err(|err| format!("Failed to add supplier override_model column: {err}"))?;
    }

    let api_key_columns = table_columns(&conn, "pool_api_keys")?;
    if !api_key_columns.contains("test_model") {
        conn.execute(
            "ALTER TABLE pool_api_keys ADD COLUMN test_model TEXT NOT NULL DEFAULT ''",
            [],
        )
        .map_err(|err| format!("Failed to add api_key test model column: {err}"))?;
    }
    if !api_key_columns.contains("test_protocol") {
        conn.execute(
            "ALTER TABLE pool_api_keys ADD COLUMN test_protocol TEXT NOT NULL DEFAULT ''",
            [],
        )
        .map_err(|err| format!("Failed to add api_key test protocol column: {err}"))?;
    }

    let access_key_columns = table_columns(&conn, "access_keys")?;
    if !access_key_columns.contains("override_model") {
        conn.execute(
            "ALTER TABLE access_keys ADD COLUMN override_model TEXT NOT NULL DEFAULT ''",
            [],
        )
        .map_err(|err| format!("Failed to add access key override_model column: {err}"))?;
    }
    if !access_key_columns.contains("image_proxy_id") {
        conn.execute(
            "ALTER TABLE access_keys ADD COLUMN image_proxy_id INTEGER",
            [],
        )
        .map_err(|err| format!("Failed to add access key image_proxy_id column: {err}"))?;
    }

    migrate_legacy_openai_config(&conn)?;
    Ok(())
}

fn normalize_pool_name(value: &str) -> Result<String, String> {
    let value = value.trim();
    if value.is_empty() {
        return Err("pool name must not be empty".to_string());
    }
    Ok(value.to_string())
}

fn normalize_note(value: &str) -> String {
    value.trim().to_string()
}

fn normalize_base_url(value: &str) -> Result<String, String> {
    let base_url = value.trim().trim_end_matches('/').to_string();
    if base_url.is_empty() {
        return Err("base_url must not be empty".to_string());
    }

    let parsed = Url::parse(&base_url).map_err(|err| format!("invalid base_url: {err}"))?;
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => return Err(format!("unsupported base_url scheme: {scheme}")),
    }

    Ok(base_url)
}

fn normalize_api_key(value: &str) -> Result<String, String> {
    let api_key = value.trim();
    if api_key.is_empty() {
        return Err("api_key must not be empty".to_string());
    }
    Ok(api_key.to_string())
}

fn normalize_access_key_name(value: &str) -> Result<String, String> {
    let value = value.trim();
    if value.is_empty() {
        return Err("access key name must not be empty".to_string());
    }
    Ok(value.to_string())
}

fn generate_access_key_value() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    let mut out = String::from("me-");
    for byte in bytes {
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn access_key_value_exists(conn: &Connection, access_key: &str) -> Result<bool, String> {
    conn.query_row(
        "SELECT 1 FROM access_keys WHERE access_key = ?1 LIMIT 1",
        params![access_key],
        |_| Ok(()),
    )
    .optional()
    .map_err(|err| format!("Failed to query access key value: {err}"))
    .map(|value| value.is_some())
}

fn generate_unique_access_key(conn: &Connection) -> Result<String, String> {
    for _ in 0..16 {
        let access_key = generate_access_key_value();
        if !access_key_value_exists(conn, &access_key)? {
            return Ok(access_key);
        }
    }
    Err("Failed to generate a unique access key".to_string())
}

fn prepare_proxy_save_payload(payload: ProxySavePayload) -> Result<PreparedProxySave, String> {
    let name = normalize_pool_name(&payload.name)?;
    let note = normalize_note(&payload.note);
    let mut seen_base_urls = HashSet::new();
    let mut suppliers = Vec::with_capacity(payload.suppliers.len());

    for supplier in payload.suppliers {
        let base_url = normalize_base_url(&supplier.base_url)?;
        if !seen_base_urls.insert(base_url.clone()) {
            return Err("base_url already exists in proxy".to_string());
        }

        let keys = prepare_api_key_saves(supplier.keys)?;
        let image_keys = prepare_api_key_saves(supplier.image_keys)?;

        suppliers.push(PreparedSupplierSave {
            id: supplier.id,
            name: supplier.name.trim().to_string(),
            base_url,
            protocol: supplier.protocol,
            override_model: supplier.override_model,
            keys,
            image_keys,
        });
    }

    Ok(PreparedProxySave {
        name,
        note,
        suppliers,
    })
}

fn prepare_api_key_saves(keys: Vec<ApiKeySavePayload>) -> Result<Vec<PreparedApiKeySave>, String> {
    let mut seen_api_keys = HashSet::new();
    let mut prepared_keys = Vec::with_capacity(keys.len());
    for key in keys {
        let api_key = normalize_api_key(&key.api_key)?;
        if !seen_api_keys.insert(api_key.clone()) {
            return Err("api_key already exists in supplier".to_string());
        }
        prepared_keys.push(PreparedApiKeySave {
            id: key.id,
            api_key,
        });
    }
    Ok(prepared_keys)
}

fn normalize_legacy_api_keys(api_keys: Vec<String>) -> Vec<String> {
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
    normalized
}

fn migrate_legacy_openai_config(conn: &Connection) -> Result<(), String> {
    let pool_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM pools", [], |row| row.get(0))
        .map_err(|err| format!("Failed to query pool count: {err}"))?;
    if pool_count > 0 {
        return Ok(());
    }

    if !table_exists(conn, "openai_config")? {
        return Ok(());
    }

    let columns = table_columns(conn, "openai_config")?;
    let has_api_key = columns.contains("api_key");

    let legacy = if has_api_key {
        conn.query_row(
            "SELECT base_url, protocol_mode, enabled, api_key FROM openai_config WHERE id = 1",
            [],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)? != 0,
                    row.get::<_, String>(3).ok(),
                ))
            },
        )
        .optional()
        .map_err(|err| format!("Failed to load legacy openai_config: {err}"))?
    } else {
        conn.query_row(
            "SELECT base_url, protocol_mode, enabled FROM openai_config WHERE id = 1",
            [],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)? != 0,
                    None,
                ))
            },
        )
        .optional()
        .map_err(|err| format!("Failed to load legacy openai_config: {err}"))?
    };

    let Some((base_url_raw, protocol_mode_raw, enabled, legacy_api_key)) = legacy else {
        return Ok(());
    };

    let base_url = match normalize_base_url(&base_url_raw) {
        Ok(base_url) => base_url,
        Err(_) => return Ok(()),
    };

    let protocol_mode = ProtocolMode::from_str(&protocol_mode_raw).unwrap_or(ProtocolMode::Both);

    let mut api_keys = Vec::new();
    if table_exists(conn, "openai_api_keys")? {
        let mut stmt = conn
            .prepare("SELECT api_key FROM openai_api_keys ORDER BY sort_order ASC, id ASC")
            .map_err(|err| format!("Failed to prepare legacy api key query: {err}"))?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|err| format!("Failed to query legacy api keys: {err}"))?;
        for row in rows {
            api_keys
                .push(row.map_err(|err| format!("Failed to decode legacy api key row: {err}"))?);
        }
    }
    if api_keys.is_empty() {
        if let Some(api_key) = legacy_api_key {
            api_keys.push(api_key);
        }
    }
    let api_keys = normalize_legacy_api_keys(api_keys);

    conn.execute(
        "INSERT INTO pools (name, is_active, created_at, updated_at)
         VALUES (?1, 1, strftime('%s', 'now'), strftime('%s', 'now'))",
        params!["默认代理"],
    )
    .map_err(|err| format!("Failed to migrate legacy pool: {err}"))?;
    let pool_id = conn.last_insert_rowid();

    conn.execute(
        "INSERT INTO pool_base_urls (pool_id, name, base_url, protocol_mode, sort_order, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
        params![pool_id, "", base_url, protocol_mode.as_str()],
    )
    .map_err(|err| format!("Failed to migrate legacy base_url: {err}"))?;
    let base_url_id = conn.last_insert_rowid();

    for (index, api_key) in api_keys.iter().enumerate() {
        conn.execute(
            "INSERT INTO pool_api_keys
             (base_url_id, api_key, sort_order, manually_disabled, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, strftime('%s', 'now'), strftime('%s', 'now'))",
            params![
                base_url_id,
                api_key,
                index as i64,
                if enabled { 0 } else { 1 }
            ],
        )
        .map_err(|err| format!("Failed to migrate legacy api_key: {err}"))?;
    }

    Ok(())
}

fn pool_name_exists(
    conn: &Connection,
    name: &str,
    exclude_id: Option<i64>,
) -> Result<bool, String> {
    let exists = if let Some(id) = exclude_id {
        conn.query_row(
            "SELECT 1 FROM pools WHERE name = ?1 AND id != ?2 LIMIT 1",
            params![name, id],
            |_| Ok(()),
        )
        .optional()
        .map_err(|err| format!("Failed to query pool name: {err}"))?
        .is_some()
    } else {
        conn.query_row(
            "SELECT 1 FROM pools WHERE name = ?1 LIMIT 1",
            params![name],
            |_| Ok(()),
        )
        .optional()
        .map_err(|err| format!("Failed to query pool name: {err}"))?
        .is_some()
    };
    Ok(exists)
}

fn duplicate_base_url_exists(
    conn: &Connection,
    pool_id: i64,
    base_url: &str,
    exclude_id: Option<i64>,
) -> Result<bool, String> {
    let exists = if let Some(id) = exclude_id {
        conn.query_row(
            "SELECT 1 FROM pool_base_urls WHERE pool_id = ?1 AND base_url = ?2 AND id != ?3 LIMIT 1",
            params![pool_id, base_url, id],
            |_| Ok(()),
        )
        .optional()
        .map_err(|err| format!("Failed to query duplicate base_url: {err}"))?
        .is_some()
    } else {
        conn.query_row(
            "SELECT 1 FROM pool_base_urls WHERE pool_id = ?1 AND base_url = ?2 LIMIT 1",
            params![pool_id, base_url],
            |_| Ok(()),
        )
        .optional()
        .map_err(|err| format!("Failed to query duplicate base_url: {err}"))?
        .is_some()
    };
    Ok(exists)
}

fn duplicate_api_key_exists(
    conn: &Connection,
    base_url_id: i64,
    api_key: &str,
    exclude_id: Option<i64>,
) -> Result<bool, String> {
    let exists = if let Some(id) = exclude_id {
        conn.query_row(
            "SELECT 1 FROM pool_api_keys WHERE base_url_id = ?1 AND api_key = ?2 AND id != ?3 LIMIT 1",
            params![base_url_id, api_key, id],
            |_| Ok(()),
        )
        .optional()
        .map_err(|err| format!("Failed to query duplicate api_key: {err}"))?
        .is_some()
    } else {
        conn.query_row(
            "SELECT 1 FROM pool_api_keys WHERE base_url_id = ?1 AND api_key = ?2 LIMIT 1",
            params![base_url_id, api_key],
            |_| Ok(()),
        )
        .optional()
        .map_err(|err| format!("Failed to query duplicate api_key: {err}"))?
        .is_some()
    };
    Ok(exists)
}

fn next_sort_order(
    conn: &Connection,
    table: &str,
    column: &str,
    owner_id: i64,
) -> Result<i64, String> {
    let sql = format!("SELECT COALESCE(MAX(sort_order), -1) + 1 FROM {table} WHERE {column} = ?1");
    conn.query_row(&sql, params![owner_id], |row| row.get(0))
        .map_err(|err| format!("Failed to compute next sort_order for {table}: {err}"))
}

fn normalize_base_url_orders(conn: &Connection, pool_id: i64) -> Result<(), String> {
    let mut stmt = conn
        .prepare("SELECT id FROM pool_base_urls WHERE pool_id = ?1 ORDER BY sort_order ASC, id ASC")
        .map_err(|err| format!("Failed to prepare base_url order query: {err}"))?;
    let ids = stmt
        .query_map(params![pool_id], |row| row.get::<_, i64>(0))
        .map_err(|err| format!("Failed to query base_url order: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to decode base_url order: {err}"))?;

    for (index, id) in ids.into_iter().enumerate() {
        conn.execute(
            "UPDATE pool_base_urls
             SET sort_order = ?1, updated_at = strftime('%s', 'now')
             WHERE id = ?2",
            params![index as i64, id],
        )
        .map_err(|err| format!("Failed to normalize base_url order: {err}"))?;
    }

    Ok(())
}

fn normalize_pool_orders(conn: &Connection) -> Result<(), String> {
    let mut stmt = conn
        .prepare("SELECT id FROM pools ORDER BY sort_order ASC, id ASC")
        .map_err(|err| format!("Failed to prepare pool order query: {err}"))?;
    let ids = stmt
        .query_map([], |row| row.get::<_, i64>(0))
        .map_err(|err| format!("Failed to query pool order: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to decode pool order: {err}"))?;

    for (index, id) in ids.into_iter().enumerate() {
        conn.execute(
            "UPDATE pools
             SET sort_order = ?1, updated_at = strftime('%s', 'now')
             WHERE id = ?2",
            params![index as i64, id],
        )
        .map_err(|err| format!("Failed to normalize pool order: {err}"))?;
    }

    Ok(())
}

fn normalize_api_key_orders(conn: &Connection, base_url_id: i64) -> Result<(), String> {
    let mut stmt = conn
        .prepare(
            "SELECT id FROM pool_api_keys WHERE base_url_id = ?1 ORDER BY sort_order ASC, id ASC",
        )
        .map_err(|err| format!("Failed to prepare api_key order query: {err}"))?;
    let ids = stmt
        .query_map(params![base_url_id], |row| row.get::<_, i64>(0))
        .map_err(|err| format!("Failed to query api_key order: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to decode api_key order: {err}"))?;

    for (index, id) in ids.into_iter().enumerate() {
        conn.execute(
            "UPDATE pool_api_keys
             SET sort_order = ?1, updated_at = strftime('%s', 'now')
             WHERE id = ?2",
            params![index as i64, id],
        )
        .map_err(|err| format!("Failed to normalize api_key order: {err}"))?;
    }

    Ok(())
}

fn ensure_single_active_pool(conn: &Connection, pool_id: i64) -> Result<(), String> {
    let updated = conn
        .execute(
            "UPDATE pools
             SET is_active = CASE WHEN id = ?1 THEN 1 ELSE 0 END,
                 updated_at = strftime('%s', 'now')",
            params![pool_id],
        )
        .map_err(|err| format!("Failed to activate pool: {err}"))?;
    if updated == 0 {
        return Err("pool not found".to_string());
    }
    Ok(())
}

fn proxy_exists(conn: &Connection, proxy_id: i64) -> Result<bool, String> {
    conn.query_row(
        "SELECT 1 FROM pools WHERE id = ?1 LIMIT 1",
        params![proxy_id],
        |_| Ok(()),
    )
    .optional()
    .map_err(|err| format!("Failed to load proxy: {err}"))
    .map(|value| value.is_some())
}

fn load_pools(path: &PathBuf) -> Result<Vec<PoolConfig>, String> {
    let conn = open_db(path)?;
    let mut pool_stmt = conn
        .prepare(
            "SELECT id, name, note, is_active
             FROM pools
             ORDER BY sort_order ASC, is_active DESC, created_at ASC, id ASC",
        )
        .map_err(|err| format!("Failed to prepare pool query: {err}"))?;

    let pool_rows = pool_stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)? != 0,
            ))
        })
        .map_err(|err| format!("Failed to query pools: {err}"))?;

    let mut pools = Vec::new();
    for row in pool_rows {
        let (pool_id, name, note, is_active) =
            row.map_err(|err| format!("Failed to decode pool row: {err}"))?;

        let mut base_stmt = conn
            .prepare(
                "SELECT id, name, base_url, protocol_mode, override_model, sort_order
                 FROM pool_base_urls
                 WHERE pool_id = ?1
                 ORDER BY sort_order ASC, id ASC",
            )
            .map_err(|err| format!("Failed to prepare base_url query: {err}"))?;
        let base_rows = base_stmt
            .query_map(params![pool_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            })
            .map_err(|err| format!("Failed to query base_urls: {err}"))?;

        let mut base_urls = Vec::new();
        for base_row in base_rows {
            let (base_url_id, supplier_name, base_url, protocol_mode, override_model, sort_order) =
                base_row.map_err(|err| format!("Failed to decode base_url row: {err}"))?;

            let mut key_stmt = conn
                .prepare(
                    "SELECT id, api_key, sort_order, manually_disabled
                     , test_model, test_protocol
                     FROM pool_api_keys
                     WHERE base_url_id = ?1
                     ORDER BY sort_order ASC, id ASC",
                )
                .map_err(|err| format!("Failed to prepare api_key query: {err}"))?;
            let key_rows = key_stmt
                .query_map(params![base_url_id], |row| {
                    let test_protocol_raw: String = row.get(5)?;
                    Ok(ApiKeyConfig {
                        id: row.get(0)?,
                        base_url_id,
                        api_key: row.get(1)?,
                        sort_order: row.get(2)?,
                        manually_disabled: row.get::<_, i64>(3)? != 0,
                        test_model: row.get(4)?,
                        test_protocol: TestProtocol::from_saved(&test_protocol_raw),
                    })
                })
                .map_err(|err| format!("Failed to query api_keys: {err}"))?;

            let mut api_keys = Vec::new();
            for key_row in key_rows {
                api_keys
                    .push(key_row.map_err(|err| format!("Failed to decode api_key row: {err}"))?);
            }

            let mut image_key_stmt = conn
                .prepare(
                    "SELECT id, api_key, sort_order, manually_disabled
                     FROM pool_image_keys
                     WHERE base_url_id = ?1
                     ORDER BY sort_order ASC, id ASC",
                )
                .map_err(|err| format!("Failed to prepare image key query: {err}"))?;
            let image_key_rows = image_key_stmt
                .query_map(params![base_url_id], |row| {
                    Ok(ApiKeyConfig {
                        id: row.get(0)?,
                        base_url_id,
                        api_key: row.get(1)?,
                        sort_order: row.get(2)?,
                        manually_disabled: row.get::<_, i64>(3)? != 0,
                        test_model: String::new(),
                        test_protocol: None,
                    })
                })
                .map_err(|err| format!("Failed to query image keys: {err}"))?;
            let mut image_keys = Vec::new();
            for key_row in image_key_rows {
                image_keys
                    .push(key_row.map_err(|err| format!("Failed to decode image key row: {err}"))?);
            }

            base_urls.push(BaseUrlConfig {
                id: base_url_id,
                pool_id,
                name: supplier_name,
                base_url,
                protocol_mode: ProtocolMode::from_str(&protocol_mode)?,
                override_model,
                sort_order,
                api_keys,
                image_keys,
            });
        }

        pools.push(PoolConfig {
            id: pool_id,
            name,
            note,
            is_active,
            base_urls,
        });
    }

    Ok(pools)
}

fn load_access_keys(path: &PathBuf) -> Result<Vec<AccessKeyConfig>, String> {
    let conn = open_db(path)?;
    let mut stmt = conn
        .prepare(
            "SELECT id, name, access_key, proxy_id, image_proxy_id, override_model, created_at, updated_at
             FROM access_keys
             ORDER BY created_at DESC, id DESC",
        )
        .map_err(|err| format!("Failed to prepare access key query: {err}"))?;
    let rows = stmt
        .query_map([], |row| {
            Ok(AccessKeyConfig {
                id: row.get(0)?,
                name: row.get(1)?,
                access_key: row.get(2)?,
                proxy_id: row.get(3)?,
                image_proxy_id: row.get(4)?,
                override_model: row.get(5)?,
                created_at: row.get(6)?,
                updated_at: row.get(7)?,
            })
        })
        .map_err(|err| format!("Failed to query access keys: {err}"))?;

    let mut access_keys = Vec::new();
    for row in rows {
        access_keys.push(row.map_err(|err| format!("Failed to decode access key row: {err}"))?);
    }
    Ok(access_keys)
}

async fn reload_pools(state: &AppState) -> Result<(), String> {
    let pools = load_pools(&state.db_path)?;
    let valid_key_ids: HashSet<i64> = pools
        .iter()
        .flat_map(|pool| pool.base_urls.iter())
        .flat_map(|base_url| base_url.api_keys.iter().map(|key| key.id))
        .collect();
    let valid_image_key_ids: HashSet<i64> = pools
        .iter()
        .flat_map(|pool| pool.base_urls.iter())
        .flat_map(|base_url| base_url.image_keys.iter().map(|key| key.id))
        .collect();

    {
        let mut runtime = state.runtime.write().await;
        runtime
            .keys
            .retain(|key_id, _| valid_key_ids.contains(key_id));
        runtime
            .image_keys
            .retain(|key_id, _| valid_image_key_ids.contains(key_id));
    }
    *state.pools.write().await = pools;
    reconcile_usage_runtime(state).await;
    Ok(())
}

async fn reload_access_keys(state: &AppState) -> Result<(), String> {
    let access_keys = load_access_keys(&state.db_path)?;
    *state.access_keys.write().await = access_keys;
    reconcile_usage_runtime(state).await;
    Ok(())
}

async fn reload_state(state: &AppState) -> Result<(), String> {
    reload_pools(state).await?;
    reload_access_keys(state).await?;
    Ok(())
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

fn json_ok() -> Response<Body> {
    Json(OkResponse { ok: true }).into_response()
}

fn json_id_ok(id: i64) -> Response<Body> {
    Json(IdResponse { ok: true, id }).into_response()
}

fn json_key_test_ok(
    message: impl Into<String>,
    status_code: Option<u16>,
    ok: bool,
) -> Response<Body> {
    Json(KeyTestResponse {
        ok,
        message: message.into(),
        status_code,
    })
    .into_response()
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

#[derive(Clone, Debug)]
struct KeyTestTarget {
    api_key: String,
    base_url: String,
    supplier_protocol: ProtocolMode,
    selected_model: Option<String>,
    selected_protocol: Option<TestProtocol>,
}

fn normalize_test_model(value: &str) -> Result<String, String> {
    let model = value.trim();
    if model.is_empty() {
        return Err("model is required".to_string());
    }
    Ok(model.to_string())
}

fn resolve_test_protocol(
    supplier_protocol: &ProtocolMode,
    requested: Option<TestProtocol>,
) -> Result<TestProtocol, String> {
    match supplier_protocol {
        ProtocolMode::Chat => match requested {
            Some(TestProtocol::Responses) => Err("supplier protocol only allows chat".to_string()),
            _ => Ok(TestProtocol::Chat),
        },
        ProtocolMode::Responses => match requested {
            Some(TestProtocol::Chat) => Err("supplier protocol only allows responses".to_string()),
            _ => Ok(TestProtocol::Responses),
        },
        ProtocolMode::Both => requested.ok_or_else(|| "protocol is required".to_string()),
    }
}

fn key_test_target(conn: &Connection, id: i64) -> Result<Option<KeyTestTarget>, String> {
    conn.query_row(
        "SELECT k.api_key, b.base_url, b.protocol_mode, k.test_model, k.test_protocol
         FROM pool_api_keys k
         JOIN pool_base_urls b ON b.id = k.base_url_id
         WHERE k.id = ?1",
        params![id],
        |row| {
            let supplier_protocol_raw: String = row.get(2)?;
            let test_model: String = row.get(3)?;
            let test_protocol_raw: String = row.get(4)?;
            let supplier_protocol =
                ProtocolMode::from_str(&supplier_protocol_raw).map_err(|err| {
                    rusqlite::Error::FromSqlConversionFailure(
                        2,
                        rusqlite::types::Type::Text,
                        Box::new(io::Error::new(io::ErrorKind::InvalidData, err)),
                    )
                })?;
            Ok(KeyTestTarget {
                api_key: row.get(0)?,
                base_url: row.get(1)?,
                supplier_protocol,
                selected_model: if test_model.trim().is_empty() {
                    None
                } else {
                    Some(test_model)
                },
                selected_protocol: TestProtocol::from_saved(&test_protocol_raw),
            })
        },
    )
    .optional()
    .map_err(|err| format!("Failed to load api_key: {err}"))
}

fn should_skip_header(name: &str, skipped: &[&str]) -> bool {
    let lower = name.to_ascii_lowercase();
    skipped.contains(&lower.as_str())
}

fn upstream_authority(url: &Url) -> Result<String, String> {
    let host = url
        .host_str()
        .ok_or_else(|| "upstream url is missing host".to_string())?;
    Ok(match url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_string(),
    })
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
    let raw_path = parsed.path().trim_end_matches('/');
    let base_path = raw_path
        .strip_suffix("/v1")
        .unwrap_or(raw_path)
        .trim_end_matches('/');
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

fn flatten_namespace_tool_name(namespace: &str, name: &str) -> String {
    format!("{namespace}.{name}")
}

fn unique_chat_tool_name(context: &mut ResponsesToolContext, base: String) -> String {
    if context.seen_chat_names.insert(base.clone()) {
        return base;
    }

    let mut index = 2;
    loop {
        let candidate = format!("{base}_{index}");
        if context.seen_chat_names.insert(candidate.clone()) {
            return candidate;
        }
        index += 1;
    }
}

fn string_input_parameters_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "input": { "type": "string" }
        },
        "required": ["input"],
        "additionalProperties": false
    })
}

fn tool_search_parameters_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": { "type": "string" },
            "limit": { "type": "integer" }
        },
        "required": ["query"],
        "additionalProperties": false
    })
}

fn build_responses_tool_context(tools: Option<&Value>) -> ResponsesToolContext {
    let mut context = ResponsesToolContext::default();
    let Some(tools) = tools.and_then(Value::as_array) else {
        return context;
    };

    for tool in tools {
        let Some(object) = tool.as_object() else {
            continue;
        };
        match object.get("type").and_then(Value::as_str) {
            Some("function") => {
                let Some(name) = object.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let chat_name = unique_chat_tool_name(&mut context, name.to_string());
                context.chat_name_to_spec.insert(
                    chat_name.clone(),
                    ResponsesToolSpec {
                        kind: ResponsesToolKind::Function,
                        name: name.to_string(),
                        namespace: None,
                    },
                );
                let mut chat_tool = object.clone();
                chat_tool.insert("type".to_string(), Value::String("function".to_string()));
                chat_tool.insert("name".to_string(), Value::String(chat_name));
                context.chat_tools.push(Value::Object(chat_tool));
            }
            Some("custom") => {
                let Some(name) = object.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let chat_name = unique_chat_tool_name(&mut context, name.to_string());
                context.chat_name_to_spec.insert(
                    chat_name.clone(),
                    ResponsesToolSpec {
                        kind: ResponsesToolKind::Custom,
                        name: name.to_string(),
                        namespace: None,
                    },
                );
                context.chat_tools.push(json!({
                    "type": "function",
                    "function": {
                        "name": chat_name,
                        "description": object.get("description").cloned().unwrap_or(Value::String(String::new())),
                        "parameters": string_input_parameters_schema()
                    }
                }));
            }
            Some("tool_search") => {
                let chat_name = unique_chat_tool_name(&mut context, "tool_search".to_string());
                context.chat_name_to_spec.insert(
                    chat_name.clone(),
                    ResponsesToolSpec {
                        kind: ResponsesToolKind::ToolSearch,
                        name: "tool_search".to_string(),
                        namespace: None,
                    },
                );
                context.chat_tools.push(json!({
                    "type": "function",
                    "function": {
                        "name": chat_name,
                        "description": object.get("description").cloned().unwrap_or(Value::String("Search across available tools".to_string())),
                        "parameters": tool_search_parameters_schema()
                    }
                }));
            }
            Some("namespace") => {
                let Some(namespace) = object.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let Some(functions) = object.get("tools").and_then(Value::as_array) else {
                    continue;
                };
                for function in functions {
                    let Some(function_object) = function.as_object() else {
                        continue;
                    };
                    let Some(name) = function_object.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    let flattened = flatten_namespace_tool_name(namespace, name);
                    let chat_name = unique_chat_tool_name(&mut context, flattened);
                    context
                        .namespace_name_to_chat_name
                        .insert((namespace.to_string(), name.to_string()), chat_name.clone());
                    context.chat_name_to_spec.insert(
                        chat_name.clone(),
                        ResponsesToolSpec {
                            kind: ResponsesToolKind::Namespace,
                            name: name.to_string(),
                            namespace: Some(namespace.to_string()),
                        },
                    );
                    context.chat_tools.push(json!({
                        "type": "function",
                        "function": {
                            "name": chat_name,
                            "description": function_object.get("description").cloned().unwrap_or(Value::String(String::new())),
                            "parameters": function_object.get("parameters").cloned().unwrap_or_else(|| json!({
                                "type": "object",
                                "properties": {},
                                "additionalProperties": true
                            }))
                        }
                    }));
                }
            }
            _ => {}
        }
    }

    context
}

fn chat_tool_name_for_response_tool(
    context: &ResponsesToolContext,
    name: &str,
    namespace: Option<&str>,
) -> String {
    if let Some(namespace) = namespace {
        if let Some(chat_name) = context
            .namespace_name_to_chat_name
            .get(&(namespace.to_string(), name.to_string()))
        {
            return chat_name.clone();
        }
        return flatten_namespace_tool_name(namespace, name);
    }

    context
        .chat_name_to_spec
        .iter()
        .find_map(|(chat_name, spec)| {
            (spec.name == name && spec.namespace.is_none()).then(|| chat_name.clone())
        })
        .unwrap_or_else(|| name.to_string())
}

fn tool_name_from_choice(choice: &Map<String, Value>) -> Option<String> {
    choice
        .get("name")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| {
            choice
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
}

fn responses_tool_choice_to_chat(choice: &Value, context: &ResponsesToolContext) -> Value {
    let Some(object) = choice.as_object() else {
        return choice.clone();
    };
    let Some(kind) = object.get("type").and_then(Value::as_str) else {
        return choice.clone();
    };

    match kind {
        "function" => tool_name_from_choice(object)
            .map(|name| {
                let chat_name = chat_tool_name_for_response_tool(
                    context,
                    &name,
                    object.get("namespace").and_then(Value::as_str),
                );
                json!({ "type": "function", "function": { "name": chat_name } })
            })
            .unwrap_or_else(|| choice.clone()),
        "custom" => tool_name_from_choice(object)
            .map(|name| json!({ "type": "function", "function": { "name": name } }))
            .unwrap_or_else(|| choice.clone()),
        "tool_search" => {
            let name = tool_name_from_choice(object).unwrap_or_else(|| "tool_search".to_string());
            let chat_name = chat_tool_name_for_response_tool(context, &name, None);
            json!({ "type": "function", "function": { "name": chat_name } })
        }
        "namespace" => object
            .get("tool")
            .and_then(Value::as_str)
            .map(|name| {
                let namespace = object
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let chat_name = chat_tool_name_for_response_tool(context, name, Some(namespace));
                json!({ "type": "function", "function": { "name": chat_name } })
            })
            .unwrap_or_else(|| choice.clone()),
        _ => choice.clone(),
    }
}

fn value_to_json_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| String::new()),
    }
}

fn flush_pending_tool_calls(messages: &mut Vec<Value>, pending_tool_calls: &mut Vec<Value>) {
    if pending_tool_calls.is_empty() {
        return;
    }
    messages.push(json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": std::mem::take(pending_tool_calls)
    }));
}

fn responses_function_call_to_chat_tool_call(
    item: &Map<String, Value>,
    context: &ResponsesToolContext,
) -> Value {
    let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
    let namespace = item.get("namespace").and_then(Value::as_str);
    let chat_name = chat_tool_name_for_response_tool(context, name, namespace);
    let call_id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .cloned()
        .unwrap_or(Value::String("call_0".to_string()));
    let arguments = item
        .get("arguments")
        .map(value_to_json_string)
        .unwrap_or_else(|| "{}".to_string());
    json!({
        "id": call_id,
        "type": "function",
        "function": {
            "name": chat_name,
            "arguments": arguments
        }
    })
}

fn responses_custom_tool_call_to_chat_tool_call(
    item: &Map<String, Value>,
    context: &ResponsesToolContext,
) -> Value {
    let call_id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .cloned()
        .unwrap_or(Value::String("call_0".to_string()));
    let response_name = item.get("name").and_then(Value::as_str).unwrap_or_default();
    let name = Value::String(chat_tool_name_for_response_tool(
        context,
        response_name,
        None,
    ));
    let input = item
        .get("input")
        .cloned()
        .unwrap_or(Value::String(String::new()));
    json!({
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json!({ "input": input }).to_string()
        }
    })
}

fn responses_tool_search_call_to_chat_tool_call(
    item: &Map<String, Value>,
    context: &ResponsesToolContext,
) -> Value {
    let call_id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .cloned()
        .unwrap_or(Value::String("call_0".to_string()));
    let mut arguments = Map::new();
    if let Some(query) = item.get("query") {
        arguments.insert("query".to_string(), query.clone());
    }
    if let Some(limit) = item.get("limit") {
        arguments.insert("limit".to_string(), limit.clone());
    }
    if arguments.is_empty() {
        if let Some(value) = item.get("arguments").and_then(Value::as_object) {
            arguments.extend(value.clone());
        }
    }
    json!({
        "id": call_id,
        "type": "function",
        "function": {
            "name": chat_tool_name_for_response_tool(
                context,
                item.get("name").and_then(Value::as_str).unwrap_or("tool_search"),
                None
            ),
            "arguments": Value::Object(arguments).to_string()
        }
    })
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

fn responses_item_to_chat_message(
    item: &Value,
    context: &ResponsesToolContext,
    messages: &mut Vec<Value>,
    pending_tool_calls: &mut Vec<Value>,
) {
    let Some(object) = item.as_object() else {
        flush_pending_tool_calls(messages, pending_tool_calls);
        messages.push(json!({ "role": "user", "content": item }));
        return;
    };

    match object.get("type").and_then(Value::as_str) {
        Some("function_call") => {
            pending_tool_calls.push(responses_function_call_to_chat_tool_call(object, context));
        }
        Some("custom_tool_call") => {
            pending_tool_calls.push(responses_custom_tool_call_to_chat_tool_call(
                object, context,
            ));
        }
        Some("tool_search_call") => {
            pending_tool_calls.push(responses_tool_search_call_to_chat_tool_call(
                object, context,
            ));
        }
        Some("function_call_output")
        | Some("custom_tool_call_output")
        | Some("tool_search_output") => {
            flush_pending_tool_calls(messages, pending_tool_calls);
            let tool_call_id = object
                .get("call_id")
                .or_else(|| object.get("id"))
                .cloned()
                .unwrap_or(Value::String("call_0".to_string()));
            let content = object
                .get("output")
                .map(value_to_json_string)
                .unwrap_or_else(|| value_to_json_string(item));
            messages.push(json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content
            }));
        }
        Some("message") | None => {
            if let Some(role) = object.get("role").and_then(Value::as_str) {
                flush_pending_tool_calls(messages, pending_tool_calls);
                let chat_role = match role {
                    "developer" => "system",
                    other => other,
                };
                messages.push(json!({
                    "role": chat_role,
                    "content": object
                        .get("content")
                        .map(responses_content_to_chat_content)
                        .unwrap_or(Value::String(String::new()))
                }));
            } else {
                flush_pending_tool_calls(messages, pending_tool_calls);
                messages.push(json!({ "role": "user", "content": item }));
            }
        }
        _ => {
            if let Some(role) = object.get("role").and_then(Value::as_str) {
                flush_pending_tool_calls(messages, pending_tool_calls);
                messages.push(json!({
                    "role": role,
                    "content": object
                        .get("content")
                        .map(responses_content_to_chat_content)
                        .unwrap_or_else(|| item.clone())
                }));
            } else {
                flush_pending_tool_calls(messages, pending_tool_calls);
                messages.push(json!({ "role": "user", "content": item }));
            }
        }
    }
}

fn responses_input_to_chat_messages(input: &Value, context: &ResponsesToolContext) -> Vec<Value> {
    match input {
        Value::String(text) => vec![json!({ "role": "user", "content": text })],
        Value::Array(items) => {
            let mut messages = Vec::new();
            let mut pending_tool_calls = Vec::new();
            for item in items {
                responses_item_to_chat_message(
                    item,
                    context,
                    &mut messages,
                    &mut pending_tool_calls,
                );
            }
            flush_pending_tool_calls(&mut messages, &mut pending_tool_calls);
            messages
        }
        _ => vec![json!({ "role": "user", "content": input })],
    }
}

fn responses_request_to_chat(body: &[u8]) -> Result<RequestTransformResult, String> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|err| format!("invalid Responses request JSON: {err}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "Responses request must be a JSON object".to_string())?;

    let input = object
        .get("input")
        .ok_or_else(|| "Responses request is missing input".to_string())?;
    let tool_context = build_responses_tool_context(object.get("tools"));
    let mut messages = Vec::new();
    if let Some(instructions) = object.get("instructions") {
        messages.push(json!({ "role": "system", "content": instructions }));
    }
    messages.extend(responses_input_to_chat_messages(input, &tool_context));

    let mut target = Map::new();
    target.insert("messages".to_string(), Value::Array(messages));

    for field in [
        "model",
        "stream",
        "temperature",
        "top_p",
        "parallel_tool_calls",
        "metadata",
        "user",
        "stop",
    ] {
        copy_field(object, &mut target, field);
    }
    if !tool_context.chat_tools.is_empty() {
        target.insert(
            "tools".to_string(),
            Value::Array(tool_context.chat_tools.clone()),
        );
    }
    if let Some(tool_choice) = object.get("tool_choice") {
        target.insert(
            "tool_choice".to_string(),
            responses_tool_choice_to_chat(tool_choice, &tool_context),
        );
    }
    copy_renamed_field(object, &mut target, "max_output_tokens", "max_tokens");

    let body = serde_json::to_vec(&Value::Object(target))
        .map_err(|err| format!("failed to encode Chat Completions request: {err}"))?;
    Ok(RequestTransformResult {
        body,
        bridge_context: BridgeContext {
            responses_tool_context: (!tool_context.chat_name_to_spec.is_empty())
                .then_some(tool_context),
        },
    })
}

fn transform_request_body(
    body: &[u8],
    conversion: ConversionMode,
) -> Result<RequestTransformResult, String> {
    match conversion {
        ConversionMode::Direct => Ok(RequestTransformResult {
            body: body.to_vec(),
            bridge_context: BridgeContext::default(),
        }),
        ConversionMode::ChatToResponses => {
            chat_request_to_responses(body).map(|body| RequestTransformResult {
                body,
                bridge_context: BridgeContext::default(),
            })
        }
        ConversionMode::ResponsesToChat => responses_request_to_chat(body),
    }
}

fn override_model_in_body(body: &[u8], model: &str) -> Result<Vec<u8>, String> {
    let mut value: Value =
        serde_json::from_slice(body).map_err(|err| format!("invalid request JSON: {err}"))?;
    if let Some(obj) = value.as_object_mut() {
        obj.insert("model".to_string(), Value::String(model.to_string()));
    }
    serde_json::to_vec(&value).map_err(|err| format!("failed to encode request: {err}"))
}

fn effective_override_model<'a>(
    access_key: &'a AccessKeyConfig,
    base_url: &'a BaseUrlConfig,
) -> Option<&'a str> {
    let access_key_model = access_key.override_model.trim();
    if !access_key_model.is_empty() {
        return Some(access_key_model);
    }
    let supplier_model = base_url.override_model.trim();
    (!supplier_model.is_empty()).then_some(supplier_model)
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

fn parse_tool_arguments(arguments: Option<&Value>) -> Value {
    let Some(arguments) = arguments else {
        return json!({});
    };
    match arguments {
        Value::String(text) => {
            serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.clone()))
        }
        value => value.clone(),
    }
}

fn chat_tool_call_to_responses_item(
    tool_call: &Value,
    context: Option<&ResponsesToolContext>,
) -> Value {
    let call_id = tool_call
        .get("id")
        .cloned()
        .unwrap_or(Value::String("call_0".to_string()));
    let chat_name = tool_call
        .pointer("/function/name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let arguments = parse_tool_arguments(tool_call.pointer("/function/arguments"));
    let spec = context.and_then(|ctx| ctx.chat_name_to_spec.get(chat_name));

    match spec.map(|item| &item.kind) {
        Some(ResponsesToolKind::Custom) => json!({
            "type": "custom_tool_call",
            "id": call_id.clone(),
            "call_id": call_id,
            "name": spec.map(|item| item.name.clone()).unwrap_or_else(|| chat_name.to_string()),
            "input": arguments.get("input").cloned().unwrap_or(Value::String(String::new()))
        }),
        Some(ResponsesToolKind::ToolSearch) => {
            let mut value = json!({
                "type": "tool_search_call",
                "id": call_id.clone(),
                "call_id": call_id
            });
            if let Some(obj) = value.as_object_mut() {
                obj.insert(
                    "name".to_string(),
                    Value::String(
                        spec.map(|item| item.name.clone())
                            .unwrap_or_else(|| chat_name.to_string()),
                    ),
                );
                if let Some(query) = arguments.get("query") {
                    obj.insert("query".to_string(), query.clone());
                }
                if let Some(limit) = arguments.get("limit") {
                    obj.insert("limit".to_string(), limit.clone());
                }
            }
            value
        }
        Some(ResponsesToolKind::Namespace) | Some(ResponsesToolKind::Function) | None => {
            let mut value = json!({
                "type": "function_call",
                "id": call_id.clone(),
                "call_id": call_id,
                "name": spec.map(|item| item.name.clone()).unwrap_or_else(|| chat_name.to_string()),
                "arguments": tool_call.pointer("/function/arguments").cloned().unwrap_or(Value::String(String::new()))
            });
            if let Some(namespace) = spec.and_then(|item| item.namespace.clone()) {
                if let Some(obj) = value.as_object_mut() {
                    obj.insert("namespace".to_string(), Value::String(namespace));
                }
            }
            value
        }
    }
}

fn chat_response_to_responses(
    body: &[u8],
    context: Option<&ResponsesToolContext>,
) -> Result<Vec<u8>, String> {
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
            output.push(chat_tool_call_to_responses_item(tool_call, context));
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

fn transform_response_body(
    body: &[u8],
    conversion: ConversionMode,
    bridge_context: &BridgeContext,
) -> Result<Vec<u8>, String> {
    match conversion {
        ConversionMode::Direct => Ok(body.to_vec()),
        ConversionMode::ChatToResponses => responses_response_to_chat(body),
        ConversionMode::ResponsesToChat => {
            chat_response_to_responses(body, bridge_context.responses_tool_context.as_ref())
        }
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

fn is_ignored_sse_payload(data: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(data) else {
        return false;
    };
    value
        .get("object")
        .and_then(Value::as_str)
        .is_some_and(|object| object == "billing.summary")
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
    _bridge_context: &BridgeContext,
    completed_sent: &mut bool,
) -> Option<String> {
    let data = sse_data_payload(event)?;
    if is_ignored_sse_payload(&data) {
        return None;
    }
    match conversion {
        ConversionMode::Direct => Some(format!("{event}\n\n")),
        ConversionMode::ChatToResponses => responses_sse_to_chat_event(&data, completed_sent),
        ConversionMode::ResponsesToChat => chat_sse_to_responses_event(&data, completed_sent),
    }
}

fn transformed_sse_body(
    response: reqwest::Response,
    conversion: ConversionMode,
    bridge_context: BridgeContext,
) -> Body {
    let state = SseTransformState {
        stream: response.bytes_stream().boxed(),
        buffer: String::new(),
        conversion,
        bridge_context,
        completed_sent: false,
    };

    let stream = futures_util::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event) = pop_sse_event(&mut state.buffer) {
                if let Some(output) = transform_sse_event(
                    &event,
                    state.conversion,
                    &state.bridge_context,
                    &mut state.completed_sent,
                ) {
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
                        if let Some(output) = transform_sse_event(
                            &event,
                            state.conversion,
                            &state.bridge_context,
                            &mut state.completed_sent,
                        ) {
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
        state.image_keys.clear();
        state.last_reset_date = today;
    }
}

fn clear_expired_ban(entry: &mut ApiKeyRuntime, now: DateTime<Local>) {
    if entry.ban_until.map(|until| until <= now).unwrap_or(false) {
        entry.ban_until = None;
    }
}

fn key_availability_in_state(
    key: &ApiKeyConfig,
    state: &mut KeyRuntimeState,
    now: DateTime<Local>,
) -> KeyAvailability {
    ensure_runtime_day(state, now);
    let mut fail_count = 0;
    let mut banned = false;
    let mut ban_until = None;

    if let Some(entry) = state.keys.get_mut(&key.id) {
        clear_expired_ban(entry, now);
        fail_count = entry.fail_count;
        ban_until = entry.ban_until.map(|until| until.timestamp());
        banned = entry.ban_until.map(|until| until > now).unwrap_or(false);
    }

    KeyAvailability {
        fail_count,
        banned,
        ban_until,
        schedulable: !key.manually_disabled && !banned,
    }
}

fn image_key_availability_in_state(
    key: &ApiKeyConfig,
    state: &mut KeyRuntimeState,
    now: DateTime<Local>,
) -> KeyAvailability {
    ensure_runtime_day(state, now);
    let mut fail_count = 0;
    let mut banned = false;
    let mut ban_until = None;

    if let Some(entry) = state.image_keys.get_mut(&key.id) {
        clear_expired_ban(entry, now);
        fail_count = entry.fail_count;
        ban_until = entry.ban_until.map(|until| until.timestamp());
        banned = entry.ban_until.map(|until| until > now).unwrap_or(false);
    }

    KeyAvailability {
        fail_count,
        banned,
        ban_until,
        schedulable: !key.manually_disabled && !banned,
    }
}

fn pool_has_single_supplier_and_key(pool: &PoolConfig) -> bool {
    pool.base_urls.len() == 1
        && pool
            .base_urls
            .first()
            .map(|base_url| base_url.api_keys.len() == 1)
            .unwrap_or(false)
}

fn image_request_has_single_candidate(pool: &PoolConfig) -> bool {
    pool.base_urls
        .iter()
        .map(|base_url| {
            if base_url.image_keys.is_empty() {
                base_url.api_keys.len()
            } else {
                base_url.image_keys.len()
            }
        })
        .sum::<usize>()
        == 1
}

fn mark_key_fail_in_state(
    key_id: i64,
    state: &mut KeyRuntimeState,
    now: DateTime<Local>,
    allow_ban: bool,
) {
    ensure_runtime_day(state, now);
    let entry = state.keys.entry(key_id).or_default();
    clear_expired_ban(entry, now);
    entry.fail_count += 1;
    if !allow_ban {
        entry.ban_until = None;
    } else if entry.fail_count >= DAILY_BAN_FAILS {
        entry.ban_until = Some(next_day_zero(now));
    } else if entry.fail_count >= TEMP_BAN_FAILS {
        entry.ban_until = Some(now + ChronoDuration::minutes(TEMP_BAN_MINUTES));
    }
}

fn mark_key_success_in_state(key_id: i64, state: &mut KeyRuntimeState, now: DateTime<Local>) {
    ensure_runtime_day(state, now);
    state.keys.remove(&key_id);
}

fn mark_image_key_fail_in_state(
    key_id: i64,
    state: &mut KeyRuntimeState,
    now: DateTime<Local>,
    allow_ban: bool,
) {
    ensure_runtime_day(state, now);
    let entry = state.image_keys.entry(key_id).or_default();
    clear_expired_ban(entry, now);
    entry.fail_count += 1;
    if !allow_ban {
        entry.ban_until = None;
    } else if entry.fail_count >= DAILY_BAN_FAILS {
        entry.ban_until = Some(next_day_zero(now));
    } else if entry.fail_count >= TEMP_BAN_FAILS {
        entry.ban_until = Some(now + ChronoDuration::minutes(TEMP_BAN_MINUTES));
    }
}

fn mark_image_key_success_in_state(key_id: i64, state: &mut KeyRuntimeState, now: DateTime<Local>) {
    ensure_runtime_day(state, now);
    state.image_keys.remove(&key_id);
}

async fn key_is_schedulable(state: &AppState, key: &ApiKeyConfig) -> bool {
    let mut runtime = state.runtime.write().await;
    key_availability_in_state(key, &mut runtime, Local::now()).schedulable
}

async fn mark_key_fail(state: &AppState, key_id: i64, allow_ban: bool) {
    let mut runtime = state.runtime.write().await;
    mark_key_fail_in_state(key_id, &mut runtime, Local::now(), allow_ban);
}

async fn mark_key_success(state: &AppState, key_id: i64) {
    let mut runtime = state.runtime.write().await;
    mark_key_success_in_state(key_id, &mut runtime, Local::now());
}

async fn image_key_is_schedulable(state: &AppState, key: &ApiKeyConfig) -> bool {
    let mut runtime = state.runtime.write().await;
    image_key_availability_in_state(key, &mut runtime, Local::now()).schedulable
}

async fn mark_image_key_fail(state: &AppState, key_id: i64, allow_ban: bool) {
    let mut runtime = state.runtime.write().await;
    mark_image_key_fail_in_state(key_id, &mut runtime, Local::now(), allow_ban);
}

async fn mark_image_key_success(state: &AppState, key_id: i64) {
    let mut runtime = state.runtime.write().await;
    mark_image_key_success_in_state(key_id, &mut runtime, Local::now());
}

async fn clear_key_runtime(state: &AppState, key_id: i64) {
    state.runtime.write().await.keys.remove(&key_id);
}

async fn clear_image_key_runtime(state: &AppState, key_id: i64) {
    state.runtime.write().await.image_keys.remove(&key_id);
}

async fn record_key_usage(
    state: &AppState,
    access_key_id: i64,
    proxy_id: i64,
    supplier_id: i64,
    key_id: i64,
) {
    let used_at = Local::now().timestamp();
    let mut usage = state.usage.write().await;
    usage.by_access_key.insert(
        access_key_id,
        AccessKeyUsage {
            access_key_id,
            proxy_id,
            supplier_id,
            key_id,
            used_at,
        },
    );
    usage.by_key.insert(
        key_id,
        KeyUsage {
            access_key_id,
            supplier_id,
            used_at,
        },
    );
}

async fn clear_access_key_usage(state: &AppState, access_key_id: i64) {
    let mut usage = state.usage.write().await;
    usage.by_access_key.remove(&access_key_id);
    usage
        .by_key
        .retain(|_, item| item.access_key_id != access_key_id);
}

async fn clear_access_key_usages<I>(state: &AppState, access_key_ids: I)
where
    I: IntoIterator<Item = i64>,
{
    let access_key_ids: HashSet<i64> = access_key_ids.into_iter().collect();
    if access_key_ids.is_empty() {
        return;
    }

    let mut usage = state.usage.write().await;
    for access_key_id in &access_key_ids {
        usage.by_access_key.remove(access_key_id);
    }
    usage
        .by_key
        .retain(|_, item| !access_key_ids.contains(&item.access_key_id));
}

async fn clear_supplier_usage(state: &AppState, supplier_id: i64) {
    let mut usage = state.usage.write().await;
    usage
        .by_access_key
        .retain(|_, item| item.supplier_id != supplier_id);
    usage
        .by_key
        .retain(|_, item| item.supplier_id != supplier_id);
}

async fn reconcile_usage_runtime(state: &AppState) {
    let pools = state.pools.read().await.clone();
    let access_keys = state.access_keys.read().await.clone();

    let valid_access_keys: HashSet<i64> = access_keys.iter().map(|item| item.id).collect();
    let valid_pools: HashSet<i64> = pools.iter().map(|pool| pool.id).collect();
    let mut valid_suppliers = HashSet::new();
    let mut valid_keys = HashSet::new();

    for pool in &pools {
        for supplier in &pool.base_urls {
            valid_suppliers.insert(supplier.id);
            for key in &supplier.api_keys {
                valid_keys.insert(key.id);
            }
        }
    }

    let mut usage = state.usage.write().await;
    usage.by_access_key.retain(|_, item| {
        valid_access_keys.contains(&item.access_key_id)
            && valid_pools.contains(&item.proxy_id)
            && valid_suppliers.contains(&item.supplier_id)
            && valid_keys.contains(&item.key_id)
    });
    usage.by_key.retain(|key_id, item| {
        valid_keys.contains(key_id)
            && valid_access_keys.contains(&item.access_key_id)
            && valid_suppliers.contains(&item.supplier_id)
    });
}

async fn clear_key_runtimes<I>(state: &AppState, key_ids: I)
where
    I: IntoIterator<Item = i64>,
{
    let mut runtime = state.runtime.write().await;
    for key_id in key_ids {
        runtime.keys.remove(&key_id);
    }
}

async fn clear_image_key_runtimes<I>(state: &AppState, key_ids: I)
where
    I: IntoIterator<Item = i64>,
{
    let mut runtime = state.runtime.write().await;
    for key_id in key_ids {
        runtime.image_keys.remove(&key_id);
    }
}

async fn clear_pool_runtime(state: &AppState, pool_id: i64) {
    let (key_ids, image_key_ids): (HashSet<i64>, HashSet<i64>) = state
        .pools
        .read()
        .await
        .iter()
        .find(|pool| pool.id == pool_id)
        .map(|pool| {
            (
                pool.base_urls
                    .iter()
                    .flat_map(|base_url| base_url.api_keys.iter().map(|key| key.id))
                    .collect(),
                pool.base_urls
                    .iter()
                    .flat_map(|base_url| base_url.image_keys.iter().map(|key| key.id))
                    .collect(),
            )
        })
        .unwrap_or_default();

    let mut runtime = state.runtime.write().await;
    for key_id in key_ids {
        runtime.keys.remove(&key_id);
    }
    for key_id in image_key_ids {
        runtime.image_keys.remove(&key_id);
    }
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

async fn pools_response(state: &AppState) -> ProxiesResponse {
    let pools = state.pools.read().await.clone();
    let access_keys = state.access_keys.read().await.clone();
    let usage = state.usage.read().await.clone();
    let now = Local::now();
    let mut runtime = state.runtime.write().await;
    ensure_runtime_day(&mut runtime, now);

    let mut binding_counts: HashMap<i64, HashSet<i64>> = HashMap::new();
    for access_key in &access_keys {
        binding_counts
            .entry(access_key.proxy_id)
            .or_default()
            .insert(access_key.id);
        if let Some(image_proxy_id) = access_key.image_proxy_id {
            binding_counts
                .entry(image_proxy_id)
                .or_default()
                .insert(access_key.id);
        }
    }
    let access_key_names: HashMap<i64, String> = access_keys
        .iter()
        .map(|item| (item.id, item.name.clone()))
        .collect();
    let proxies = pools
        .into_iter()
        .map(|pool| {
            let access_key_count = binding_counts.get(&pool.id).map(HashSet::len).unwrap_or(0);
            let mut available_supplier_count = 0usize;
            let suppliers = pool
                .base_urls
                .into_iter()
                .map(|base_url| {
                    let mut available_key_count = 0usize;
                    let supplier_usage = base_url
                        .api_keys
                        .iter()
                        .filter_map(|key| usage.by_key.get(&key.id))
                        .max_by_key(|item| item.used_at)
                        .cloned();
                    let keys = base_url
                        .api_keys
                        .into_iter()
                        .map(|key| {
                            let availability = key_availability_in_state(&key, &mut runtime, now);
                            if availability.schedulable {
                                available_key_count += 1;
                            }
                            ApiKeyView {
                                id: key.id,
                                api_key: key.api_key.clone(),
                                masked_key: mask_key(&key.api_key),
                                sort_order: key.sort_order,
                                manually_disabled: key.manually_disabled,
                                test_model: if key.test_model.is_empty() {
                                    None
                                } else {
                                    Some(key.test_model.clone())
                                },
                                test_protocol: key.test_protocol,
                                fail_count: availability.fail_count,
                                banned: availability.banned,
                                ban_until: availability.ban_until,
                                last_used: usage.by_key.contains_key(&key.id),
                                last_used_at: usage.by_key.get(&key.id).map(|item| item.used_at),
                                last_used_by_access_key_name: usage
                                    .by_key
                                    .get(&key.id)
                                    .and_then(|item| access_key_names.get(&item.access_key_id))
                                    .cloned(),
                            }
                        })
                        .collect::<Vec<_>>();
                    let image_keys = base_url
                        .image_keys
                        .into_iter()
                        .map(|key| {
                            let availability =
                                image_key_availability_in_state(&key, &mut runtime, now);
                            ApiKeyView {
                                id: key.id,
                                api_key: key.api_key.clone(),
                                masked_key: mask_key(&key.api_key),
                                sort_order: key.sort_order,
                                manually_disabled: key.manually_disabled,
                                test_model: None,
                                test_protocol: None,
                                fail_count: availability.fail_count,
                                banned: availability.banned,
                                ban_until: availability.ban_until,
                                last_used: false,
                                last_used_at: None,
                                last_used_by_access_key_name: None,
                            }
                        })
                        .collect::<Vec<_>>();
                    let schedulable = available_key_count > 0;
                    if schedulable {
                        available_supplier_count += 1;
                    }
                    SupplierView {
                        id: base_url.id,
                        proxy_id: base_url.pool_id,
                        name: base_url.name,
                        base_url: base_url.base_url,
                        protocol: base_url.protocol_mode,
                        override_model: base_url.override_model,
                        sort_order: base_url.sort_order,
                        available_key_count,
                        total_key_count: keys.len(),
                        schedulable,
                        last_used_at: supplier_usage.as_ref().map(|item| item.used_at),
                        last_used_by_access_key_name: supplier_usage
                            .as_ref()
                            .and_then(|item| access_key_names.get(&item.access_key_id))
                            .cloned(),
                        keys,
                        image_keys,
                    }
                })
                .collect::<Vec<_>>();

            ProxyView {
                id: pool.id,
                name: pool.name,
                note: pool.note,
                is_active: pool.is_active,
                access_key_count,
                in_use: access_key_count > 0,
                available_supplier_count,
                total_supplier_count: suppliers.len(),
                suppliers,
            }
        })
        .collect();

    ProxiesResponse {
        active_proxy_id: None,
        proxies,
    }
}

async fn admin_page() -> Html<&'static str> {
    Html(ADMIN_HTML)
}

async fn access_keys_page() -> Html<&'static str> {
    Html(ACCESS_KEYS_HTML)
}

async fn get_proxies(State(state): State<AppState>) -> Json<ProxiesResponse> {
    Json(pools_response(&state).await)
}

async fn access_keys_response(state: &AppState) -> AccessKeysResponse {
    let access_keys = state.access_keys.read().await.clone();
    let pools = state.pools.read().await.clone();
    let usage = state.usage.read().await.clone();

    let proxy_names: HashMap<i64, String> = pools
        .iter()
        .map(|pool| (pool.id, pool.name.clone()))
        .collect();
    let supplier_names: HashMap<i64, String> = pools
        .iter()
        .flat_map(|pool| pool.base_urls.iter())
        .map(|supplier| (supplier.id, supplier.name.clone()))
        .collect();
    let masked_key_by_id: HashMap<i64, String> = pools
        .iter()
        .flat_map(|pool| pool.base_urls.iter())
        .flat_map(|supplier| supplier.api_keys.iter())
        .map(|key| (key.id, mask_key(&key.api_key)))
        .collect();
    let proxies = pools
        .into_iter()
        .map(|pool| ProxyOptionView {
            id: pool.id,
            name: pool.name,
            suppliers: pool
                .base_urls
                .into_iter()
                .map(|supplier| SupplierOptionView {
                    id: supplier.id,
                    name: supplier.name,
                    base_url: supplier.base_url,
                })
                .collect(),
        })
        .collect();

    let access_keys = access_keys
        .into_iter()
        .map(|access_key| AccessKeyView {
            id: access_key.id,
            name: access_key.name,
            access_key: access_key.access_key.clone(),
            masked_key: mask_key(&access_key.access_key),
            proxy_id: access_key.proxy_id,
            proxy_name: proxy_names
                .get(&access_key.proxy_id)
                .cloned()
                .unwrap_or_else(|| "已删除代理".to_string()),
            image_proxy_id: access_key.image_proxy_id,
            image_proxy_name: access_key
                .image_proxy_id
                .and_then(|proxy_id| proxy_names.get(&proxy_id).cloned()),
            override_model: access_key.override_model,
            created_at: access_key.created_at,
            updated_at: access_key.updated_at,
            last_used_supplier_name: usage
                .by_access_key
                .get(&access_key.id)
                .and_then(|item| supplier_names.get(&item.supplier_id))
                .cloned(),
            last_used_key_masked: usage
                .by_access_key
                .get(&access_key.id)
                .and_then(|item| masked_key_by_id.get(&item.key_id))
                .cloned(),
            last_used_at: usage
                .by_access_key
                .get(&access_key.id)
                .map(|item| item.used_at),
        })
        .collect();

    AccessKeysResponse {
        access_keys,
        proxies,
    }
}

async fn get_access_keys(State(state): State<AppState>) -> Json<AccessKeysResponse> {
    Json(access_keys_response(&state).await)
}

async fn create_proxy_save(
    State(state): State<AppState>,
    Json(payload): Json<ProxySavePayload>,
) -> Response<Body> {
    save_proxy_draft(&state, None, payload).await
}

async fn update_proxy_save(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(payload): Json<ProxySavePayload>,
) -> Response<Body> {
    save_proxy_draft(&state, Some(id), payload).await
}

async fn save_proxy_draft(
    state: &AppState,
    pool_id: Option<i64>,
    payload: ProxySavePayload,
) -> Response<Body> {
    let payload = match prepare_proxy_save_payload(payload) {
        Ok(payload) => payload,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };

    let (saved_pool_id, runtime_ids_to_clear, image_runtime_ids_to_clear) = {
        let mut conn = match open_db(&state.db_path) {
            Ok(conn) => conn,
            Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
        };

        match pool_name_exists(&conn, &payload.name, pool_id) {
            Ok(true) => return json_error(StatusCode::BAD_REQUEST, "proxy name already exists"),
            Ok(false) => {}
            Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
        }

        let tx = match conn.transaction() {
            Ok(tx) => tx,
            Err(err) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to start save transaction: {err}"),
                )
            }
        };

        let save_result: Result<(i64, HashSet<i64>, HashSet<i64>), String> = (|| {
            let saved_pool_id = if let Some(pool_id) = pool_id {
                let exists = tx
                    .query_row(
                        "SELECT 1 FROM pools WHERE id = ?1",
                        params![pool_id],
                        |_| Ok(()),
                    )
                    .optional()
                    .map_err(|err| format!("Failed to load proxy: {err}"))?
                    .is_some();
                if !exists {
                    return Err("proxy not found".to_string());
                }

                tx.execute(
                    "UPDATE pools
                     SET name = ?1, note = ?2, updated_at = strftime('%s', 'now')
                     WHERE id = ?3",
                    params![&payload.name, &payload.note, pool_id],
                )
                .map_err(|err| format!("Failed to update proxy: {err}"))?;
                pool_id
            } else {
                let existing_count: i64 = tx
                    .query_row("SELECT COUNT(*) FROM pools", [], |row| row.get(0))
                    .map_err(|err| format!("Failed to query proxies: {err}"))?;
                // 将新代理放在列表最前面
                tx.execute(
                    "UPDATE pools SET sort_order = sort_order + 1, updated_at = strftime('%s', 'now')",
                    [],
                )
                .map_err(|err| format!("Failed to update proxy sort_order: {err}"))?;
                tx.execute(
                    "INSERT INTO pools (name, note, is_active, sort_order, created_at, updated_at)
                     VALUES (?1, ?2, ?3, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                    params![
                        &payload.name,
                        &payload.note,
                        if existing_count == 0 { 1 } else { 0 }
                    ],
                )
                .map_err(|err| format!("Failed to create proxy: {err}"))?;
                tx.last_insert_rowid()
            };

            let mut existing_supplier_ids = HashSet::new();
            let mut existing_keys_by_supplier: HashMap<i64, HashMap<i64, String>> = HashMap::new();
            let mut existing_image_keys_by_supplier: HashMap<i64, HashMap<i64, String>> =
                HashMap::new();

            if pool_id.is_some() {
                let mut supplier_stmt = tx
                    .prepare("SELECT id FROM pool_base_urls WHERE pool_id = ?1")
                    .map_err(|err| format!("Failed to prepare supplier query: {err}"))?;
                let supplier_rows = supplier_stmt
                    .query_map(params![saved_pool_id], |row| row.get::<_, i64>(0))
                    .map_err(|err| format!("Failed to query suppliers: {err}"))?;
                for row in supplier_rows {
                    existing_supplier_ids.insert(
                        row.map_err(|err| format!("Failed to decode supplier row: {err}"))?,
                    );
                }

                let mut key_stmt = tx
                    .prepare(
                        "SELECT k.id, k.base_url_id, k.api_key
                         FROM pool_api_keys k
                         JOIN pool_base_urls b ON b.id = k.base_url_id
                         WHERE b.pool_id = ?1",
                    )
                    .map_err(|err| format!("Failed to prepare api_key query: {err}"))?;
                let key_rows = key_stmt
                    .query_map(params![saved_pool_id], |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    })
                    .map_err(|err| format!("Failed to query api_keys: {err}"))?;
                for row in key_rows {
                    let (key_id, supplier_id, api_key) =
                        row.map_err(|err| format!("Failed to decode api_key row: {err}"))?;
                    existing_keys_by_supplier
                        .entry(supplier_id)
                        .or_default()
                        .insert(key_id, api_key);
                }

                let mut image_key_stmt = tx
                    .prepare(
                        "SELECT k.id, k.base_url_id, k.api_key
                         FROM pool_image_keys k
                         JOIN pool_base_urls b ON b.id = k.base_url_id
                         WHERE b.pool_id = ?1",
                    )
                    .map_err(|err| format!("Failed to prepare image key query: {err}"))?;
                let image_key_rows = image_key_stmt
                    .query_map(params![saved_pool_id], |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    })
                    .map_err(|err| format!("Failed to query image keys: {err}"))?;
                for row in image_key_rows {
                    let (key_id, supplier_id, api_key) =
                        row.map_err(|err| format!("Failed to decode image key row: {err}"))?;
                    existing_image_keys_by_supplier
                        .entry(supplier_id)
                        .or_default()
                        .insert(key_id, api_key);
                }
            }

            let mut seen_supplier_ids = HashSet::new();
            let mut runtime_ids_to_clear = HashSet::new();
            let mut image_runtime_ids_to_clear = HashSet::new();

            for (supplier_sort_order, supplier) in payload.suppliers.iter().enumerate() {
                let supplier_id = if let Some(supplier_id) = supplier.id {
                    if !existing_supplier_ids.contains(&supplier_id) {
                        return Err("supplier not found".to_string());
                    }
                    if !seen_supplier_ids.insert(supplier_id) {
                        return Err("duplicate supplier id in payload".to_string());
                    }

                    tx.execute(
                        "UPDATE pool_base_urls
                         SET name = ?1, base_url = ?2, protocol_mode = ?3, override_model = ?4, sort_order = ?5,
                             updated_at = strftime('%s', 'now')
                         WHERE id = ?6",
                        params![
                            &supplier.name,
                            &supplier.base_url,
                            supplier.protocol.as_str(),
                            &supplier.override_model,
                            supplier_sort_order as i64,
                            supplier_id
                        ],
                    )
                    .map_err(|err| format!("Failed to update supplier: {err}"))?;
                    supplier_id
                } else {
                    tx.execute(
                        "INSERT INTO pool_base_urls
                         (pool_id, name, base_url, protocol_mode, override_model, sort_order, created_at, updated_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, strftime('%s', 'now'), strftime('%s', 'now'))",
                        params![
                            saved_pool_id,
                            &supplier.name,
                            &supplier.base_url,
                            supplier.protocol.as_str(),
                            &supplier.override_model,
                            supplier_sort_order as i64
                        ],
                    )
                    .map_err(|err| format!("Failed to create supplier: {err}"))?;
                    tx.last_insert_rowid()
                };

                let supplier_existing_keys = existing_keys_by_supplier
                    .get(&supplier_id)
                    .cloned()
                    .unwrap_or_default();
                let mut seen_key_ids = HashSet::new();

                for (key_sort_order, key) in supplier.keys.iter().enumerate() {
                    if let Some(key_id) = key.id {
                        let Some(existing_api_key) = supplier_existing_keys.get(&key_id) else {
                            return Err("api_key not found".to_string());
                        };
                        if !seen_key_ids.insert(key_id) {
                            return Err("duplicate api_key id in payload".to_string());
                        }

                        if existing_api_key != &key.api_key {
                            runtime_ids_to_clear.insert(key_id);
                        }

                        tx.execute(
                            "UPDATE pool_api_keys
                             SET api_key = ?1, sort_order = ?2, updated_at = strftime('%s', 'now')
                             WHERE id = ?3",
                            params![&key.api_key, key_sort_order as i64, key_id],
                        )
                        .map_err(|err| format!("Failed to update api_key: {err}"))?;
                    } else {
                        tx.execute(
                            "INSERT INTO pool_api_keys
                             (base_url_id, api_key, sort_order, manually_disabled, created_at, updated_at)
                             VALUES (?1, ?2, ?3, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                            params![supplier_id, &key.api_key, key_sort_order as i64],
                        )
                        .map_err(|err| format!("Failed to create api_key: {err}"))?;
                    }
                }

                for key_id in supplier_existing_keys.keys() {
                    if !seen_key_ids.contains(key_id) {
                        tx.execute("DELETE FROM pool_api_keys WHERE id = ?1", params![key_id])
                            .map_err(|err| format!("Failed to delete api_key: {err}"))?;
                        runtime_ids_to_clear.insert(*key_id);
                    }
                }

                let supplier_existing_image_keys = existing_image_keys_by_supplier
                    .get(&supplier_id)
                    .cloned()
                    .unwrap_or_default();
                let mut seen_image_key_ids = HashSet::new();

                for (key_sort_order, key) in supplier.image_keys.iter().enumerate() {
                    if let Some(key_id) = key.id {
                        let Some(existing_api_key) = supplier_existing_image_keys.get(&key_id)
                        else {
                            return Err("image_api_key not found".to_string());
                        };
                        if !seen_image_key_ids.insert(key_id) {
                            return Err("duplicate image_api_key id in payload".to_string());
                        }

                        if existing_api_key != &key.api_key {
                            image_runtime_ids_to_clear.insert(key_id);
                        }

                        tx.execute(
                            "UPDATE pool_image_keys
                             SET api_key = ?1, sort_order = ?2, updated_at = strftime('%s', 'now')
                             WHERE id = ?3",
                            params![&key.api_key, key_sort_order as i64, key_id],
                        )
                        .map_err(|err| format!("Failed to update image api_key: {err}"))?;
                    } else {
                        tx.execute(
                            "INSERT INTO pool_image_keys
                             (base_url_id, api_key, sort_order, manually_disabled, created_at, updated_at)
                             VALUES (?1, ?2, ?3, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                            params![supplier_id, &key.api_key, key_sort_order as i64],
                        )
                        .map_err(|err| format!("Failed to create image api_key: {err}"))?;
                    }
                }

                for key_id in supplier_existing_image_keys.keys() {
                    if !seen_image_key_ids.contains(key_id) {
                        tx.execute("DELETE FROM pool_image_keys WHERE id = ?1", params![key_id])
                            .map_err(|err| format!("Failed to delete image api_key: {err}"))?;
                        image_runtime_ids_to_clear.insert(*key_id);
                    }
                }
            }

            for supplier_id in existing_supplier_ids {
                if !seen_supplier_ids.contains(&supplier_id) {
                    if let Some(existing_keys) = existing_keys_by_supplier.get(&supplier_id) {
                        for key_id in existing_keys.keys() {
                            runtime_ids_to_clear.insert(*key_id);
                        }
                    }
                    if let Some(existing_image_keys) =
                        existing_image_keys_by_supplier.get(&supplier_id)
                    {
                        for key_id in existing_image_keys.keys() {
                            image_runtime_ids_to_clear.insert(*key_id);
                        }
                    }
                    tx.execute(
                        "DELETE FROM pool_base_urls WHERE id = ?1",
                        params![supplier_id],
                    )
                    .map_err(|err| format!("Failed to delete supplier: {err}"))?;
                }
            }

            Ok((
                saved_pool_id,
                runtime_ids_to_clear,
                image_runtime_ids_to_clear,
            ))
        })();

        let (saved_pool_id, runtime_ids_to_clear, image_runtime_ids_to_clear) = match save_result {
            Ok(result) => result,
            Err(err) if err == "proxy not found" => {
                return json_error(StatusCode::NOT_FOUND, err);
            }
            Err(err)
                if err == "supplier not found"
                    || err == "api_key not found"
                    || err == "image_api_key not found"
                    || err == "duplicate supplier id in payload"
                    || err == "duplicate api_key id in payload"
                    || err == "duplicate image_api_key id in payload" =>
            {
                return json_error(StatusCode::BAD_REQUEST, err);
            }
            Err(err) if err.starts_with("Failed to ") => {
                return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
            }
            Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
        };

        if let Err(err) = tx.commit() {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to commit proxy save: {err}"),
            );
        }

        (
            saved_pool_id,
            runtime_ids_to_clear,
            image_runtime_ids_to_clear,
        )
    };

    clear_key_runtimes(state, runtime_ids_to_clear).await;
    clear_image_key_runtimes(state, image_runtime_ids_to_clear).await;
    if let Err(err) = reload_state(state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }

    json_id_ok(saved_pool_id)
}

async fn create_pool(
    State(state): State<AppState>,
    Json(payload): Json<PoolPayload>,
) -> Response<Body> {
    let name = match normalize_pool_name(&payload.name) {
        Ok(name) => name,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };
    let note = normalize_note(payload.note.as_deref().unwrap_or(""));

    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    match pool_name_exists(&conn, &name, None) {
        Ok(true) => return json_error(StatusCode::BAD_REQUEST, "proxy name already exists"),
        Ok(false) => {}
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    }

    let existing_count: i64 =
        match conn.query_row("SELECT COUNT(*) FROM pools", [], |row| row.get(0)) {
            Ok(count) => count,
            Err(err) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to query proxies: {err}"),
                )
            }
        };

    // 将新代理放在列表最前面
    if let Err(err) = conn.execute(
        "UPDATE pools SET sort_order = sort_order + 1, updated_at = strftime('%s', 'now')",
        [],
    ) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to update proxy sort_order: {err}"),
        );
    }

    if let Err(err) = conn.execute(
        "INSERT INTO pools (name, note, is_active, sort_order, created_at, updated_at)
         VALUES (?1, ?2, ?3, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
        params![name, note, if existing_count == 0 { 1 } else { 0 }],
    ) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create proxy: {err}"),
        );
    }

    if let Err(err) = reload_state(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

#[derive(Deserialize)]
struct ReorderPayload {
    ids: Vec<i64>,
}

async fn reorder_pools(
    State(state): State<AppState>,
    Json(payload): Json<ReorderPayload>,
) -> Response<Body> {
    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    for (index, id) in payload.ids.iter().enumerate() {
        if let Err(err) = conn.execute(
            "UPDATE pools SET sort_order = ?1, updated_at = strftime('%s', 'now') WHERE id = ?2",
            params![index as i64, id],
        ) {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to update proxy sort_order: {err}"),
            );
        }
    }

    if let Err(err) = reload_state(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn update_pool(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(payload): Json<PoolPayload>,
) -> Response<Body> {
    let name = match normalize_pool_name(&payload.name) {
        Ok(name) => name,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };
    let note = normalize_note(payload.note.as_deref().unwrap_or(""));

    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    match pool_name_exists(&conn, &name, Some(id)) {
        Ok(true) => return json_error(StatusCode::BAD_REQUEST, "proxy name already exists"),
        Ok(false) => {}
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    }

    match conn.execute(
        "UPDATE pools
         SET name = ?1, note = ?2, updated_at = strftime('%s', 'now')
         WHERE id = ?3",
        params![name, note, id],
    ) {
        Ok(0) => return json_error(StatusCode::NOT_FOUND, "proxy not found"),
        Ok(_) => {}
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to update proxy: {err}"),
            )
        }
    }

    if let Err(err) = reload_state(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn create_access_key(
    State(state): State<AppState>,
    Json(payload): Json<AccessKeyPayload>,
) -> Response<Body> {
    let name = match normalize_access_key_name(&payload.name) {
        Ok(name) => name,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };
    let override_model = payload.override_model.trim().to_string();

    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    match proxy_exists(&conn, payload.proxy_id) {
        Ok(true) => {}
        Ok(false) => return json_error(StatusCode::BAD_REQUEST, "proxy not found"),
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    }
    if let Some(image_proxy_id) = payload.image_proxy_id {
        match proxy_exists(&conn, image_proxy_id) {
            Ok(true) => {}
            Ok(false) => return json_error(StatusCode::BAD_REQUEST, "image proxy not found"),
            Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
        }
    }

    let access_key = match generate_unique_access_key(&conn) {
        Ok(access_key) => access_key,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let id = match conn.execute(
        "INSERT INTO access_keys (name, access_key, proxy_id, image_proxy_id, override_model, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, strftime('%s', 'now'), strftime('%s', 'now'))",
        params![
            name,
            access_key,
            payload.proxy_id,
            payload.image_proxy_id,
            override_model
        ],
    ) {
        Ok(_) => conn.last_insert_rowid(),
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create access key: {err}"),
            )
        }
    };

    if let Err(err) = reload_access_keys(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_id_ok(id)
}

async fn update_access_key(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(payload): Json<AccessKeyPayload>,
) -> Response<Body> {
    let name = match normalize_access_key_name(&payload.name) {
        Ok(name) => name,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };
    let override_model = payload.override_model.trim().to_string();

    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    match proxy_exists(&conn, payload.proxy_id) {
        Ok(true) => {}
        Ok(false) => return json_error(StatusCode::BAD_REQUEST, "proxy not found"),
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    }
    if let Some(image_proxy_id) = payload.image_proxy_id {
        match proxy_exists(&conn, image_proxy_id) {
            Ok(true) => {}
            Ok(false) => return json_error(StatusCode::BAD_REQUEST, "image proxy not found"),
            Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
        }
    }

    let current_proxy_ids: Option<(i64, Option<i64>)> = match conn
        .query_row(
            "SELECT proxy_id, image_proxy_id FROM access_keys WHERE id = ?1",
            params![id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
    {
        Ok(value) => value,
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load access key: {err}"),
            )
        }
    };
    let Some((current_proxy_id, current_image_proxy_id)) = current_proxy_ids else {
        return json_error(StatusCode::NOT_FOUND, "access key not found");
    };

    match conn.execute(
        "UPDATE access_keys
         SET name = ?1, proxy_id = ?2, image_proxy_id = ?3, override_model = ?4, updated_at = strftime('%s', 'now')
         WHERE id = ?5",
        params![
            name,
            payload.proxy_id,
            payload.image_proxy_id,
            override_model,
            id
        ],
    ) {
        Ok(_) => {}
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to update access key: {err}"),
            )
        }
    }

    if current_proxy_id != payload.proxy_id || current_image_proxy_id != payload.image_proxy_id {
        clear_access_key_usage(&state, id).await;
    }
    if let Err(err) = reload_access_keys(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn delete_access_key(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    match conn.execute("DELETE FROM access_keys WHERE id = ?1", params![id]) {
        Ok(0) => return json_error(StatusCode::NOT_FOUND, "access key not found"),
        Ok(_) => {}
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to delete access key: {err}"),
            )
        }
    }

    clear_access_key_usage(&state, id).await;
    if let Err(err) = reload_access_keys(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn activate_pool(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    let exists = state.pools.read().await.iter().any(|pool| pool.id == id);
    if !exists {
        return json_error(StatusCode::NOT_FOUND, "proxy not found");
    }
    json_error(
        StatusCode::BAD_REQUEST,
        "global proxy activation has been removed; bind a me-key to use this proxy",
    )
}

async fn delete_pool(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let was_active: Option<bool> = match conn
        .query_row(
            "SELECT is_active FROM pools WHERE id = ?1",
            params![id],
            |row| Ok(row.get::<_, i64>(0)? != 0),
        )
        .optional()
    {
        Ok(value) => value,
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load proxy: {err}"),
            )
        }
    };
    let Some(was_active) = was_active else {
        return json_error(StatusCode::NOT_FOUND, "proxy not found");
    };

    let bound_access_key_ids = {
        let mut stmt = match conn
            .prepare("SELECT id FROM access_keys WHERE proxy_id = ?1 OR image_proxy_id = ?1")
        {
            Ok(stmt) => stmt,
            Err(err) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to prepare bound access key query: {err}"),
                )
            }
        };
        let rows = match stmt.query_map(params![id], |row| row.get::<_, i64>(0)) {
            Ok(rows) => rows,
            Err(err) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to query bound access keys: {err}"),
                )
            }
        };
        let mut ids = Vec::new();
        for row in rows {
            match row {
                Ok(value) => ids.push(value),
                Err(err) => {
                    return json_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to decode bound access key: {err}"),
                    )
                }
            }
        }
        ids
    };

    if let Err(err) = conn.execute(
        "UPDATE access_keys
         SET image_proxy_id = NULL, updated_at = strftime('%s', 'now')
         WHERE image_proxy_id = ?1 AND proxy_id != ?1",
        params![id],
    ) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to clear image proxy bindings: {err}"),
        );
    }

    if let Err(err) = conn.execute("DELETE FROM pools WHERE id = ?1", params![id]) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to delete proxy: {err}"),
        );
    }

    // 重新排序剩余的代理
    if let Err(err) = normalize_pool_orders(&conn) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }

    if was_active {
        let next_pool_id: Option<i64> = match conn
            .query_row(
                "SELECT id FROM pools ORDER BY created_at ASC, id ASC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()
        {
            Ok(value) => value,
            Err(err) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to load next proxy: {err}"),
                )
            }
        };

        if let Some(next_pool_id) = next_pool_id {
            if let Err(err) = ensure_single_active_pool(&conn, next_pool_id) {
                return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
            }
        }
    }

    clear_access_key_usages(&state, bound_access_key_ids).await;
    if let Err(err) = reload_state(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn create_base_url(
    State(state): State<AppState>,
    Path(pool_id): Path<i64>,
    Json(payload): Json<BaseUrlPayload>,
) -> Response<Body> {
    let supplier_name = payload.name.trim().to_string();
    let base_url = match normalize_base_url(&payload.base_url) {
        Ok(base_url) => base_url,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };

    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let pool_exists = match conn
        .query_row(
            "SELECT 1 FROM pools WHERE id = ?1",
            params![pool_id],
            |_| Ok(()),
        )
        .optional()
    {
        Ok(value) => value.is_some(),
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load proxy: {err}"),
            )
        }
    };
    if !pool_exists {
        return json_error(StatusCode::NOT_FOUND, "proxy not found");
    }

    match duplicate_base_url_exists(&conn, pool_id, &base_url, None) {
        Ok(true) => return json_error(StatusCode::BAD_REQUEST, "base_url already exists in proxy"),
        Ok(false) => {}
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    }

    let sort_order = payload.sort_order.unwrap_or_else(|| {
        next_sort_order(&conn, "pool_base_urls", "pool_id", pool_id).unwrap_or(0)
    });

    if let Err(err) = conn.execute(
        "INSERT INTO pool_base_urls (pool_id, name, base_url, protocol_mode, override_model, sort_order, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, strftime('%s', 'now'), strftime('%s', 'now'))",
        params![
            pool_id,
            supplier_name,
            base_url,
            payload.protocol.as_str(),
            payload.override_model,
            sort_order
        ],
    ) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create supplier: {err}"),
        );
    }
    if let Err(err) = normalize_base_url_orders(&conn, pool_id) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    if let Err(err) = reload_pools(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn update_base_url(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(payload): Json<BaseUrlPayload>,
) -> Response<Body> {
    let supplier_name = payload.name.trim().to_string();
    let base_url = match normalize_base_url(&payload.base_url) {
        Ok(base_url) => base_url,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };

    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let current: Option<(i64, i64)> = match conn
        .query_row(
            "SELECT pool_id, sort_order FROM pool_base_urls WHERE id = ?1",
            params![id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
    {
        Ok(value) => value,
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load supplier: {err}"),
            )
        }
    };
    let Some((pool_id, current_sort_order)) = current else {
        return json_error(StatusCode::NOT_FOUND, "supplier not found");
    };

    match duplicate_base_url_exists(&conn, pool_id, &base_url, Some(id)) {
        Ok(true) => return json_error(StatusCode::BAD_REQUEST, "base_url already exists in proxy"),
        Ok(false) => {}
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    }

    let sort_order = payload.sort_order.unwrap_or(current_sort_order);
    if let Err(err) = conn.execute(
        "UPDATE pool_base_urls
         SET name = ?1, base_url = ?2, protocol_mode = ?3, override_model = ?4, sort_order = ?5, updated_at = strftime('%s', 'now')
         WHERE id = ?6",
        params![
            supplier_name,
            base_url,
            payload.protocol.as_str(),
            payload.override_model,
            sort_order,
            id
        ],
    ) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to update supplier: {err}"),
        );
    }
    if let Err(err) = normalize_base_url_orders(&conn, pool_id) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    if let Err(err) = reload_pools(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn delete_base_url(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let pool_id: Option<i64> = match conn
        .query_row(
            "SELECT pool_id FROM pool_base_urls WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )
        .optional()
    {
        Ok(value) => value,
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load supplier: {err}"),
            )
        }
    };
    let Some(pool_id) = pool_id else {
        return json_error(StatusCode::NOT_FOUND, "supplier not found");
    };

    if let Err(err) = conn.execute("DELETE FROM pool_base_urls WHERE id = ?1", params![id]) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to delete supplier: {err}"),
        );
    }
    if let Err(err) = normalize_base_url_orders(&conn, pool_id) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    clear_supplier_usage(&state, id).await;
    if let Err(err) = reload_pools(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn create_key(
    State(state): State<AppState>,
    Path(base_url_id): Path<i64>,
    Json(payload): Json<ApiKeyPayload>,
) -> Response<Body> {
    let api_key = match normalize_api_key(&payload.api_key) {
        Ok(api_key) => api_key,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };

    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let base_exists = match conn
        .query_row(
            "SELECT 1 FROM pool_base_urls WHERE id = ?1",
            params![base_url_id],
            |_| Ok(()),
        )
        .optional()
    {
        Ok(value) => value.is_some(),
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load supplier: {err}"),
            )
        }
    };
    if !base_exists {
        return json_error(StatusCode::NOT_FOUND, "supplier not found");
    }

    match duplicate_api_key_exists(&conn, base_url_id, &api_key, None) {
        Ok(true) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "api_key already exists in supplier",
            )
        }
        Ok(false) => {}
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    }

    let sort_order = payload.sort_order.unwrap_or_else(|| {
        next_sort_order(&conn, "pool_api_keys", "base_url_id", base_url_id).unwrap_or(0)
    });

    if let Err(err) = conn.execute(
        "INSERT INTO pool_api_keys
         (base_url_id, api_key, sort_order, manually_disabled, created_at, updated_at)
         VALUES (?1, ?2, ?3, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
        params![base_url_id, api_key, sort_order],
    ) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create api_key: {err}"),
        );
    }
    if let Err(err) = normalize_api_key_orders(&conn, base_url_id) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    if let Err(err) = reload_pools(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn update_key(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(payload): Json<ApiKeyPayload>,
) -> Response<Body> {
    let api_key = match normalize_api_key(&payload.api_key) {
        Ok(api_key) => api_key,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };

    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let current: Option<(i64, i64)> = match conn
        .query_row(
            "SELECT base_url_id, sort_order FROM pool_api_keys WHERE id = ?1",
            params![id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
    {
        Ok(value) => value,
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load api_key: {err}"),
            )
        }
    };
    let Some((base_url_id, current_sort_order)) = current else {
        return json_error(StatusCode::NOT_FOUND, "api_key not found");
    };

    match duplicate_api_key_exists(&conn, base_url_id, &api_key, Some(id)) {
        Ok(true) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "api_key already exists in supplier",
            )
        }
        Ok(false) => {}
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    }

    let sort_order = payload.sort_order.unwrap_or(current_sort_order);
    if let Err(err) = conn.execute(
        "UPDATE pool_api_keys
         SET api_key = ?1, sort_order = ?2, updated_at = strftime('%s', 'now')
         WHERE id = ?3",
        params![api_key, sort_order, id],
    ) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to update api_key: {err}"),
        );
    }
    if let Err(err) = normalize_api_key_orders(&conn, base_url_id) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    if let Err(err) = reload_pools(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn delete_key(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let base_url_id: Option<i64> = match conn
        .query_row(
            "SELECT base_url_id FROM pool_api_keys WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )
        .optional()
    {
        Ok(value) => value,
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load api_key: {err}"),
            )
        }
    };
    let Some(base_url_id) = base_url_id else {
        return json_error(StatusCode::NOT_FOUND, "api_key not found");
    };

    if let Err(err) = conn.execute("DELETE FROM pool_api_keys WHERE id = ?1", params![id]) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to delete api_key: {err}"),
        );
    }
    if let Err(err) = normalize_api_key_orders(&conn, base_url_id) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    clear_key_runtime(&state, id).await;
    if let Err(err) = reload_pools(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn disable_key(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    match conn.execute(
        "UPDATE pool_api_keys
         SET manually_disabled = 1, updated_at = strftime('%s', 'now')
         WHERE id = ?1",
        params![id],
    ) {
        Ok(0) => return json_error(StatusCode::NOT_FOUND, "api_key not found"),
        Ok(_) => {}
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to disable api_key: {err}"),
            )
        }
    }

    if let Err(err) = reload_pools(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn enable_key(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    match conn.execute(
        "UPDATE pool_api_keys
         SET manually_disabled = 0, updated_at = strftime('%s', 'now')
         WHERE id = ?1",
        params![id],
    ) {
        Ok(0) => return json_error(StatusCode::NOT_FOUND, "api_key not found"),
        Ok(_) => {}
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to enable api_key: {err}"),
            )
        }
    }

    clear_key_runtime(&state, id).await;
    if let Err(err) = reload_pools(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn unban_key(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    let exists = state
        .pools
        .read()
        .await
        .iter()
        .flat_map(|pool| pool.base_urls.iter())
        .flat_map(|base_url| base_url.api_keys.iter())
        .any(|key| key.id == id);
    if !exists {
        return json_error(StatusCode::NOT_FOUND, "api_key not found");
    }

    clear_key_runtime(&state, id).await;
    json_ok()
}

async fn set_image_key_disabled(state: &AppState, id: i64, disabled: bool) -> Response<Body> {
    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    match conn.execute(
        "UPDATE pool_image_keys
         SET manually_disabled = ?1, updated_at = strftime('%s', 'now')
         WHERE id = ?2",
        params![if disabled { 1 } else { 0 }, id],
    ) {
        Ok(0) => return json_error(StatusCode::NOT_FOUND, "image api_key not found"),
        Ok(_) => {}
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to update image api_key: {err}"),
            )
        }
    }

    if !disabled {
        clear_image_key_runtime(state, id).await;
    }
    if let Err(err) = reload_pools(state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }
    json_ok()
}

async fn disable_image_key(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    set_image_key_disabled(&state, id, true).await
}

async fn enable_image_key(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    set_image_key_disabled(&state, id, false).await
}

async fn unban_image_key(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    let exists = state
        .pools
        .read()
        .await
        .iter()
        .flat_map(|pool| pool.base_urls.iter())
        .flat_map(|base_url| base_url.image_keys.iter())
        .any(|key| key.id == id);
    if !exists {
        return json_error(StatusCode::NOT_FOUND, "image api_key not found");
    }

    clear_image_key_runtime(&state, id).await;
    json_ok()
}

fn parse_model_ids(body: &[u8]) -> Result<Vec<String>, String> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|err| format!("invalid models response JSON: {err}"))?;
    let data = value
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| "models response missing data".to_string())?;
    let mut models = Vec::new();
    let mut seen = HashSet::new();
    for item in data {
        let Some(model) = item.get("id").and_then(Value::as_str) else {
            continue;
        };
        if seen.insert(model.to_string()) {
            models.push(model.to_string());
        }
    }
    Ok(models)
}

fn test_request_body(protocol: TestProtocol, model: &str) -> Result<Vec<u8>, String> {
    let body = match protocol {
        TestProtocol::Chat => json!({
            "model": model,
            "messages": [{ "role": "user", "content": "hi" }]
        }),
        TestProtocol::Responses => json!({
            "model": model,
            "input": "hi"
        }),
    };
    serde_json::to_vec(&body).map_err(|err| format!("failed to encode test request: {err}"))
}

async fn key_models(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let target = match key_test_target(&conn, id) {
        Ok(Some(target)) => target,
        Ok(None) => return json_error(StatusCode::NOT_FOUND, "api_key not found"),
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let models = match load_upstream_models(&state, &target.base_url, &target.api_key).await {
        Ok(models) => models,
        Err(response) => return response,
    };
    Json(KeyModelsResponse {
        models,
        selected_model: target.selected_model,
        selected_protocol: target.selected_protocol,
        supplier_protocol: target.supplier_protocol,
    })
    .into_response()
}

async fn load_upstream_models(
    state: &AppState,
    base_url: &str,
    api_key: &str,
) -> Result<Vec<String>, Response<Body>> {
    let upstream_url = match build_upstream_url(base_url, MODELS_PATH, None) {
        Ok(url) => url,
        Err(err) => return Err(json_error(StatusCode::BAD_REQUEST, err)),
    };
    let authority = match upstream_authority(&upstream_url) {
        Ok(authority) => authority,
        Err(err) => return Err(json_error(StatusCode::BAD_GATEWAY, err)),
    };
    let headers = match build_upstream_headers(&HeaderMap::new(), &authority, api_key) {
        Ok(headers) => headers,
        Err(err) => return Err(json_error(StatusCode::BAD_GATEWAY, err)),
    };

    match state.client.get(upstream_url).headers(headers).send().await {
        Ok(response) if response.status().is_success() => {
            let body = match response.bytes().await {
                Ok(bytes) => bytes,
                Err(err) => {
                    return Err(json_error(
                        StatusCode::BAD_GATEWAY,
                        format!("Failed to read upstream models response: {err}"),
                    ))
                }
            };
            parse_model_ids(&body).map_err(|err| json_error(StatusCode::BAD_GATEWAY, err))
        }
        Ok(response) => Err(json_error(
            StatusCode::BAD_GATEWAY,
            format!("Failed to load models ({})", response.status().as_u16()),
        )),
        Err(err) if err.is_timeout() => Err(json_error(StatusCode::BAD_GATEWAY, "加载模型超时")),
        Err(err) => Err(json_error(
            StatusCode::BAD_GATEWAY,
            format!("加载模型失败: {err}"),
        )),
    }
}

async fn supplier_models(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let base_url: Option<String> = match conn
        .query_row(
            "SELECT base_url FROM pool_base_urls WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )
        .optional()
    {
        Ok(value) => value,
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load supplier: {err}"),
            )
        }
    };
    let Some(base_url) = base_url else {
        return json_error(StatusCode::NOT_FOUND, "supplier not found");
    };

    let api_key: Option<String> = match conn
        .query_row(
            "SELECT api_key
             FROM pool_api_keys
             WHERE base_url_id = ?1 AND manually_disabled = 0
             ORDER BY sort_order ASC, id ASC
             LIMIT 1",
            params![id],
            |row| row.get(0),
        )
        .optional()
    {
        Ok(value) => value,
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load supplier api_key: {err}"),
            )
        }
    };
    let Some(api_key) = api_key else {
        return json_error(StatusCode::BAD_REQUEST, "supplier has no available api_key");
    };

    match load_upstream_models(&state, &base_url, &api_key).await {
        Ok(models) => Json(SupplierModelsResponse { models }).into_response(),
        Err(response) => response,
    }
}

async fn test_key(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(payload): Json<KeyTestRequest>,
) -> Response<Body> {
    let model = match normalize_test_model(&payload.model) {
        Ok(model) => model,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };

    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let Some(target) = (match key_test_target(&conn, id) {
        Ok(target) => target,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    }) else {
        return json_error(StatusCode::NOT_FOUND, "api_key not found");
    };

    let protocol = match resolve_test_protocol(&target.supplier_protocol, payload.protocol) {
        Ok(protocol) => protocol,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };

    if let Err(err) = conn.execute(
        "UPDATE pool_api_keys
         SET test_model = ?1, test_protocol = ?2, updated_at = strftime('%s', 'now')
         WHERE id = ?3",
        params![&model, protocol.as_str(), id],
    ) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to save test settings: {err}"),
        );
    }
    if let Err(err) = reload_pools(&state).await {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, err);
    }

    let upstream_url = match build_upstream_url(&target.base_url, protocol.endpoint().path(), None)
    {
        Ok(url) => url,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };
    let authority = match upstream_authority(&upstream_url) {
        Ok(authority) => authority,
        Err(err) => return json_error(StatusCode::BAD_GATEWAY, err),
    };
    let headers = match build_upstream_headers(&HeaderMap::new(), &authority, &target.api_key) {
        Ok(headers) => headers,
        Err(err) => return json_error(StatusCode::BAD_GATEWAY, err),
    };
    let body = match test_request_body(protocol, &model) {
        Ok(body) => body,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    match state
        .client
        .post(upstream_url)
        .headers(headers)
        .body(body)
        .send()
        .await
    {
        Ok(response) => {
            let status = response.status();
            let ok = status.is_success();
            let message = if ok {
                format!("测试成功 ({})", status.as_u16())
            } else {
                format!("测试失败 ({})", status.as_u16())
            };
            json_key_test_ok(message, Some(status.as_u16()), ok)
        }
        Err(err) if err.is_timeout() => json_key_test_ok("测试超时", None, false),
        Err(err) => json_key_test_ok(format!("测试失败: {err}"), None, false),
    }
}

async fn unban_all_pool(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    let exists = state.pools.read().await.iter().any(|pool| pool.id == id);
    if !exists {
        return json_error(StatusCode::NOT_FOUND, "proxy not found");
    }

    clear_pool_runtime(&state, id).await;
    json_ok()
}

fn parse_bearer_auth(headers: &HeaderMap) -> Option<String> {
    headers
        .get(header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .map(ToOwned::to_owned)
}

fn access_key_proxy_id(access_key: &AccessKeyConfig, is_image_request: bool) -> i64 {
    if is_image_request {
        access_key.image_proxy_id.unwrap_or(access_key.proxy_id)
    } else {
        access_key.proxy_id
    }
}

async fn proxy_for_access_key(
    state: &AppState,
    headers: &HeaderMap,
    is_image_request: bool,
) -> Result<(PoolConfig, AccessKeyConfig), Response<Body>> {
    let Some(access_key_value) = parse_bearer_auth(headers) else {
        return Err(json_error(
            StatusCode::UNAUTHORIZED,
            "Authorization: Bearer me-... is required",
        ));
    };

    let access_key = {
        let access_keys = state.access_keys.read().await;
        access_keys
            .iter()
            .find(|item| item.access_key == access_key_value)
            .cloned()
    }
    .ok_or_else(|| json_error(StatusCode::UNAUTHORIZED, "Invalid access key"))?;

    let pool = {
        let pools = state.pools.read().await;
        let proxy_id = access_key_proxy_id(&access_key, is_image_request);
        pools.iter().find(|pool| pool.id == proxy_id).cloned()
    }
    .ok_or_else(|| {
        json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            if is_image_request {
                "Bound image proxy not found for this access key"
            } else {
                "Bound proxy not found for this access key"
            },
        )
    })?;

    Ok((pool, access_key))
}

fn build_override_models_response(model: &str) -> Response<Body> {
    let response = serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "proxy"
            }
        ]
    });
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        response.to_string(),
    )
        .into_response()
}

async fn proxy_openai(State(state): State<AppState>, req: Request) -> Response<Body> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let is_image_request = method == Method::POST && path == IMAGES_GENERATIONS_PATH;
    let (pool, access_key) =
        match proxy_for_access_key(&state, req.headers(), is_image_request).await {
            Ok(result) => result,
            Err(response) => return response,
        };
    let inbound_endpoint = if method == Method::POST {
        OpenAiEndpoint::from_path(&path)
    } else {
        None
    };
    let query = req.uri().query().map(ToOwned::to_owned);
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
    let mut last_error: Option<ProxyAttemptError> = None;
    let allow_auto_ban = if is_image_request {
        !image_request_has_single_candidate(&pool)
    } else {
        !pool_has_single_supplier_and_key(&pool)
    };
    let request_body_preview = if path == RESPONSES_PATH {
        Some(
            String::from_utf8_lossy(&buffered_body)
                .chars()
                .take(800)
                .collect::<String>(),
        )
    } else {
        None
    };

    for base_url in pool.base_urls {
        let (upstream_path, conversion) = match inbound_endpoint {
            Some(endpoint) => {
                let plan = upstream_plan(endpoint, &base_url.protocol_mode);
                (plan.upstream_endpoint.path().to_string(), plan.conversion)
            }
            None => (path.clone(), ConversionMode::Direct),
        };
        let transformed_request = match transform_request_body(&buffered_body, conversion) {
            Ok(result) => result,
            Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
        };
        let mut bridge_context = transformed_request.bridge_context.clone();
        let upstream_body = transformed_request.body;
        let effective_override_model = (!is_image_request)
            .then(|| effective_override_model(&access_key, &base_url).map(ToOwned::to_owned))
            .flatten();
        let upstream_body = if let Some(model) = effective_override_model.as_deref() {
            match override_model_in_body(&upstream_body, model) {
                Ok(body) => body,
                Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
            }
        } else {
            upstream_body
        };
        let (keys, using_image_keys) = if is_image_request && !base_url.image_keys.is_empty() {
            (base_url.image_keys, true)
        } else {
            (base_url.api_keys, false)
        };
        for key in keys {
            loop {
                let schedulable = if using_image_keys {
                    image_key_is_schedulable(&state, &key).await
                } else {
                    key_is_schedulable(&state, &key).await
                };
                if !schedulable {
                    break;
                }

                let upstream_url = match build_upstream_url(
                    &base_url.base_url,
                    &upstream_path,
                    query.as_deref(),
                ) {
                    Ok(url) => url,
                    Err(err) => return (StatusCode::BAD_GATEWAY, err).into_response(),
                };
                let upstream_url_text = upstream_url.to_string();
                let authority = match upstream_authority(&upstream_url) {
                    Ok(authority) => authority,
                    Err(err) => return (StatusCode::BAD_GATEWAY, err).into_response(),
                };
                let headers = match build_upstream_headers(&parts.headers, &authority, &key.api_key)
                {
                    Ok(headers) => headers,
                    Err(err) => return (StatusCode::BAD_GATEWAY, err).into_response(),
                };

                info!(
                    "{} {} -> access_key={} proxy={} supplier_id={} key_id={} key_scope={} {}",
                    parts.method,
                    path,
                    access_key.id,
                    pool.id,
                    base_url.id,
                    key.id,
                    if using_image_keys { "image" } else { "chat" },
                    upstream_path
                );
                if path == RESPONSES_PATH {
                    info!(
                        "responses debug -> upstream_url={} protocol_mode={} conversion={:?} stream={} body={}",
                        upstream_url_text,
                        base_url.protocol_mode.as_str(),
                        conversion,
                        is_streaming,
                        request_body_preview.as_deref().unwrap_or("")
                    );
                }

                let upstream = state
                    .client
                    .request(parts.method.clone(), upstream_url)
                    .headers(headers)
                    .body(upstream_body.clone())
                    .send()
                    .await;

                match upstream {
                    Ok(response)
                        if response.status().is_client_error()
                            || response.status().is_server_error() =>
                    {
                        let status = response.status();
                        if path == RESPONSES_PATH {
                            info!(
                                "responses debug -> upstream_error_status={} supplier_id={} key_id={} url={}",
                                status.as_u16(),
                                base_url.id,
                                key.id,
                                upstream_url_text
                            );
                        }
                        if using_image_keys {
                            mark_image_key_fail(&state, key.id, allow_auto_ban).await;
                        } else {
                            mark_key_fail(&state, key.id, allow_auto_ban).await;
                        }

                        if let Some(model) = effective_override_model
                            .as_deref()
                            .filter(|_| path == MODELS_PATH)
                        {
                            info!(
                                "models fallback -> using override_model={} for supplier_id={}",
                                model, base_url.id
                            );
                            return build_override_models_response(model);
                        }

                        last_error = Some(ProxyAttemptError {
                            body: format!("Upstream returned {}", status.as_u16()),
                        });

                        if !allow_auto_ban {
                            break;
                        }
                        let schedulable = if using_image_keys {
                            image_key_is_schedulable(&state, &key).await
                        } else {
                            key_is_schedulable(&state, &key).await
                        };
                        if schedulable {
                            tokio::time::sleep(Duration::from_secs(1)).await;
                            continue;
                        }
                        break;
                    }
                    Ok(response) => {
                        if path == RESPONSES_PATH {
                            info!(
                                "responses debug -> upstream_success_status={} supplier_id={} key_id={} url={}",
                                response.status().as_u16(),
                                base_url.id,
                                key.id,
                                upstream_url_text
                            );
                        }
                        if using_image_keys {
                            mark_image_key_success(&state, key.id).await;
                        } else {
                            mark_key_success(&state, key.id).await;
                            record_key_usage(&state, access_key.id, pool.id, base_url.id, key.id)
                                .await;
                        }
                        let mut downstream = Response::builder().status(response.status());
                        if let Some(headers_mut) = downstream.headers_mut() {
                            match response_headers(
                                response.headers(),
                                conversion != ConversionMode::Direct,
                                if conversion != ConversionMode::Direct {
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

                        if conversion != ConversionMode::Direct && is_streaming {
                            return downstream
                                .body(transformed_sse_body(
                                    response,
                                    conversion,
                                    std::mem::take(&mut bridge_context),
                                ))
                                .unwrap_or_else(|err| {
                                    error!("Failed to build transformed stream response: {err}");
                                    StatusCode::INTERNAL_SERVER_ERROR.into_response()
                                });
                        }

                        if conversion != ConversionMode::Direct {
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
                            let body =
                                match transform_response_body(&body, conversion, &bridge_context) {
                                    Ok(body) => body,
                                    Err(err) => {
                                        return (StatusCode::BAD_GATEWAY, err).into_response()
                                    }
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
                        if path == RESPONSES_PATH {
                            info!(
                                "responses debug -> upstream_timeout supplier_id={} key_id={} url={}",
                                base_url.id,
                                key.id,
                                upstream_url_text
                            );
                        }
                        if using_image_keys {
                            mark_image_key_fail(&state, key.id, allow_auto_ban).await;
                        } else {
                            mark_key_fail(&state, key.id, allow_auto_ban).await;
                        }

                        if let Some(model) = effective_override_model
                            .as_deref()
                            .filter(|_| path == MODELS_PATH)
                        {
                            info!(
                                "models fallback -> using override_model={} for supplier_id={} (timeout)",
                                model, base_url.id
                            );
                            return build_override_models_response(model);
                        }

                        last_error = Some(ProxyAttemptError {
                            body: "Gateway Timeout".to_string(),
                        });

                        if !allow_auto_ban {
                            break;
                        }
                        let schedulable = if using_image_keys {
                            image_key_is_schedulable(&state, &key).await
                        } else {
                            key_is_schedulable(&state, &key).await
                        };
                        if schedulable {
                            tokio::time::sleep(Duration::from_secs(1)).await;
                            continue;
                        }
                        break;
                    }
                    Err(err) => {
                        if path == RESPONSES_PATH {
                            info!(
                                "responses debug -> upstream_request_error supplier_id={} key_id={} url={} error={}",
                                base_url.id,
                                key.id,
                                upstream_url_text,
                                err
                            );
                        }
                        if using_image_keys {
                            mark_image_key_fail(&state, key.id, allow_auto_ban).await;
                        } else {
                            mark_key_fail(&state, key.id, allow_auto_ban).await;
                        }

                        if let Some(model) = effective_override_model
                            .as_deref()
                            .filter(|_| path == MODELS_PATH)
                        {
                            info!(
                                "models fallback -> using override_model={} for supplier_id={} (error)",
                                model, base_url.id
                            );
                            return build_override_models_response(model);
                        }

                        last_error = Some(ProxyAttemptError {
                            body: format!("Bad Gateway: {err}"),
                        });

                        if !allow_auto_ban {
                            break;
                        }
                        let schedulable = if using_image_keys {
                            image_key_is_schedulable(&state, &key).await
                        } else {
                            key_is_schedulable(&state, &key).await
                        };
                        if schedulable {
                            tokio::time::sleep(Duration::from_secs(1)).await;
                            continue;
                        }
                        break;
                    }
                }
            }
        }
    }

    let message = last_error
        .map(|err| {
            format!(
                "All supplier / API Key candidates are unavailable. {}",
                err.body
            )
        })
        .unwrap_or_else(|| "All supplier / API Key candidates are unavailable".to_string());

    (StatusCode::SERVICE_UNAVAILABLE, message).into_response()
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
    let pools = load_pools(&db_path).expect("Failed to load pools");
    let access_keys = load_access_keys(&db_path).expect("Failed to load access keys");
    let state = AppState {
        pools: Arc::new(RwLock::new(pools)),
        access_keys: Arc::new(RwLock::new(access_keys)),
        runtime: Arc::new(RwLock::new(KeyRuntimeState::default())),
        usage: Arc::new(RwLock::new(UsageRuntimeState::default())),
        client: build_client(),
        db_path,
        admin,
    };

    let admin_routes = Router::new()
        .route("/admin", get(admin_page))
        .route("/admin/keys", get(access_keys_page))
        .route("/admin/api/proxies", get(get_proxies).post(create_pool))
        .route("/admin/api/proxies/save", post(create_proxy_save))
        .route("/admin/api/proxies/reorder", post(reorder_pools))
        .route(
            "/admin/api/access-keys",
            get(get_access_keys).post(create_access_key),
        )
        .route(
            "/admin/api/access-keys/{id}",
            put(update_access_key).delete(delete_access_key),
        )
        .route(
            "/admin/api/proxies/{id}",
            put(update_pool).delete(delete_pool),
        )
        .route("/admin/api/proxies/{id}/save", put(update_proxy_save))
        .route("/admin/api/proxies/{id}/activate", post(activate_pool))
        .route("/admin/api/proxies/{id}/suppliers", post(create_base_url))
        .route("/admin/api/proxies/{id}/unban-all", post(unban_all_pool))
        .route(
            "/admin/api/suppliers/{id}",
            put(update_base_url).delete(delete_base_url),
        )
        .route("/admin/api/suppliers/{id}/models", get(supplier_models))
        .route("/admin/api/suppliers/{id}/keys", post(create_key))
        .route("/admin/api/pools", get(get_proxies).post(create_pool))
        .route(
            "/admin/api/pools/{id}",
            put(update_pool).delete(delete_pool),
        )
        .route("/admin/api/pools/{id}/activate", post(activate_pool))
        .route("/admin/api/pools/{id}/base-urls", post(create_base_url))
        .route("/admin/api/pools/{id}/unban-all", post(unban_all_pool))
        .route(
            "/admin/api/base-urls/{id}",
            put(update_base_url).delete(delete_base_url),
        )
        .route("/admin/api/base-urls/{id}/keys", post(create_key))
        .route("/admin/api/keys/{id}", put(update_key).delete(delete_key))
        .route("/admin/api/keys/{id}/disable", post(disable_key))
        .route("/admin/api/keys/{id}/enable", post(enable_key))
        .route("/admin/api/keys/{id}/models", get(key_models))
        .route("/admin/api/keys/{id}/test", post(test_key))
        .route("/admin/api/keys/{id}/unban", post(unban_key))
        .route(
            "/admin/api/image-keys/{id}/disable",
            post(disable_image_key),
        )
        .route("/admin/api/image-keys/{id}/enable", post(enable_image_key))
        .route("/admin/api/image-keys/{id}/unban", post(unban_image_key))
        .layer(middleware::from_fn_with_state(state.clone(), admin_auth));

    let app = Router::new()
        .merge(admin_routes)
        .route("/v1/{*path}", any(proxy_openai))
        .route("/", get(|| async { "me-api-proxy" }))
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

    fn fixed_now() -> DateTime<Local> {
        Local
            .with_ymd_and_hms(2026, 1, 10, 12, 0, 0)
            .single()
            .unwrap()
    }

    fn temp_db_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "me-api-proxy-{name}-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn test_app_state(path: &PathBuf) -> AppState {
        AppState {
            pools: Arc::new(RwLock::new(load_pools(path).unwrap())),
            access_keys: Arc::new(RwLock::new(load_access_keys(path).unwrap())),
            runtime: Arc::new(RwLock::new(KeyRuntimeState::default())),
            usage: Arc::new(RwLock::new(UsageRuntimeState::default())),
            client: reqwest::Client::new(),
            db_path: path.clone(),
            admin: AdminCredentials {
                username: "admin".to_string(),
                password: "password".to_string(),
            },
        }
    }

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
    fn build_upstream_url_deduplicates_v1() {
        let root = build_upstream_url("https://api.example.com/v1", RESPONSES_PATH, None).unwrap();
        let gateway =
            build_upstream_url("https://gateway.example.com/openai/v1", CHAT_PATH, None).unwrap();

        assert_eq!(root.as_str(), "https://api.example.com/v1/responses");
        assert_eq!(
            gateway.as_str(),
            "https://gateway.example.com/openai/v1/chat/completions"
        );
    }

    #[test]
    fn build_upstream_url_preserves_query_and_prefix() {
        let url = build_upstream_url(
            "https://gateway.example.com/openai",
            RESPONSES_PATH,
            Some("stream=true"),
        )
        .unwrap();
        assert_eq!(
            url.as_str(),
            "https://gateway.example.com/openai/v1/responses?stream=true"
        );
    }

    #[test]
    fn drops_billing_summary_sse_event_but_keeps_chat_events() {
        let mut completed_sent = false;
        let context = BridgeContext::default();
        let billing_event = "data: {\"object\":\"billing.summary\",\"billing\":{}}";
        assert_eq!(
            transform_sse_event(
                billing_event,
                ConversionMode::Direct,
                &context,
                &mut completed_sent,
            ),
            None
        );

        let chat_event = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[]}";
        assert_eq!(
            transform_sse_event(
                chat_event,
                ConversionMode::Direct,
                &context,
                &mut completed_sent,
            ),
            Some(format!("{chat_event}\n\n"))
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

        let converted_request = responses_request_to_chat(body).unwrap();
        let converted: Value = serde_json::from_slice(&converted_request.body).unwrap();

        assert_eq!(converted["model"], "gpt-test");
        assert_eq!(converted["messages"][0]["role"], "system");
        assert_eq!(converted["messages"][1]["role"], "user");
        assert_eq!(converted["messages"][1]["content"], "hello");
        assert_eq!(converted["max_tokens"], 64);
    }

    #[test]
    fn converts_responses_custom_tool_to_chat_function_tool() {
        let body = br#"{
            "model":"gpt-test",
            "input":"hello",
            "tools":[{"type":"custom","name":"run_shell","description":"Run shell"}]
        }"#;

        let converted_request = responses_request_to_chat(body).unwrap();
        let converted: Value = serde_json::from_slice(&converted_request.body).unwrap();

        assert_eq!(converted["tools"][0]["type"], "function");
        assert_eq!(converted["tools"][0]["function"]["name"], "run_shell");
        assert_eq!(
            converted["tools"][0]["function"]["parameters"]["required"][0],
            "input"
        );
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
            serde_json::from_slice(&chat_response_to_responses(body, None).unwrap()).unwrap();

        assert_eq!(converted["object"], "response");
        assert_eq!(converted["output_text"], "hello");
        assert_eq!(converted["usage"]["input_tokens"], 2);
    }

    #[test]
    fn converts_chat_tool_call_back_to_custom_tool_call() {
        let body = br#"{
            "id":"chatcmpl_1",
            "created":123,
            "model":"gpt-test",
            "choices":[{
                "message":{
                    "role":"assistant",
                    "content":"",
                    "tool_calls":[{
                        "id":"call_1",
                        "type":"function",
                        "function":{
                            "name":"run_shell",
                            "arguments":"{\"input\":\"ls\"}"
                        }
                    }]
                }
            }]
        }"#;
        let context = build_responses_tool_context(Some(&json!([
            {"type":"custom","name":"run_shell","description":"Run shell"}
        ])));

        let converted: Value =
            serde_json::from_slice(&chat_response_to_responses(body, Some(&context)).unwrap())
                .unwrap();

        assert_eq!(converted["output"][0]["type"], "custom_tool_call");
        assert_eq!(converted["output"][0]["name"], "run_shell");
        assert_eq!(converted["output"][0]["input"], "ls");
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
    fn key_failures_escalate_from_temp_ban_to_daily_ban() {
        let key = ApiKeyConfig {
            id: 1,
            base_url_id: 1,
            api_key: "sk-test".to_string(),
            sort_order: 0,
            manually_disabled: false,
            test_model: String::new(),
            test_protocol: None,
        };
        let now = fixed_now();
        let mut state = KeyRuntimeState::default();
        state.last_reset_date = now.date_naive();

        mark_key_fail_in_state(key.id, &mut state, now, true);
        mark_key_fail_in_state(key.id, &mut state, now, true);
        let availability = key_availability_in_state(&key, &mut state, now);
        assert_eq!(availability.fail_count, 2);
        assert!(availability.schedulable);

        mark_key_fail_in_state(key.id, &mut state, now, true);
        let availability = key_availability_in_state(&key, &mut state, now);
        assert_eq!(availability.fail_count, 3);
        assert!(availability.banned);

        let after_temp_ban = now + ChronoDuration::minutes(6);
        let availability = key_availability_in_state(&key, &mut state, after_temp_ban);
        assert_eq!(availability.fail_count, 3);
        assert!(!availability.banned);

        mark_key_fail_in_state(key.id, &mut state, after_temp_ban, true);
        let after_second_temp_ban = after_temp_ban + ChronoDuration::minutes(6);
        mark_key_fail_in_state(key.id, &mut state, after_second_temp_ban, true);
        let availability = key_availability_in_state(&key, &mut state, after_second_temp_ban);
        assert_eq!(availability.fail_count, 5);
        assert!(availability.banned);
    }

    #[test]
    fn single_supplier_single_key_failures_do_not_ban() {
        let key = ApiKeyConfig {
            id: 1,
            base_url_id: 1,
            api_key: "sk-test".to_string(),
            sort_order: 0,
            manually_disabled: false,
            test_model: String::new(),
            test_protocol: None,
        };
        let now = fixed_now();
        let mut state = KeyRuntimeState::default();
        state.last_reset_date = now.date_naive();

        for _ in 0..DAILY_BAN_FAILS + 2 {
            mark_key_fail_in_state(key.id, &mut state, now, false);
        }

        let availability = key_availability_in_state(&key, &mut state, now);
        assert_eq!(availability.fail_count, DAILY_BAN_FAILS + 2);
        assert!(!availability.banned);
        assert!(availability.schedulable);
    }

    #[test]
    fn key_success_clears_runtime_state() {
        let now = fixed_now();
        let mut state = KeyRuntimeState::default();
        mark_key_fail_in_state(1, &mut state, now, true);
        mark_key_fail_in_state(1, &mut state, now, true);
        assert!(state.keys.contains_key(&1));

        mark_key_success_in_state(1, &mut state, now);
        assert!(!state.keys.contains_key(&1));
    }

    #[test]
    fn manually_disabled_key_is_not_schedulable() {
        let key = ApiKeyConfig {
            id: 1,
            base_url_id: 1,
            api_key: "sk-test".to_string(),
            sort_order: 0,
            manually_disabled: true,
            test_model: String::new(),
            test_protocol: None,
        };
        let mut state = KeyRuntimeState::default();
        let availability = key_availability_in_state(&key, &mut state, fixed_now());
        assert!(!availability.schedulable);
    }

    #[test]
    fn image_key_runtime_is_isolated_from_chat_key_runtime() {
        let key = ApiKeyConfig {
            id: 1,
            base_url_id: 1,
            api_key: "sk-test".to_string(),
            sort_order: 0,
            manually_disabled: false,
            test_model: String::new(),
            test_protocol: None,
        };
        let now = fixed_now();
        let mut state = KeyRuntimeState::default();
        state.last_reset_date = now.date_naive();

        for _ in 0..TEMP_BAN_FAILS {
            mark_key_fail_in_state(key.id, &mut state, now, true);
        }

        assert!(key_availability_in_state(&key, &mut state, now).banned);
        assert!(image_key_availability_in_state(&key, &mut state, now).schedulable);

        mark_image_key_fail_in_state(key.id, &mut state, now, false);
        assert_eq!(
            image_key_availability_in_state(&key, &mut state, now).fail_count,
            1
        );
        assert_eq!(
            key_availability_in_state(&key, &mut state, now).fail_count,
            3
        );
    }

    #[test]
    fn image_request_uses_only_image_candidates_when_present() {
        let make_key = |id| ApiKeyConfig {
            id,
            base_url_id: 1,
            api_key: format!("sk-{id}"),
            sort_order: id,
            manually_disabled: false,
            test_model: String::new(),
            test_protocol: None,
        };
        let mut pool = PoolConfig {
            id: 1,
            name: "测试代理".to_string(),
            note: String::new(),
            is_active: true,
            base_urls: vec![BaseUrlConfig {
                id: 1,
                pool_id: 1,
                name: "供应商".to_string(),
                base_url: "https://api.example.com/v1".to_string(),
                protocol_mode: ProtocolMode::Both,
                override_model: String::new(),
                sort_order: 0,
                api_keys: vec![make_key(1), make_key(2)],
                image_keys: vec![make_key(3)],
            }],
        };

        assert!(image_request_has_single_candidate(&pool));

        pool.base_urls[0].image_keys.clear();
        assert!(!image_request_has_single_candidate(&pool));
    }

    #[test]
    fn image_keys_are_persisted_and_loaded_independently() {
        let path = temp_db_path("image-keys");
        init_database(&path).unwrap();

        {
            let conn = open_db(&path).unwrap();
            conn.execute(
                "INSERT INTO pools (name, note, is_active, created_at, updated_at)
                 VALUES ('测试代理', '', 1, strftime('%s', 'now'), strftime('%s', 'now'))",
                [],
            )
            .unwrap();
            let pool_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pool_base_urls (pool_id, name, base_url, protocol_mode, sort_order, created_at, updated_at)
                 VALUES (?1, '供应商 A', 'https://api.example.com/v1', 'both', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![pool_id],
            )
            .unwrap();
            let supplier_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pool_api_keys (base_url_id, api_key, sort_order, manually_disabled, created_at, updated_at)
                 VALUES (?1, 'sk-chat', 0, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![supplier_id],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO pool_image_keys (base_url_id, api_key, sort_order, manually_disabled, created_at, updated_at)
                 VALUES (?1, 'sk-image', 0, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![supplier_id],
            )
            .unwrap();
        }

        let pools = load_pools(&path).unwrap();
        let supplier = &pools[0].base_urls[0];
        assert_eq!(supplier.api_keys[0].api_key, "sk-chat");
        assert_eq!(supplier.image_keys[0].api_key, "sk-image");

        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn image_generation_uses_image_key_without_overriding_image_model() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (request_tx, mut request_rx) = tokio::sync::mpsc::unbounded_channel();
        let upstream = Router::new().route(
            IMAGES_GENERATIONS_PATH,
            post(move |headers: HeaderMap, body: Bytes| {
                let request_tx = request_tx.clone();
                async move {
                    let authorization = headers
                        .get(header::AUTHORIZATION)
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    request_tx.send((authorization, body)).unwrap();
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/json")],
                        r#"{"created":0,"data":[]}"#,
                    )
                }
            }),
        );
        let upstream_task = tokio::spawn(async move {
            axum::serve(listener, upstream).await.unwrap();
        });

        let path = temp_db_path("image-generation-route");
        init_database(&path).unwrap();
        let access_key = "me-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd";
        {
            let conn = open_db(&path).unwrap();
            conn.execute(
                "INSERT INTO pools (name, note, is_active, created_at, updated_at)
                 VALUES ('测试代理', '', 1, strftime('%s', 'now'), strftime('%s', 'now'))",
                [],
            )
            .unwrap();
            let pool_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pool_base_urls (pool_id, name, base_url, protocol_mode, override_model, sort_order, created_at, updated_at)
                 VALUES (?1, '供应商 A', ?2, 'both', 'gpt-chat-only', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![pool_id, format!("http://{address}")],
            )
            .unwrap();
            let supplier_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pool_api_keys (base_url_id, api_key, sort_order, manually_disabled, created_at, updated_at)
                 VALUES (?1, 'sk-chat', 0, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![supplier_id],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO pool_image_keys (base_url_id, api_key, sort_order, manually_disabled, created_at, updated_at)
                 VALUES (?1, 'sk-image', 0, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![supplier_id],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO access_keys (name, access_key, proxy_id, created_at, updated_at)
                 VALUES ('客户端 A', ?1, ?2, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![access_key, pool_id],
            )
            .unwrap();
        }

        let state = test_app_state(&path);
        let request = Request::builder()
            .method(Method::POST)
            .uri(IMAGES_GENERATIONS_PATH)
            .header(header::AUTHORIZATION, format!("Bearer {access_key}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"model":"gpt-image-1","prompt":"a cat"}"#))
            .unwrap();
        let response = proxy_openai(State(state), request).await;

        assert_eq!(response.status(), StatusCode::OK);
        let (authorization, body) = request_rx.recv().await.unwrap();
        assert_eq!(authorization, "Bearer sk-image");
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["model"], "gpt-image-1");

        upstream_task.abort();
        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn image_generation_uses_access_key_image_proxy() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (request_tx, mut request_rx) = tokio::sync::mpsc::unbounded_channel();
        let upstream = Router::new().route(
            IMAGES_GENERATIONS_PATH,
            post(move |headers: HeaderMap| {
                let request_tx = request_tx.clone();
                async move {
                    let authorization = headers
                        .get(header::AUTHORIZATION)
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    request_tx.send(authorization).unwrap();
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/json")],
                        r#"{"created":0,"data":[]}"#,
                    )
                }
            }),
        );
        let upstream_task = tokio::spawn(async move {
            axum::serve(listener, upstream).await.unwrap();
        });

        let path = temp_db_path("image-access-key-proxy");
        init_database(&path).unwrap();
        let access_key = "me-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd";
        {
            let conn = open_db(&path).unwrap();
            conn.execute(
                "INSERT INTO pools (name, note, is_active, created_at, updated_at)
                 VALUES ('聊天代理', '', 1, strftime('%s', 'now'), strftime('%s', 'now'))",
                [],
            )
            .unwrap();
            let chat_pool_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pools (name, note, is_active, created_at, updated_at)
                 VALUES ('生图代理', '', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                [],
            )
            .unwrap();
            let image_pool_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pool_base_urls (pool_id, name, base_url, protocol_mode, sort_order, created_at, updated_at)
                 VALUES (?1, '生图供应商', ?2, 'both', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![image_pool_id, format!("http://{address}")],
            )
            .unwrap();
            let supplier_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pool_image_keys (base_url_id, api_key, sort_order, manually_disabled, created_at, updated_at)
                 VALUES (?1, 'sk-image-bound', 0, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![supplier_id],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO access_keys (name, access_key, proxy_id, image_proxy_id, created_at, updated_at)
                 VALUES ('客户端 A', ?1, ?2, ?3, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![access_key, chat_pool_id, image_pool_id],
            )
            .unwrap();
        }

        let state = test_app_state(&path);
        let request = Request::builder()
            .method(Method::POST)
            .uri(IMAGES_GENERATIONS_PATH)
            .header(header::AUTHORIZATION, format!("Bearer {access_key}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"model":"gpt-image-1","prompt":"a cat"}"#))
            .unwrap();
        let response = proxy_openai(State(state), request).await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(request_rx.recv().await.unwrap(), "Bearer sk-image-bound");

        upstream_task.abort();
        let _ = fs::remove_file(path);
    }

    #[test]
    fn generated_access_key_uses_me_prefix() {
        let access_key = generate_access_key_value();
        assert!(access_key.starts_with("me-"));
        assert_eq!(access_key.len(), 67);
        assert!(access_key[3..].chars().all(|ch| ch.is_ascii_hexdigit()));
    }

    fn test_access_key_with_override(model: &str) -> AccessKeyConfig {
        AccessKeyConfig {
            id: 1,
            name: "客户端 A".to_string(),
            access_key: "me-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd"
                .to_string(),
            proxy_id: 1,
            image_proxy_id: None,
            override_model: model.to_string(),
            created_at: 0,
            updated_at: 0,
        }
    }

    fn test_base_url_with_override(model: &str) -> BaseUrlConfig {
        BaseUrlConfig {
            id: 1,
            pool_id: 1,
            name: "供应商 A".to_string(),
            base_url: "https://api.example.com/v1".to_string(),
            protocol_mode: ProtocolMode::Both,
            override_model: model.to_string(),
            sort_order: 0,
            api_keys: Vec::new(),
            image_keys: Vec::new(),
        }
    }

    #[test]
    fn access_key_override_model_takes_priority_over_supplier_model() {
        let access_key = test_access_key_with_override("gpt-5.4");
        let supplier = test_base_url_with_override("gpt-5.5");

        assert_eq!(
            effective_override_model(&access_key, &supplier),
            Some("gpt-5.4")
        );
    }

    #[test]
    fn supplier_override_model_applies_when_access_key_model_is_empty() {
        let access_key = test_access_key_with_override("");
        let supplier = test_base_url_with_override("gpt-5.5");
        let body = br#"{"model":"gpt-original","messages":[]}"#;

        let model = effective_override_model(&access_key, &supplier).unwrap();
        let converted: Value =
            serde_json::from_slice(&override_model_in_body(body, model).unwrap()).unwrap();

        assert_eq!(model, "gpt-5.5");
        assert_eq!(converted["model"], "gpt-5.5");
    }

    #[tokio::test]
    async fn override_models_response_uses_effective_access_key_model() {
        let access_key = test_access_key_with_override("gpt-5.4");
        let supplier = test_base_url_with_override("gpt-5.5");
        let model = effective_override_model(&access_key, &supplier).unwrap();
        let response = build_override_models_response(model);
        let body = to_bytes(response.into_body(), MAX_REQUEST_BODY_BYTES)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(value["data"][0]["id"], "gpt-5.4");
    }

    #[test]
    fn init_database_supports_loading_access_keys() {
        let path = temp_db_path("access-keys");
        init_database(&path).unwrap();

        {
            let conn = open_db(&path).unwrap();
            conn.execute(
                "INSERT INTO pools (name, note, is_active, created_at, updated_at)
                 VALUES ('测试代理', '', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                [],
            )
            .unwrap();
            let pool_id = conn.last_insert_rowid();

            conn.execute(
                "INSERT INTO access_keys (name, access_key, proxy_id, created_at, updated_at)
                 VALUES (?1, ?2, ?3, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![
                    "客户端 A",
                    "me-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd",
                    pool_id
                ],
            )
            .unwrap();
        }

        let access_keys = load_access_keys(&path).unwrap();
        assert_eq!(access_keys.len(), 1);
        assert_eq!(access_keys[0].name, "客户端 A");
        assert_eq!(access_keys[0].proxy_id, 1);
        assert_eq!(access_keys[0].override_model, "");
        assert!(access_keys[0].access_key.starts_with("me-"));

        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn create_and_update_access_key_save_override_model() {
        let path = temp_db_path("access-key-override-save");
        init_database(&path).unwrap();

        let pool_id = {
            let conn = open_db(&path).unwrap();
            conn.execute(
                "INSERT INTO pools (name, note, is_active, created_at, updated_at)
                 VALUES ('测试代理', '', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                [],
            )
            .unwrap();
            conn.last_insert_rowid()
        };

        let state = test_app_state(&path);
        let response = create_access_key(
            State(state.clone()),
            Json(AccessKeyPayload {
                name: "客户端 A".to_string(),
                proxy_id: pool_id,
                image_proxy_id: None,
                override_model: "gpt-5.4".to_string(),
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);

        let access_key_id: i64 = open_db(&path)
            .unwrap()
            .query_row(
                "SELECT id FROM access_keys WHERE name = '客户端 A'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        let response = update_access_key(
            State(state.clone()),
            Path(access_key_id),
            Json(AccessKeyPayload {
                name: "客户端 B".to_string(),
                proxy_id: pool_id,
                image_proxy_id: None,
                override_model: "gpt-5.5".to_string(),
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);

        let access_keys = access_keys_response(&state).await;
        assert_eq!(access_keys.access_keys[0].name, "客户端 B");
        assert_eq!(access_keys.access_keys[0].override_model, "gpt-5.5");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn init_database_adds_key_test_columns_and_loads_saved_values() {
        let path = temp_db_path("key-test-columns");

        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "
                CREATE TABLE pools (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    is_active INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
                    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
                );
                CREATE TABLE pool_base_urls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pool_id INTEGER NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    base_url TEXT NOT NULL,
                    protocol_mode TEXT NOT NULL DEFAULT 'both',
                    override_model TEXT NOT NULL DEFAULT '',
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
                    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
                );
                CREATE TABLE pool_api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    base_url_id INTEGER NOT NULL,
                    api_key TEXT NOT NULL,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    manually_disabled INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
                    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
                );
                CREATE TABLE access_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    access_key TEXT NOT NULL UNIQUE,
                    proxy_id INTEGER NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
                    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
                );
                INSERT INTO pools (name, note, is_active) VALUES ('测试代理', '', 0);
                INSERT INTO pool_base_urls (pool_id, name, base_url, protocol_mode, sort_order)
                VALUES (1, '供应商 A', 'https://api.example.com/v1', 'both', 0);
                INSERT INTO pool_api_keys (base_url_id, api_key, sort_order, manually_disabled)
                VALUES (1, 'sk-test-123', 0, 0);
                ",
            )
            .unwrap();
        }

        init_database(&path).unwrap();

        {
            let conn = open_db(&path).unwrap();
            conn.execute(
                "UPDATE pool_api_keys
                 SET test_model = ?1, test_protocol = ?2
                 WHERE id = 1",
                params!["gpt-test", "responses"],
            )
            .unwrap();
        }

        let pools = load_pools(&path).unwrap();
        let key = &pools[0].base_urls[0].api_keys[0];
        assert_eq!(key.test_model, "gpt-test");
        assert_eq!(key.test_protocol, Some(TestProtocol::Responses));

        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn record_key_usage_updates_runtime_state() {
        let path = temp_db_path("record-usage");
        init_database(&path).unwrap();

        let (pool_id, supplier_id, key_id, access_key_id) = {
            let conn = open_db(&path).unwrap();
            conn.execute(
                "INSERT INTO pools (name, note, is_active, created_at, updated_at)
                 VALUES ('测试代理', '', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                [],
            )
            .unwrap();
            let pool_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pool_base_urls (pool_id, name, base_url, protocol_mode, sort_order, created_at, updated_at)
                 VALUES (?1, '供应商 A', 'https://api.example.com/v1', 'both', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![pool_id],
            )
            .unwrap();
            let supplier_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pool_api_keys (base_url_id, api_key, sort_order, manually_disabled, created_at, updated_at)
                 VALUES (?1, 'sk-test-1234567890', 0, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![supplier_id],
            )
            .unwrap();
            let key_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO access_keys (name, access_key, proxy_id, override_model, created_at, updated_at)
                 VALUES ('客户端 A', 'me-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd', ?1, 'gpt-5.4', strftime('%s', 'now'), strftime('%s', 'now'))",
                params![pool_id],
            )
            .unwrap();
            let access_key_id = conn.last_insert_rowid();
            (pool_id, supplier_id, key_id, access_key_id)
        };

        let state = test_app_state(&path);
        record_key_usage(&state, access_key_id, pool_id, supplier_id, key_id).await;

        let usage = state.usage.read().await;
        let access_usage = usage.by_access_key.get(&access_key_id).unwrap();
        assert_eq!(access_usage.proxy_id, pool_id);
        assert_eq!(access_usage.supplier_id, supplier_id);
        assert_eq!(access_usage.key_id, key_id);
        assert!(usage.by_key.contains_key(&key_id));

        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn access_key_and_pool_responses_include_last_used_summary() {
        let path = temp_db_path("usage-response");
        init_database(&path).unwrap();

        let (pool_id, supplier_id, key_id, access_key_id) = {
            let conn = open_db(&path).unwrap();
            conn.execute(
                "INSERT INTO pools (name, note, is_active, created_at, updated_at)
                 VALUES ('测试代理', '', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                [],
            )
            .unwrap();
            let pool_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pool_base_urls (pool_id, name, base_url, protocol_mode, sort_order, created_at, updated_at)
                 VALUES (?1, 'OpenAI', 'https://api.example.com/v1', 'both', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![pool_id],
            )
            .unwrap();
            let supplier_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pool_api_keys (base_url_id, api_key, sort_order, manually_disabled, created_at, updated_at)
                 VALUES (?1, 'sk-test-1234567890', 0, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![supplier_id],
            )
            .unwrap();
            let key_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO access_keys (name, access_key, proxy_id, override_model, created_at, updated_at)
                 VALUES ('客户端 A', 'me-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd', ?1, 'gpt-5.4', strftime('%s', 'now'), strftime('%s', 'now'))",
                params![pool_id],
            )
            .unwrap();
            let access_key_id = conn.last_insert_rowid();
            (pool_id, supplier_id, key_id, access_key_id)
        };

        let state = test_app_state(&path);
        record_key_usage(&state, access_key_id, pool_id, supplier_id, key_id).await;

        let access_keys = access_keys_response(&state).await;
        assert_eq!(access_keys.access_keys.len(), 1);
        assert_eq!(access_keys.access_keys[0].override_model, "gpt-5.4");
        assert_eq!(access_keys.proxies[0].suppliers[0].id, supplier_id);
        assert_eq!(
            access_keys.access_keys[0]
                .last_used_supplier_name
                .as_deref(),
            Some("OpenAI")
        );
        assert_eq!(
            access_keys.access_keys[0].last_used_key_masked.as_deref(),
            Some("sk-tes...7890")
        );
        assert!(access_keys.access_keys[0].last_used_at.is_some());

        let proxies = pools_response(&state).await;
        assert_eq!(proxies.proxies.len(), 1);
        assert_eq!(
            proxies.proxies[0].suppliers[0]
                .last_used_by_access_key_name
                .as_deref(),
            Some("客户端 A")
        );
        assert!(proxies.proxies[0].suppliers[0].keys[0].last_used);

        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn pools_response_includes_saved_test_preferences() {
        let path = temp_db_path("pool-test-preferences");
        init_database(&path).unwrap();

        {
            let conn = open_db(&path).unwrap();
            conn.execute(
                "INSERT INTO pools (name, note, is_active, created_at, updated_at)
                 VALUES ('测试代理', '', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                [],
            )
            .unwrap();
            let pool_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pool_base_urls (pool_id, name, base_url, protocol_mode, sort_order, created_at, updated_at)
                 VALUES (?1, '供应商 A', 'https://api.example.com/v1', 'both', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![pool_id],
            )
            .unwrap();
            let supplier_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pool_api_keys
                 (base_url_id, api_key, sort_order, manually_disabled, test_model, test_protocol, created_at, updated_at)
                 VALUES (?1, 'sk-test-1234567890', 0, 0, 'gpt-4.1-mini', 'chat', strftime('%s', 'now'), strftime('%s', 'now'))",
                params![supplier_id],
            )
            .unwrap();
        }

        let state = test_app_state(&path);
        let proxies = pools_response(&state).await;
        let key = &proxies.proxies[0].suppliers[0].keys[0];
        assert_eq!(key.test_model.as_deref(), Some("gpt-4.1-mini"));
        assert_eq!(key.test_protocol, Some(TestProtocol::Chat));

        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn deleting_image_only_proxy_clears_access_key_image_binding() {
        let path = temp_db_path("delete-image-proxy-binding");
        init_database(&path).unwrap();

        let (chat_pool_id, image_pool_id, access_key_id) = {
            let conn = open_db(&path).unwrap();
            conn.execute(
                "INSERT INTO pools (name, note, is_active, created_at, updated_at)
                 VALUES ('聊天代理', '', 1, strftime('%s', 'now'), strftime('%s', 'now'))",
                [],
            )
            .unwrap();
            let chat_pool_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO pools (name, note, is_active, created_at, updated_at)
                 VALUES ('生图代理', '', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                [],
            )
            .unwrap();
            let image_pool_id = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO access_keys (name, access_key, proxy_id, image_proxy_id, created_at, updated_at)
                 VALUES ('客户端 A', 'me-delete-image-proxy', ?1, ?2, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![chat_pool_id, image_pool_id],
            )
            .unwrap();
            (chat_pool_id, image_pool_id, conn.last_insert_rowid())
        };

        let state = test_app_state(&path);
        let response = delete_pool(State(state.clone()), Path(image_pool_id)).await;
        assert_eq!(response.status(), StatusCode::OK);

        let image_proxy_id: Option<i64> = open_db(&path)
            .unwrap()
            .query_row(
                "SELECT image_proxy_id FROM access_keys WHERE id = ?1",
                params![access_key_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(image_proxy_id, None);
        assert_eq!(state.access_keys.read().await[0].proxy_id, chat_pool_id);
        assert_eq!(state.access_keys.read().await[0].image_proxy_id, None);

        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn delete_pool_cascades_bound_access_keys_and_refreshes_state() {
        let path = temp_db_path("delete-pool-cascade");
        init_database(&path).unwrap();

        let pool_id = {
            let conn = open_db(&path).unwrap();
            conn.execute(
                "INSERT INTO pools (name, note, is_active, created_at, updated_at)
                 VALUES ('测试代理', '备注', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                [],
            )
            .unwrap();
            let pool_id = conn.last_insert_rowid();

            conn.execute(
                "INSERT INTO pool_base_urls (pool_id, name, base_url, protocol_mode, sort_order, created_at, updated_at)
                 VALUES (?1, '供应商 A', 'https://api.example.com/v1', 'both', 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![pool_id],
            )
            .unwrap();
            let base_url_id = conn.last_insert_rowid();

            conn.execute(
                "INSERT INTO pool_api_keys (base_url_id, api_key, sort_order, manually_disabled, created_at, updated_at)
                 VALUES (?1, 'sk-test', 0, 0, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![base_url_id],
            )
            .unwrap();

            conn.execute(
                "INSERT INTO access_keys (name, access_key, proxy_id, created_at, updated_at)
                 VALUES (?1, ?2, ?3, strftime('%s', 'now'), strftime('%s', 'now'))",
                params![
                    "客户端 A",
                    "me-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd",
                    pool_id
                ],
            )
            .unwrap();

            pool_id
        };

        let state = test_app_state(&path);

        let response = delete_pool(State(state.clone()), Path(pool_id)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(std::str::from_utf8(&body).unwrap(), "{\"ok\":true}");

        let conn = open_db(&path).unwrap();
        let pool_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM pools", [], |row| row.get(0))
            .unwrap();
        let access_key_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM access_keys", [], |row| row.get(0))
            .unwrap();
        let supplier_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM pool_base_urls", [], |row| row.get(0))
            .unwrap();
        let api_key_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM pool_api_keys", [], |row| row.get(0))
            .unwrap();

        assert_eq!(pool_count, 0);
        assert_eq!(access_key_count, 0);
        assert_eq!(supplier_count, 0);
        assert_eq!(api_key_count, 0);
        assert!(state.pools.read().await.is_empty());
        assert!(state.access_keys.read().await.is_empty());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn init_database_migrates_legacy_single_config_to_pool_hierarchy() {
        let path = temp_db_path("migrate");

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
                CREATE TABLE openai_api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_key TEXT NOT NULL,
                    sort_order INTEGER NOT NULL DEFAULT 0
                );
                INSERT INTO openai_config (id, base_url, api_key, protocol_mode, enabled)
                VALUES (1, 'https://api.example.com/v1', 'sk-legacy', 'chat', 1);
                INSERT INTO openai_api_keys (api_key, sort_order)
                VALUES ('sk-legacy', 0), ('sk-second', 1);
                ",
            )
            .unwrap();
        }

        init_database(&path).unwrap();
        let pools = load_pools(&path).unwrap();
        assert_eq!(pools.len(), 1);
        assert!(pools[0].is_active);
        assert_eq!(pools[0].base_urls.len(), 1);
        assert_eq!(pools[0].base_urls[0].protocol_mode, ProtocolMode::Chat);
        assert_eq!(pools[0].base_urls[0].base_url, "https://api.example.com/v1");
        assert_eq!(pools[0].base_urls[0].api_keys.len(), 2);

        let _ = fs::remove_file(path);
    }
}
