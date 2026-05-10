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
    sort_order: i64,
    api_keys: Vec<ApiKeyConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct ApiKeyConfig {
    id: i64,
    base_url_id: i64,
    api_key: String,
    sort_order: i64,
    manually_disabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct AccessKeyConfig {
    id: i64,
    name: String,
    access_key: String,
    proxy_id: i64,
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
    keys: Vec<ApiKeySavePayload>,
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
    keys: Vec<PreparedApiKeySave>,
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
    sort_order: i64,
    available_key_count: usize,
    total_key_count: usize,
    schedulable: bool,
    keys: Vec<ApiKeyView>,
}

#[derive(Debug, Serialize)]
struct ApiKeyView {
    id: i64,
    api_key: String,
    masked_key: String,
    sort_order: i64,
    manually_disabled: bool,
    fail_count: u32,
    banned: bool,
    ban_until: Option<i64>,
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
    created_at: i64,
    updated_at: i64,
}

#[derive(Debug, Serialize)]
struct ProxyOptionView {
    id: i64,
    name: String,
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
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
        );

        CREATE TABLE IF NOT EXISTS pool_base_urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pool_id INTEGER NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            base_url TEXT NOT NULL,
            protocol_mode TEXT NOT NULL DEFAULT 'both',
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
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            FOREIGN KEY (base_url_id) REFERENCES pool_base_urls(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS access_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            access_key TEXT NOT NULL UNIQUE,
            proxy_id INTEGER NOT NULL,
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

        let mut seen_api_keys = HashSet::new();
        let mut keys = Vec::with_capacity(supplier.keys.len());
        for key in supplier.keys {
            let api_key = normalize_api_key(&key.api_key)?;
            if !seen_api_keys.insert(api_key.clone()) {
                return Err("api_key already exists in supplier".to_string());
            }
            keys.push(PreparedApiKeySave {
                id: key.id,
                api_key,
            });
        }

        suppliers.push(PreparedSupplierSave {
            id: supplier.id,
            name: supplier.name.trim().to_string(),
            base_url,
            protocol: supplier.protocol,
            keys,
        });
    }

    Ok(PreparedProxySave {
        name,
        note,
        suppliers,
    })
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
             ORDER BY is_active DESC, created_at ASC, id ASC",
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
                "SELECT id, name, base_url, protocol_mode, sort_order
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
                    row.get::<_, i64>(4)?,
                ))
            })
            .map_err(|err| format!("Failed to query base_urls: {err}"))?;

        let mut base_urls = Vec::new();
        for base_row in base_rows {
            let (base_url_id, supplier_name, base_url, protocol_mode, sort_order) =
                base_row.map_err(|err| format!("Failed to decode base_url row: {err}"))?;

            let mut key_stmt = conn
                .prepare(
                    "SELECT id, api_key, sort_order, manually_disabled
                     FROM pool_api_keys
                     WHERE base_url_id = ?1
                     ORDER BY sort_order ASC, id ASC",
                )
                .map_err(|err| format!("Failed to prepare api_key query: {err}"))?;
            let key_rows = key_stmt
                .query_map(params![base_url_id], |row| {
                    Ok(ApiKeyConfig {
                        id: row.get(0)?,
                        base_url_id,
                        api_key: row.get(1)?,
                        sort_order: row.get(2)?,
                        manually_disabled: row.get::<_, i64>(3)? != 0,
                    })
                })
                .map_err(|err| format!("Failed to query api_keys: {err}"))?;

            let mut api_keys = Vec::new();
            for key_row in key_rows {
                api_keys
                    .push(key_row.map_err(|err| format!("Failed to decode api_key row: {err}"))?);
            }

            base_urls.push(BaseUrlConfig {
                id: base_url_id,
                pool_id,
                name: supplier_name,
                base_url,
                protocol_mode: ProtocolMode::from_str(&protocol_mode)?,
                sort_order,
                api_keys,
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
            "SELECT id, name, access_key, proxy_id, created_at, updated_at
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
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
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

    {
        let mut runtime = state.runtime.write().await;
        runtime
            .keys
            .retain(|key_id, _| valid_key_ids.contains(key_id));
    }
    *state.pools.write().await = pools;
    Ok(())
}

async fn reload_access_keys(state: &AppState) -> Result<(), String> {
    let access_keys = load_access_keys(&state.db_path)?;
    *state.access_keys.write().await = access_keys;
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

fn json_key_test_ok(message: impl Into<String>, status_code: Option<u16>, ok: bool) -> Response<Body> {
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

fn mark_key_fail_in_state(key_id: i64, state: &mut KeyRuntimeState, now: DateTime<Local>) {
    ensure_runtime_day(state, now);
    let entry = state.keys.entry(key_id).or_default();
    clear_expired_ban(entry, now);
    entry.fail_count += 1;
    if entry.fail_count >= DAILY_BAN_FAILS {
        entry.ban_until = Some(next_day_zero(now));
    } else if entry.fail_count >= TEMP_BAN_FAILS {
        entry.ban_until = Some(now + ChronoDuration::minutes(TEMP_BAN_MINUTES));
    }
}

fn mark_key_success_in_state(key_id: i64, state: &mut KeyRuntimeState, now: DateTime<Local>) {
    ensure_runtime_day(state, now);
    state.keys.remove(&key_id);
}

async fn key_is_schedulable(state: &AppState, key: &ApiKeyConfig) -> bool {
    let mut runtime = state.runtime.write().await;
    key_availability_in_state(key, &mut runtime, Local::now()).schedulable
}

async fn mark_key_fail(state: &AppState, key_id: i64) {
    let mut runtime = state.runtime.write().await;
    mark_key_fail_in_state(key_id, &mut runtime, Local::now());
}

async fn mark_key_success(state: &AppState, key_id: i64) {
    let mut runtime = state.runtime.write().await;
    mark_key_success_in_state(key_id, &mut runtime, Local::now());
}

async fn clear_key_runtime(state: &AppState, key_id: i64) {
    state.runtime.write().await.keys.remove(&key_id);
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

async fn clear_pool_runtime(state: &AppState, pool_id: i64) {
    let key_ids: HashSet<i64> = state
        .pools
        .read()
        .await
        .iter()
        .find(|pool| pool.id == pool_id)
        .map(|pool| {
            pool.base_urls
                .iter()
                .flat_map(|base_url| base_url.api_keys.iter().map(|key| key.id))
                .collect()
        })
        .unwrap_or_default();

    let mut runtime = state.runtime.write().await;
    for key_id in key_ids {
        runtime.keys.remove(&key_id);
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
    let now = Local::now();
    let mut runtime = state.runtime.write().await;
    ensure_runtime_day(&mut runtime, now);

    let binding_counts = access_keys.into_iter().fold(HashMap::new(), |mut map, access_key| {
        *map.entry(access_key.proxy_id).or_insert(0usize) += 1;
        map
    });
    let proxies = pools
        .into_iter()
        .map(|pool| {
            let access_key_count = binding_counts.get(&pool.id).copied().unwrap_or(0);
            let mut available_supplier_count = 0usize;
            let suppliers = pool
                .base_urls
                .into_iter()
                .map(|base_url| {
                    let mut available_key_count = 0usize;
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
                                fail_count: availability.fail_count,
                                banned: availability.banned,
                                ban_until: availability.ban_until,
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
                        sort_order: base_url.sort_order,
                        available_key_count,
                        total_key_count: keys.len(),
                        schedulable,
                        keys,
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

    let proxy_names: HashMap<i64, String> = pools
        .iter()
        .map(|pool| (pool.id, pool.name.clone()))
        .collect();
    let proxies = pools
        .into_iter()
        .map(|pool| ProxyOptionView {
            id: pool.id,
            name: pool.name,
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
            created_at: access_key.created_at,
            updated_at: access_key.updated_at,
        })
        .collect();

    AccessKeysResponse { access_keys, proxies }
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

    let (saved_pool_id, runtime_ids_to_clear) = {
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

        let save_result: Result<(i64, HashSet<i64>), String> = (|| {
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
                tx.execute(
                    "INSERT INTO pools (name, note, is_active, created_at, updated_at)
                     VALUES (?1, ?2, ?3, strftime('%s', 'now'), strftime('%s', 'now'))",
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
            }

            let mut seen_supplier_ids = HashSet::new();
            let mut runtime_ids_to_clear = HashSet::new();

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
                         SET name = ?1, base_url = ?2, protocol_mode = ?3, sort_order = ?4,
                             updated_at = strftime('%s', 'now')
                         WHERE id = ?5",
                        params![
                            &supplier.name,
                            &supplier.base_url,
                            supplier.protocol.as_str(),
                            supplier_sort_order as i64,
                            supplier_id
                        ],
                    )
                    .map_err(|err| format!("Failed to update supplier: {err}"))?;
                    supplier_id
                } else {
                    tx.execute(
                        "INSERT INTO pool_base_urls
                         (pool_id, name, base_url, protocol_mode, sort_order, created_at, updated_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, strftime('%s', 'now'), strftime('%s', 'now'))",
                        params![
                            saved_pool_id,
                            &supplier.name,
                            &supplier.base_url,
                            supplier.protocol.as_str(),
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
            }

            for supplier_id in existing_supplier_ids {
                if !seen_supplier_ids.contains(&supplier_id) {
                    if let Some(existing_keys) = existing_keys_by_supplier.get(&supplier_id) {
                        for key_id in existing_keys.keys() {
                            runtime_ids_to_clear.insert(*key_id);
                        }
                    }
                    tx.execute("DELETE FROM pool_base_urls WHERE id = ?1", params![supplier_id])
                        .map_err(|err| format!("Failed to delete supplier: {err}"))?;
                }
            }

            Ok((saved_pool_id, runtime_ids_to_clear))
        })();

        let (saved_pool_id, runtime_ids_to_clear) = match save_result {
            Ok(result) => result,
            Err(err) if err == "proxy not found" => {
                return json_error(StatusCode::NOT_FOUND, err);
            }
            Err(err)
                if err == "supplier not found"
                    || err == "api_key not found"
                    || err == "duplicate supplier id in payload"
                    || err == "duplicate api_key id in payload" =>
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

        (saved_pool_id, runtime_ids_to_clear)
    };

    clear_key_runtimes(state, runtime_ids_to_clear).await;
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

    if let Err(err) = conn.execute(
        "INSERT INTO pools (name, note, is_active, created_at, updated_at)
         VALUES (?1, ?2, ?3, strftime('%s', 'now'), strftime('%s', 'now'))",
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

    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    match proxy_exists(&conn, payload.proxy_id) {
        Ok(true) => {}
        Ok(false) => return json_error(StatusCode::BAD_REQUEST, "proxy not found"),
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    }

    let access_key = match generate_unique_access_key(&conn) {
        Ok(access_key) => access_key,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let id = match conn.execute(
        "INSERT INTO access_keys (name, access_key, proxy_id, created_at, updated_at)
         VALUES (?1, ?2, ?3, strftime('%s', 'now'), strftime('%s', 'now'))",
        params![name, access_key, payload.proxy_id],
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

    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    match proxy_exists(&conn, payload.proxy_id) {
        Ok(true) => {}
        Ok(false) => return json_error(StatusCode::BAD_REQUEST, "proxy not found"),
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    }

    match conn.execute(
        "UPDATE access_keys
         SET name = ?1, proxy_id = ?2, updated_at = strftime('%s', 'now')
         WHERE id = ?3",
        params![name, payload.proxy_id, id],
    ) {
        Ok(0) => return json_error(StatusCode::NOT_FOUND, "access key not found"),
        Ok(_) => {}
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to update access key: {err}"),
            )
        }
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

    if let Err(err) = conn.execute("DELETE FROM pools WHERE id = ?1", params![id]) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to delete proxy: {err}"),
        );
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
        Ok(true) => {
            return json_error(StatusCode::BAD_REQUEST, "base_url already exists in proxy")
        }
        Ok(false) => {}
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    }

    let sort_order = payload.sort_order.unwrap_or_else(|| {
        next_sort_order(&conn, "pool_base_urls", "pool_id", pool_id).unwrap_or(0)
    });

    if let Err(err) = conn.execute(
        "INSERT INTO pool_base_urls (pool_id, name, base_url, protocol_mode, sort_order, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, strftime('%s', 'now'), strftime('%s', 'now'))",
        params![
            pool_id,
            supplier_name,
            base_url,
            payload.protocol.as_str(),
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
        Ok(true) => {
            return json_error(StatusCode::BAD_REQUEST, "base_url already exists in proxy")
        }
        Ok(false) => {}
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    }

    let sort_order = payload.sort_order.unwrap_or(current_sort_order);
    if let Err(err) = conn.execute(
        "UPDATE pool_base_urls
         SET name = ?1, base_url = ?2, protocol_mode = ?3, sort_order = ?4, updated_at = strftime('%s', 'now')
         WHERE id = ?5",
        params![
            supplier_name,
            base_url,
            payload.protocol.as_str(),
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

async fn test_key(State(state): State<AppState>, Path(id): Path<i64>) -> Response<Body> {
    let conn = match open_db(&state.db_path) {
        Ok(conn) => conn,
        Err(err) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, err),
    };

    let record: Option<(String, String)> = match conn
        .query_row(
            "SELECT k.api_key, b.base_url
             FROM pool_api_keys k
             JOIN pool_base_urls b ON b.id = k.base_url_id
             WHERE k.id = ?1",
            params![id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
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
    let Some((api_key, base_url)) = record else {
        return json_error(StatusCode::NOT_FOUND, "api_key not found");
    };

    let upstream_url = match build_upstream_url(&base_url, MODELS_PATH, None) {
        Ok(url) => url,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
    };
    let authority = match upstream_authority(&upstream_url) {
        Ok(authority) => authority,
        Err(err) => return json_error(StatusCode::BAD_GATEWAY, err),
    };
    let headers = match build_upstream_headers(&HeaderMap::new(), &authority, &api_key) {
        Ok(headers) => headers,
        Err(err) => return json_error(StatusCode::BAD_GATEWAY, err),
    };

    match state
        .client
        .get(upstream_url)
        .headers(headers)
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

async fn proxy_for_access_key(
    state: &AppState,
    headers: &HeaderMap,
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
        pools.iter().find(|pool| pool.id == access_key.proxy_id).cloned()
    }
    .ok_or_else(|| {
        json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Bound proxy not found for this access key",
        )
    })?;

    Ok((pool, access_key))
}

async fn proxy_openai(State(state): State<AppState>, req: Request) -> Response<Body> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let (pool, access_key) = match proxy_for_access_key(&state, req.headers()).await {
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
    let request_body_preview = if path == RESPONSES_PATH {
        Some(String::from_utf8_lossy(&buffered_body).chars().take(800).collect::<String>())
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
        let upstream_body = match transform_request_body(&buffered_body, conversion) {
            Ok(body) => body,
            Err(err) => return json_error(StatusCode::BAD_REQUEST, err),
        };
        for key in base_url.api_keys {
            loop {
                if !key_is_schedulable(&state, &key).await {
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
                    "{} {} -> access_key={} proxy={} supplier_id={} key_id={} {}",
                    parts.method,
                    path,
                    access_key.id,
                    pool.id,
                    base_url.id,
                    key.id,
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
                        mark_key_fail(&state, key.id).await;
                        last_error = Some(ProxyAttemptError {
                            body: format!("Upstream returned {}", status.as_u16()),
                        });

                        if key_is_schedulable(&state, &key).await {
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
                        mark_key_success(&state, key.id).await;
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
                                .body(transformed_sse_body(response, conversion))
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
                            let body = match transform_response_body(&body, conversion) {
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
                        if path == RESPONSES_PATH {
                            info!(
                                "responses debug -> upstream_timeout supplier_id={} key_id={} url={}",
                                base_url.id,
                                key.id,
                                upstream_url_text
                            );
                        }
                        mark_key_fail(&state, key.id).await;
                        last_error = Some(ProxyAttemptError {
                            body: "Gateway Timeout".to_string(),
                        });

                        if key_is_schedulable(&state, &key).await {
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
                        mark_key_fail(&state, key.id).await;
                        last_error = Some(ProxyAttemptError {
                            body: format!("Bad Gateway: {err}"),
                        });

                        if key_is_schedulable(&state, &key).await {
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
        client: build_client(),
        db_path,
        admin,
    };

    let admin_routes = Router::new()
        .route("/admin", get(admin_page))
        .route("/admin/keys", get(access_keys_page))
        .route("/admin/api/proxies", get(get_proxies).post(create_pool))
        .route("/admin/api/proxies/save", post(create_proxy_save))
        .route("/admin/api/access-keys", get(get_access_keys).post(create_access_key))
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
        .route("/admin/api/keys/{id}/test", post(test_key))
        .route("/admin/api/keys/{id}/unban", post(unban_key))
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
    fn key_failures_escalate_from_temp_ban_to_daily_ban() {
        let key = ApiKeyConfig {
            id: 1,
            base_url_id: 1,
            api_key: "sk-test".to_string(),
            sort_order: 0,
            manually_disabled: false,
        };
        let now = fixed_now();
        let mut state = KeyRuntimeState::default();
        state.last_reset_date = now.date_naive();

        mark_key_fail_in_state(key.id, &mut state, now);
        mark_key_fail_in_state(key.id, &mut state, now);
        let availability = key_availability_in_state(&key, &mut state, now);
        assert_eq!(availability.fail_count, 2);
        assert!(availability.schedulable);

        mark_key_fail_in_state(key.id, &mut state, now);
        let availability = key_availability_in_state(&key, &mut state, now);
        assert_eq!(availability.fail_count, 3);
        assert!(availability.banned);

        let after_temp_ban = now + ChronoDuration::minutes(6);
        let availability = key_availability_in_state(&key, &mut state, after_temp_ban);
        assert_eq!(availability.fail_count, 3);
        assert!(!availability.banned);

        mark_key_fail_in_state(key.id, &mut state, after_temp_ban);
        let after_second_temp_ban = after_temp_ban + ChronoDuration::minutes(6);
        mark_key_fail_in_state(key.id, &mut state, after_second_temp_ban);
        let availability = key_availability_in_state(&key, &mut state, after_second_temp_ban);
        assert_eq!(availability.fail_count, 5);
        assert!(availability.banned);
    }

    #[test]
    fn key_success_clears_runtime_state() {
        let now = fixed_now();
        let mut state = KeyRuntimeState::default();
        mark_key_fail_in_state(1, &mut state, now);
        mark_key_fail_in_state(1, &mut state, now);
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
        };
        let mut state = KeyRuntimeState::default();
        let availability = key_availability_in_state(&key, &mut state, fixed_now());
        assert!(!availability.schedulable);
    }

    #[test]
    fn generated_access_key_uses_me_prefix() {
        let access_key = generate_access_key_value();
        assert!(access_key.starts_with("me-"));
        assert_eq!(access_key.len(), 67);
        assert!(access_key[3..].chars().all(|ch| ch.is_ascii_hexdigit()));
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
                params!["客户端 A", "me-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd", pool_id],
            )
            .unwrap();
        }

        let access_keys = load_access_keys(&path).unwrap();
        assert_eq!(access_keys.len(), 1);
        assert_eq!(access_keys[0].name, "客户端 A");
        assert_eq!(access_keys[0].proxy_id, 1);
        assert!(access_keys[0].access_key.starts_with("me-"));

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
