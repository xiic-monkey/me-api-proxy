use axum::{
    body::{to_bytes, Body, Bytes},
    extract::{Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Router,
};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;
use tracing;

/// 默认配置：当配置文件不存在时自动创建
const DEFAULT_CONFIG: &str = r#"{
  "newapis": "https://newapis.xyz",
  "default": "http://demo.com"
}"#;

/// 逐跳头部列表 - 这些头部仅在单次连接中有效，不应被代理转发
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

/// 应用状态：包含 HTTP 客户端和路由映射
#[derive(Clone)]
struct AppState {
    routes: Arc<RwLock<HashMap<String, String>>>, // 路由映射，支持并发读写
}

/// 获取配置文件路径：~/.me-api-proxy/apis.json
fn get_config_path() -> PathBuf {
    let home = dirs::home_dir().expect("Cannot find home directory");
    home.join(".me-api-proxy").join("apis.json")
}

/// 从配置文件加载路由映射
fn load_routes() -> HashMap<String, String> {
    let config_path = get_config_path();

    // 如果配置目录不存在，自动创建
    if let Some(parent) = config_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).expect("Failed to create config directory");
            tracing::info!("Created config directory: {:?}", parent);
        }
    }

    // 如果配置文件不存在，使用默认配置创建
    if !config_path.exists() {
        std::fs::write(&config_path, DEFAULT_CONFIG).expect("Failed to create config file");
        tracing::info!("Created config file: {:?}", config_path);
    }

    // 读取并解析配置文件
    let content = std::fs::read_to_string(&config_path).expect("Failed to read config file");
    let json: Map<String, Value> =
        serde_json::from_str(&content).expect("Failed to parse config JSON");

    // 将 JSON 对象转换为路由映射
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
    // 初始化日志系统，支持通过环境变量 RUST_LOG 控制日志级别
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "me_api_proxy=info,tower_http=info".into()),
        )
        .init();

    let config_path = get_config_path();
    let routes = load_routes();
    let state = AppState {
        routes: Arc::new(RwLock::new(routes)),
    };

    // 启动配置热重载任务：每 5 秒检查配置文件是否有变更
    let routes_clone = state.routes.clone();
    let config_path_clone = config_path.clone();
    tokio::spawn(async move {
        let mut last_modified: Option<SystemTime> = None;
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            if let Ok(meta) = std::fs::metadata(&config_path_clone) {
                if let Ok(modified) = meta.modified() {
                    // 检测文件修改时间是否变化
                    if last_modified.is_none() || last_modified.unwrap() != modified {
                        last_modified = Some(modified);
                        // 重新加载配置并更新路由映射
                        let new_routes = load_routes();
                        *routes_clone.write().await = new_routes;
                        tracing::info!("Config reloaded from {:?}", config_path_clone);
                    }
                }
            }
        }
    });

    // 构建路由：使用 fallback 处理所有请求，添加 HTTP 追踪中间件
    let app = Router::new()
        .fallback(proxy_handler)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = "0.0.0.0:8080";
    tracing::info!("Listening on {}", addr);

    // 启动 TCP 监听器并开始服务
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// 代理处理器：将请求转发到对应的上游服务器
async fn proxy_handler(State(state): State<AppState>, req: Request) -> Response {
    let path = req.uri().path().to_string();
    let query = req.uri().query().map(|q| format!("?{q}")).unwrap_or_default();

    // 解析 URL 路径：第一段作为路由名称，剩余部分作为转发路径
    // 例如：/newapis/v1/users -> 路由名=newapis, 转发路径=/v1/users
    let segments: Vec<&str> = path.trim_start_matches('/').splitn(2, '/').collect();
    let route_name = segments.first().unwrap_or(&"");
    let remaining_path = if segments.len() > 1 {
        format!("/{}", segments[1])
    } else {
        String::from("/")
    };

    // 查找对应的上游服务器地址
    let upstream_base = {
        let routes = state.routes.read().await;
        routes.get(*route_name).cloned()
    };

    // 如果路由不存在，返回 404 错误
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

    // 拼接完整的上游请求 URL
    let full_url = format!(
        "{}{}{}",
        upstream_base.trim_end_matches('/'),
        remaining_path,
        query
    );

    tracing::info!("{} {} -> {}", req.method(), path, full_url);

    // 拆分请求为头部和请求体
    let (parts, body) = req.into_parts();

    let request_body = match to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::error!("Failed to read request body: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                format!("Failed to read request body: {}", e),
            )
                .into_response();
        }
    };

    let upstream = match send_via_curl(&parts.method, &parts.headers, &full_url, request_body).await
    {
        Ok(response) => response,
        Err(e) => {
            tracing::error!("Upstream error: {}", e);
            return (StatusCode::BAD_GATEWAY, format!("Upstream error: {}", e)).into_response();
        }
    };

    // 构建响应，保留上游服务器的状态码
    let mut builder = Response::builder().status(upstream.status);

    // 转发所有响应头，过滤掉逐跳头部
    if let Some(headers) = builder.headers_mut() {
        for (k, v) in &upstream.headers {
            if !HOP_BY_HOP_HEADERS.contains(&k.as_str().to_lowercase().as_str()) {
                headers.insert(k.clone(), v.clone());
            }
        }
    }

    builder
        .body(Body::from(upstream.body))
        .unwrap_or_else(|e| {
            tracing::error!("Failed to build response: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        })
}

struct CurlResponse {
    status: StatusCode,
    headers: HeaderMap,
    body: Vec<u8>,
}

async fn send_via_curl(
    method: &axum::http::Method,
    headers: &HeaderMap,
    url: &str,
    body: Bytes,
) -> Result<CurlResponse, String> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let tmp_dir = std::env::temp_dir();
    let header_path = tmp_dir.join(format!("me-api-proxy-{nonce}.headers"));
    let body_path = tmp_dir.join(format!("me-api-proxy-{nonce}.body"));

    let mut command = Command::new("curl");
    command
        .arg("--silent")
        .arg("--show-error")
        .arg("--http1.1")
        .arg("--request")
        .arg(method.as_str())
        .arg("--url")
        .arg(url)
        .arg("--dump-header")
        .arg(&header_path)
        .arg("--output")
        .arg(&body_path)
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null());

    for (key, value) in headers {
        if should_forward_request_header(key.as_str()) {
            let value = value
                .to_str()
                .map_err(|e| format!("Invalid header {}: {}", key.as_str(), e))?;
            command.arg("-H").arg(format!("{}: {}", key.as_str(), value));
        }
    }

    if body.is_empty() {
        command.stdin(std::process::Stdio::null());
    } else {
        command.arg("--data-binary").arg("@-");
        command.stdin(std::process::Stdio::piped());
    }

    let mut child = command
        .spawn()
        .map_err(|e| format!("Failed to spawn curl: {}", e))?;

    if !body.is_empty() {
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(&body)
                .await
                .map_err(|e| format!("Failed to write request body to curl: {}", e))?;
        }
    }

    let output = child
        .wait_with_output()
        .await
        .map_err(|e| format!("Failed to wait for curl: {}", e))?;

    let raw_headers = tokio::fs::read(&header_path)
        .await
        .map_err(|e| format!("Failed to read curl response headers: {}", e))?;
    let response_body = tokio::fs::read(&body_path)
        .await
        .map_err(|e| format!("Failed to read curl response body: {}", e))?;

    let _ = tokio::fs::remove_file(&header_path).await;
    let _ = tokio::fs::remove_file(&body_path).await;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("curl exited with {}: {}", output.status, stderr.trim()));
    }

    let (status, headers) = parse_curl_headers(&raw_headers)?;

    Ok(CurlResponse {
        status,
        headers,
        body: response_body,
    })
}

fn should_forward_request_header(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower != "host"
        && lower != "content-length"
        && !HOP_BY_HOP_HEADERS.contains(&lower.as_str())
}

fn parse_curl_headers(raw: &[u8]) -> Result<(StatusCode, HeaderMap), String> {
    let normalized = String::from_utf8_lossy(raw).replace("\r\n", "\n");
    let block = normalized
        .split("\n\n")
        .filter(|block| block.trim_start().starts_with("HTTP/"))
        .last()
        .ok_or_else(|| "curl did not return HTTP response headers".to_string())?;

    let mut lines = block.lines();
    let status_line = lines
        .next()
        .ok_or_else(|| "curl response headers missing status line".to_string())?;
    let status_code = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| format!("Invalid status line from curl: {}", status_line))?
        .parse::<u16>()
        .map_err(|e| format!("Invalid status code from curl: {}", e))?;
    let status =
        StatusCode::from_u16(status_code).map_err(|e| format!("Invalid HTTP status: {}", e))?;

    let mut headers = HeaderMap::new();
    for line in lines {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }

        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| format!("Invalid response header line: {}", line))?;
        let name = HeaderName::from_bytes(name.trim().as_bytes())
            .map_err(|e| format!("Invalid response header name '{}': {}", name.trim(), e))?;
        let value = HeaderValue::from_str(value.trim())
            .map_err(|e| format!("Invalid response header value for '{}': {}", name, e))?;
        headers.append(name, value);
    }

    Ok((status, headers))
}
