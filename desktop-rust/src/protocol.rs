use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PROTOCOL_VERSION: &str = "campus-agent-desktop/v1";

#[derive(Debug, Serialize)]
pub struct WorkerRequest {
    pub protocol: &'static str,
    pub request_id: String,
    pub command: String,
    pub payload: Value,
}

impl WorkerRequest {
    pub fn new(request_id: String, command: impl Into<String>, payload: Value) -> Self {
        Self {
            protocol: PROTOCOL_VERSION,
            request_id,
            command: command.into(),
            payload,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct WorkerEvent {
    pub protocol: String,
    #[serde(rename = "request_id")]
    pub _request_id: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub payload: Value,
}

impl WorkerEvent {
    pub fn local_error(message: impl Into<String>) -> Self {
        Self {
            protocol: PROTOCOL_VERSION.to_owned(),
            _request_id: "desktop".to_owned(),
            kind: "fatal".to_owned(),
            payload: serde_json::json!({ "message": message.into() }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_uses_versioned_ndjson_shape() {
        let request = WorkerRequest::new(
            "desktop-1".to_owned(),
            "chat",
            serde_json::json!({ "query": "校园卡" }),
        );
        let encoded = serde_json::to_value(request).expect("request must serialize");

        assert_eq!(encoded["protocol"], PROTOCOL_VERSION);
        assert_eq!(encoded["request_id"], "desktop-1");
        assert_eq!(encoded["command"], "chat");
        assert_eq!(encoded["payload"]["query"], "校园卡");
    }

    #[test]
    fn response_deserializes_type_and_request_id() {
        let event: WorkerEvent = serde_json::from_value(serde_json::json!({
            "protocol": PROTOCOL_VERSION,
            "request_id": "desktop-2",
            "type": "done",
            "payload": { "ok": true }
        }))
        .expect("event must deserialize");

        assert_eq!(event._request_id, "desktop-2");
        assert_eq!(event.kind, "done");
        assert_eq!(event.payload["ok"], true);
    }
}
