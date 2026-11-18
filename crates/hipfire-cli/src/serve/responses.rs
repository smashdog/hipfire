// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! OpenAI Responses API adapter (`POST /v1/responses`).
//!
//! Concern: lowering a Responses request (`instructions`, `input` items,
//! function/namespace/custom tools, `reasoning`, `text.format`) onto the
//! chat-completions body the serve pipeline already runs (hosted tools are
//! left out rather than refused), and rendering that pipeline's
//! [`Completion`] and stream events back as a `response` object or Responses
//! SSE events. There is no second generation path, and serve keeps no
//! response state: `previous_response_id` and item references are refused,
//! and the client resends the conversation in `input`.

use crate::serve::complete::{
    completion_hipfire, completion_timings, completion_usage, openai_finish_reason, Completion,
};
use crate::serve::http::{id_suffix, sse_event};
use hipfire_runtime::prompt_frame::ToolCall;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// A Responses request lowered onto the chat-completions body.
#[derive(Debug)]
pub(crate) struct Lowered {
    pub(crate) chat: Value,
    pub(crate) stream: bool,
    pub(crate) echo: ResponseEcho,
}

/// Request parameters a `response` object echoes back. `tool_names` is not
/// echoed: it maps the flat names the model sees back to the namespace and
/// tool kind the client sent.
#[derive(Debug, Clone)]
pub(crate) struct ResponseEcho {
    fields: Map<String, Value>,
    tool_names: ToolNames,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolKind {
    Function,
    Custom,
}

#[derive(Debug, Clone)]
struct ToolRef {
    namespace: Option<String>,
    name: String,
    kind: ToolKind,
}

/// Flat model-facing tool name → the tool the client actually declared.
#[derive(Debug, Clone, Default)]
struct ToolNames {
    by_name: HashMap<String, ToolRef>,
}

#[derive(Debug)]
struct NormalizedTools {
    tools: Vec<Value>,
    names: ToolNames,
    skipped: Vec<String>,
}

/// Lower a Responses request body. `Err` is a client error (HTTP 400).
pub(crate) fn lower_request(body: &Value) -> Result<Lowered, String> {
    let obj = body
        .as_object()
        .ok_or_else(|| "request body must be a JSON object".to_owned())?;
    for key in ["previous_response_id", "conversation", "prompt"] {
        if present(obj, key).is_some() {
            return Err(format!(
                "{key} is not supported: hipfire serve stores no responses or prompts; \
                 send the full conversation in input"
            ));
        }
    }
    if present(obj, "background").and_then(Value::as_bool) == Some(true) {
        return Err("background responses are not supported".into());
    }
    let mut chat = Map::new();
    if let Some(model) = obj.get("model") {
        chat.insert("model".into(), model.clone());
    }
    if let Some(max) = present(obj, "max_output_tokens") {
        let max = max
            .as_u64()
            .filter(|n| *n >= 1)
            .ok_or_else(|| "max_output_tokens must be a positive integer".to_owned())?;
        chat.insert("max_tokens".into(), max.into());
    }

    let mut messages = Vec::new();
    if let Some(instructions) = present(obj, "instructions") {
        let text = instructions
            .as_str()
            .ok_or_else(|| "instructions must be a string".to_owned())?;
        if !text.is_empty() {
            messages.push(json!({ "role": "system", "content": text }));
        }
    }
    match present(obj, "input") {
        None => return Err("input: field required".into()),
        Some(Value::String(text)) => messages.push(json!({ "role": "user", "content": text })),
        Some(Value::Array(items)) => lower_items(items, &mut messages)?,
        Some(_) => return Err("input must be a string or an array of items".into()),
    }
    normalize_system_messages(&mut messages);
    chat.insert("messages".into(), Value::Array(messages));

    // `top_k` and `stop` are hipfire extensions (the Responses API has
    // neither); they reach the pipeline exactly as on chat completions.
    for key in ["temperature", "top_p", "top_k", "stop"] {
        if let Some(value) = present(obj, key) {
            chat.insert(key.into(), value.clone());
        }
    }
    // `additional_tools` input items (Codex) carry more tool definitions; they
    // join the request's own `tools` before name lowering.
    let mut given_tools = match present(obj, "tools") {
        None => Vec::new(),
        Some(tools) => tools
            .as_array()
            .cloned()
            .ok_or_else(|| "tools must be an array".to_owned())?,
    };
    if let Some(items) = present(obj, "input").and_then(Value::as_array) {
        for item in items {
            if item.get("type").and_then(Value::as_str) == Some("additional_tools") {
                if let Some(extra) = item.get("tools").and_then(Value::as_array) {
                    given_tools.extend(extra.iter().cloned());
                }
            }
        }
    }
    let mut tool_names = ToolNames::default();
    if !given_tools.is_empty() {
        let normalized = lower_tools(&given_tools)?;
        tool_names = normalized.names;
        // `tool_choice: none` means the model may not call anything, so the
        // tools need not reach the chat template at all (as in PolyStrata).
        let choice_none = present(obj, "tool_choice").and_then(Value::as_str) == Some("none");
        if !choice_none && !normalized.tools.is_empty() {
            chat.insert("tools".into(), Value::Array(normalized.tools));
        }
        note_skipped_tools(&normalized.skipped);
    }
    if let Some(choice) = present(obj, "tool_choice") {
        chat.insert("tool_choice".into(), lower_tool_choice(choice)?);
    }
    if let Some(format) = present(obj, "text").and_then(|text| text.get("format")) {
        if let Some(response_format) = lower_text_format(format)? {
            chat.insert("response_format".into(), response_format);
        }
    }
    let effort = match present(obj, "reasoning").and_then(|r| r.get("effort")) {
        None | Some(Value::Null) => None,
        Some(Value::String(effort)) => Some(effort.clone()),
        Some(_) => return Err("reasoning.effort must be a string".into()),
    };
    if let Some(effort) = &effort {
        chat.insert("reasoning_effort".into(), Value::String(effort.clone()));
    }
    let stream = match present(obj, "stream") {
        None => false,
        Some(Value::Bool(stream)) => *stream,
        Some(_) => return Err("stream must be a boolean".into()),
    };

    let echo = |key: &str, default: Value| present(obj, key).cloned().unwrap_or(default);
    let mut fields = Map::new();
    fields.insert("instructions".into(), echo("instructions", Value::Null));
    fields.insert(
        "max_output_tokens".into(),
        echo("max_output_tokens", Value::Null),
    );
    fields.insert("temperature".into(), echo("temperature", Value::Null));
    fields.insert("top_p".into(), echo("top_p", Value::Null));
    fields.insert("tool_choice".into(), echo("tool_choice", json!("auto")));
    fields.insert("tools".into(), echo("tools", json!([])));
    fields.insert(
        "parallel_tool_calls".into(),
        echo("parallel_tool_calls", json!(true)),
    );
    fields.insert(
        "reasoning".into(),
        json!({ "effort": effort, "summary": Value::Null }),
    );
    fields.insert(
        "text".into(),
        echo("text", json!({ "format": { "type": "text" } })),
    );
    fields.insert("metadata".into(), echo("metadata", json!({})));
    fields.insert("previous_response_id".into(), Value::Null);
    fields.insert("store".into(), Value::Bool(false));
    fields.insert("truncation".into(), json!("disabled"));
    if let Some(user) = present(obj, "user") {
        fields.insert("user".into(), user.clone());
    }
    Ok(Lowered {
        chat: Value::Object(chat),
        stream,
        echo: ResponseEcho { fields, tool_names },
    })
}

/// A field that is present and not JSON `null`.
fn present<'a>(obj: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    obj.get(key).filter(|value| !value.is_null())
}

/// Folds `input` items into chat messages. One assistant turn is the run of
/// `reasoning`, assistant `message` and `function_call` items between other
/// items: reasoning becomes the turn's `reasoning_content`, function calls
/// its `tool_calls`.
struct ItemFolder<'a> {
    out: &'a mut Vec<Value>,
    /// Index in `out` of the assistant message the current turn's
    /// `function_call` items attach to.
    open_assistant: Option<usize>,
    pending_reasoning: String,
}

impl ItemFolder<'_> {
    fn close_turn(&mut self) {
        self.open_assistant = None;
    }

    fn start_assistant(&mut self, content: Value) -> usize {
        let mut entry = json!({ "role": "assistant", "content": content });
        if !self.pending_reasoning.is_empty() {
            entry["reasoning_content"] = Value::String(std::mem::take(&mut self.pending_reasoning));
        }
        self.out.push(entry);
        let index = self.out.len() - 1;
        self.open_assistant = Some(index);
        index
    }

    fn push_call(&mut self, call: Value) {
        let index = match self.open_assistant {
            Some(index) => index,
            None => self.start_assistant(json!("")),
        };
        let entry = &mut self.out[index];
        match entry.get_mut("tool_calls").and_then(Value::as_array_mut) {
            Some(calls) => calls.push(call),
            None => entry["tool_calls"] = json!([call]),
        }
    }
}

fn lower_items(items: &[Value], out: &mut Vec<Value>) -> Result<(), String> {
    let mut folder = ItemFolder {
        out,
        open_assistant: None,
        pending_reasoning: String::new(),
    };
    for (at, item) in items.iter().enumerate() {
        let path = format!("input.{at}");
        match item.get("type").and_then(Value::as_str) {
            None | Some("message") => {
                let role = item
                    .get("role")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("{path}.role: field required"))?;
                let content = message_content(item.get("content"), &path)?;
                match role {
                    "assistant" => {
                        folder.start_assistant(content);
                    }
                    "user" | "system" | "developer" => {
                        folder.close_turn();
                        // The chat template has no `developer` role; Codex's
                        // developer items carry the same standing instructions.
                        let role = if role == "developer" { "system" } else { role };
                        folder.out.push(json!({ "role": role, "content": content }));
                    }
                    other => return Err(format!("{path}.role: unexpected role `{other}`")),
                }
            }
            Some("function_call" | "custom_tool_call") => {
                let field = |key: &str| {
                    item.get(key)
                        .and_then(Value::as_str)
                        .ok_or_else(|| format!("{path}.{key}: field required"))
                };
                let name = qualified_name(item, field("name")?);
                let arguments =
                    if item.get("type").and_then(Value::as_str) == Some("custom_tool_call") {
                        let input = item.get("input").and_then(Value::as_str).unwrap_or("");
                        json!({ "input": input }).to_string()
                    } else {
                        field("arguments")?.to_owned()
                    };
                folder.push_call(json!({
                    "id": field("call_id")?,
                    "type": "function",
                    "function": { "name": name, "arguments": arguments },
                }));
            }
            Some("function_call_output" | "custom_tool_call_output") => {
                folder.close_turn();
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("{path}.call_id: field required"))?;
                let output = match item.get("output") {
                    Some(Value::String(output)) => output.clone(),
                    Some(Value::Array(parts)) => {
                        let mut text = String::new();
                        for (k, part) in parts.iter().enumerate() {
                            match part.get("type").and_then(Value::as_str) {
                                Some("input_text" | "output_text" | "text") => text.push_str(
                                    part.get("text").and_then(Value::as_str).ok_or_else(|| {
                                        format!("{path}.output.{k}.text: field required")
                                    })?,
                                ),
                                _ => {
                                    return Err(format!(
                                        "{path}.output.{k}: only text output parts are supported"
                                    ))
                                }
                            }
                        }
                        text
                    }
                    _ => return Err(format!("{path}.output: field required (string)")),
                };
                folder
                    .out
                    .push(json!({ "role": "tool", "tool_call_id": call_id, "content": output }));
            }
            Some("additional_tools") => {
                // Merged into the request's tools before lowering; it is not
                // a conversation item.
            }
            Some("reasoning") => {
                // Raw reasoning text when the client kept it, else the summary.
                let texts = |key: &str| -> Vec<&str> {
                    item.get(key)
                        .and_then(Value::as_array)
                        .map(|parts| {
                            parts
                                .iter()
                                .filter_map(|part| part.get("text").and_then(Value::as_str))
                                .collect()
                        })
                        .unwrap_or_default()
                };
                let content = texts("content");
                let text = if content.is_empty() {
                    texts("summary").join("\n\n")
                } else {
                    content.concat()
                };
                folder.pending_reasoning.push_str(&text);
            }
            Some("item_reference") => {
                return Err(format!(
                    "{path}: item_reference needs stored responses; serve keeps none"
                ));
            }
            Some(other) => return Err(format!("{path}: item type `{other}` is not supported")),
        }
    }
    Ok(())
}

/// A tool call's `namespace` joins its name the way the flattened tool list
/// spells it. Codex sends namespaced calls this way; a call without a
/// namespace is unchanged.
fn qualified_name(item: &Value, name: &str) -> String {
    match item
        .get("namespace")
        .and_then(Value::as_str)
        .filter(|namespace| !namespace.is_empty())
    {
        Some(namespace) => format!("{namespace}.{name}"),
        None => name.to_owned(),
    }
}

/// The Qwen chat template accepts a system message only at the very start.
/// Codex sends `instructions` plus developer/system input items, so consecutive
/// leading system messages become one system message, and a later system
/// message becomes a user message in place (as in PolyStrata).
fn normalize_system_messages(messages: &mut Vec<Value>) {
    let mut lead = 0;
    while messages
        .get(lead)
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str)
        == Some("system")
    {
        lead += 1;
    }
    if lead > 1 {
        let mut text = String::new();
        for message in &messages[..lead] {
            let part = content_text(message.get("content"));
            if part.is_empty() {
                continue;
            }
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&part);
        }
        messages.drain(..lead);
        messages.insert(0, json!({ "role": "system", "content": text }));
    }
    for (i, message) in messages.iter_mut().enumerate() {
        if i > 0 && message.get("role").and_then(Value::as_str) == Some("system") {
            if let Some(object) = message.as_object_mut() {
                object.insert("role".into(), json!("user"));
            }
        }
    }
}

fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                if let Some(piece) = part
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.get("refusal").and_then(Value::as_str))
                {
                    text.push_str(piece);
                }
            }
            text
        }
        _ => String::new(),
    }
}

/// Message `content`: a string, or parts. Text parts concatenate exactly like
/// chat-completions content parts; an `input_image` becomes an `image_url`
/// part, which keeps the content an array as on chat.
fn message_content(content: Option<&Value>, path: &str) -> Result<Value, String> {
    let parts = match content {
        Some(Value::String(text)) => return Ok(Value::String(text.clone())),
        Some(Value::Array(parts)) => parts,
        _ => return Err(format!("{path}.content: field required (string or array)")),
    };
    let mut text = String::new();
    let mut images = Vec::new();
    for (k, part) in parts.iter().enumerate() {
        let part_path = format!("{path}.content.{k}");
        match part.get("type").and_then(Value::as_str) {
            Some("input_text" | "output_text" | "text") => text.push_str(
                part.get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("{part_path}.text: field required"))?,
            ),
            Some("refusal") => text.push_str(
                part.get("refusal")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("{part_path}.refusal: field required"))?,
            ),
            Some("input_image") => {
                let url = part
                    .get("image_url")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        format!(
                            "{part_path}: input_image needs image_url (file_id is not supported)"
                        )
                    })?;
                images.push(json!({ "type": "image_url", "image_url": { "url": url } }));
            }
            Some(other) => {
                return Err(format!(
                    "{part_path}: content part type `{other}` is not supported"
                ))
            }
            None => return Err(format!("{part_path}.type: field required")),
        }
    }
    if images.is_empty() {
        return Ok(Value::String(text));
    }
    let mut out = Vec::with_capacity(images.len() + 1);
    if !text.is_empty() {
        out.push(json!({ "type": "text", "text": text }));
    }
    out.extend(images);
    Ok(Value::Array(out))
}

/// Responses tools → chat function tools, with the map that turns the flat
/// names back into the client's namespace/name/kind. A namespace's members
/// become `namespace.name` (Qwen also writes `namespace__name` for MCP
/// tools, so that spelling aliases the same entry). Hosted tools (web search,
/// file search, …) cannot run here and are left out instead of failing the
/// request.
fn lower_tools(tools: &[Value]) -> Result<NormalizedTools, String> {
    let mut out = NormalizedTools {
        tools: Vec::new(),
        names: ToolNames::default(),
        skipped: Vec::new(),
    };
    for (k, tool) in tools.iter().enumerate() {
        let path = format!("tools.{k}");
        let tool = tool
            .as_object()
            .ok_or_else(|| format!("{path}: expected a tool object"))?;
        let kind = match tool.get("type") {
            Some(Value::String(kind)) => kind.as_str(),
            Some(_) => return Err(format!("{path}.type: must be a string")),
            None => return Err(format!("{path}.type: field required")),
        };
        match kind {
            "function" | "custom" => add_tool(tool, &path, None, "", &mut out)?,
            "namespace" => {
                let namespace = tool
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty());
                let description = tool
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let members: &[Value] = match tool.get("tools") {
                    None | Some(Value::Null) => &[],
                    Some(Value::Array(members)) => members,
                    Some(_) => return Err(format!("{path}.tools: must be an array")),
                };
                for (j, member) in members.iter().enumerate() {
                    let member_path = format!("{path}.tools.{j}");
                    let member = member.as_object().ok_or_else(|| {
                        format!("{member_path}: a namespace holds function tools")
                    })?;
                    let member_kind = match member.get("type") {
                        None => "function",
                        Some(Value::String(kind)) => kind.as_str(),
                        Some(_) => {
                            return Err(format!("{member_path}: a namespace holds function tools"))
                        }
                    };
                    if !matches!(member_kind, "function" | "custom") {
                        return Err(format!("{member_path}: a namespace holds function tools"));
                    }
                    add_tool(member, &member_path, namespace, description, &mut out)?;
                }
            }
            other => out.skipped.push(other.to_owned()),
        }
    }
    Ok(out)
}

fn add_tool(
    tool: &Map<String, Value>,
    path: &str,
    namespace: Option<&str>,
    namespace_description: &str,
    out: &mut NormalizedTools,
) -> Result<(), String> {
    let name = tool
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| format!("{path}.name: field required"))?;
    let kind = match tool.get("type").and_then(Value::as_str) {
        Some("custom") => ToolKind::Custom,
        _ => ToolKind::Function,
    };
    let flat = match namespace {
        Some(namespace) => format!("{namespace}.{name}"),
        None => name.to_owned(),
    };
    let mut description = tool
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    if let Some(namespace) = namespace {
        if !namespace_description.is_empty() {
            description = if description.is_empty() {
                namespace_description.to_owned()
            } else {
                format!("{namespace_description}\n\n{description}")
            };
        }
    }
    let parameters = match kind {
        ToolKind::Custom => {
            // Free-form text, optionally constrained by a grammar the model
            // reads. The model sees one `input` string; the adapter packs it
            // into the chat tool-call arguments on the way back in.
            if let Some(format) = tool.get("format") {
                if format.get("type").and_then(Value::as_str) == Some("grammar") {
                    if let Some(definition) = format
                        .get("definition")
                        .and_then(Value::as_str)
                        .filter(|definition| !definition.is_empty())
                    {
                        let syntax = format.get("syntax").and_then(Value::as_str).unwrap_or("");
                        description.push_str(&format!(
                            "\n\nThe input must follow this {syntax} grammar:\n{definition}"
                        ));
                    }
                }
            }
            json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string", "description": "the raw input" }
                },
                "required": ["input"],
            })
        }
        ToolKind::Function => match tool.get("parameters") {
            None | Some(Value::Null) => default_parameters(),
            Some(Value::Object(parameters)) if parameters.is_empty() => default_parameters(),
            Some(parameters) => parameters.clone(),
        },
    };
    out.tools.push(json!({
        "type": "function",
        "function": {
            "name": flat,
            "description": description,
            "parameters": parameters,
        },
    }));
    let reference = ToolRef {
        namespace: namespace.map(str::to_owned),
        name: name.to_owned(),
        kind,
    };
    out.names.by_name.insert(flat, reference.clone());
    if let Some(namespace) = namespace {
        out.names
            .by_name
            .entry(format!("{namespace}__{name}"))
            .or_insert(reference);
    }
    Ok(())
}

fn default_parameters() -> Value {
    json!({ "type": "object", "properties": {} })
}

/// Say once per tool that a hosted tool was left out, like PolyStrata.
fn note_skipped_tools(skipped: &[String]) {
    if skipped.is_empty() {
        return;
    }
    static NOTED: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let noted = NOTED.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let mut noted = noted.lock().unwrap_or_else(|poison| poison.into_inner());
    let mut fresh: Vec<&str> = skipped
        .iter()
        .map(String::as_str)
        .filter(|name| !noted.contains(*name))
        .collect();
    if fresh.is_empty() {
        return;
    }
    fresh.sort_unstable();
    fresh.dedup();
    eprintln!(
        "[hipfire] /v1/responses: left out hosted tools {} (this server cannot run them)",
        fresh.join(", ")
    );
    noted.extend(skipped.iter().cloned());
}

fn lower_tool_choice(choice: &Value) -> Result<Value, String> {
    match choice {
        Value::String(mode) if matches!(mode.as_str(), "auto" | "none" | "required") => {
            Ok(choice.clone())
        }
        Value::Object(_) if choice.get("type").and_then(Value::as_str) == Some("function") => {
            let name = choice
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "tool_choice.name: field required".to_owned())?;
            let name = qualified_name(choice, name);
            Ok(json!({ "type": "function", "function": { "name": name } }))
        }
        _ => Err("tool_choice must be auto, none, required or {type: function, name}".into()),
    }
}

/// `text.format` → chat `response_format` (enforced where chat enforces it:
/// the multi-slot route's `json_schema`).
fn lower_text_format(format: &Value) -> Result<Option<Value>, String> {
    match format.get("type").and_then(Value::as_str) {
        Some("text") => Ok(None),
        Some("json_object") => Ok(Some(json!({ "type": "json_object" }))),
        Some("json_schema") => {
            let mut json_schema = Map::new();
            for key in ["name", "schema", "strict", "description"] {
                if let Some(value) = format.get(key).filter(|v| !v.is_null()) {
                    json_schema.insert(key.into(), value.clone());
                }
            }
            Ok(Some(
                json!({ "type": "json_schema", "json_schema": json_schema }),
            ))
        }
        _ => Err("text.format.type must be text, json_schema or json_object".into()),
    }
}

/// `resp_…` id derived from the serve request id.
pub(crate) fn response_id(request_id: &str) -> String {
    format!("resp_{}", id_suffix(request_id))
}

fn item_id(prefix: &str, request_id: &str, output_index: usize) -> String {
    format!("{prefix}_{}_{output_index}", id_suffix(request_id))
}

fn call_id(request_id: &str, index: usize, call: &ToolCall) -> String {
    call.id
        .clone()
        .unwrap_or_else(|| format!("call_{}_{index}", id_suffix(request_id)))
}

/// The tool calls this completion releases: only on a tool-safe terminal,
/// exactly like the chat route's `tool_calls`.
fn released_calls(completion: &Completion) -> &[ToolCall] {
    if openai_finish_reason(&completion.done) == "tool_calls" {
        &completion.tool_calls
    } else {
        &[]
    }
}

/// `length` → `incomplete` (`max_output_tokens`); everything else completed.
fn status(completion: &Completion) -> (&'static str, Value) {
    if openai_finish_reason(&completion.done) == "length" {
        ("incomplete", json!({ "reason": "max_output_tokens" }))
    } else {
        ("completed", Value::Null)
    }
}

/// Responses usage. `input_tokens` includes the session-cache hits, which
/// `input_tokens_details.cached_tokens` reports.
pub(crate) fn usage(completion: &Completion) -> Value {
    let usage = completion_usage(completion);
    json!({
        "input_tokens": usage["prompt_tokens"],
        "input_tokens_details": { "cached_tokens": usage["prompt_tokens_details"]["cached_tokens"] },
        "output_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
    })
}

fn reasoning_item(id: String, text: &str) -> Value {
    json!({
        "type": "reasoning",
        "id": id,
        "summary": [],
        "content": [{ "type": "reasoning_text", "text": text }],
    })
}

fn output_text_part(text: &str) -> Value {
    json!({ "type": "output_text", "text": text, "annotations": [], "logprobs": [] })
}

fn message_item(id: String, text: &str, status: &str) -> Value {
    let content = if status == "completed" {
        json!([output_text_part(text)])
    } else {
        json!([])
    };
    json!({ "type": "message", "id": id, "status": status, "role": "assistant", "content": content })
}

/// Resolve the model's flat call name to the client's namespace, name and
/// tool kind. A name the request did not declare stays a plain function.
fn resolved_tool(echo: &ResponseEcho, name: &str) -> (Option<String>, String, ToolKind) {
    match echo.tool_names.by_name.get(name) {
        Some(tool) => (tool.namespace.clone(), tool.name.clone(), tool.kind),
        None => (None, name.to_owned(), ToolKind::Function),
    }
}

fn function_call_item(
    id: String,
    call_id: String,
    name: &str,
    namespace: Option<&str>,
    arguments: &str,
    status: &str,
) -> Value {
    let mut item = json!({
        "type": "function_call",
        "id": id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    });
    if let Some(namespace) = namespace {
        item["namespace"] = Value::String(namespace.to_owned());
    }
    item
}

fn custom_tool_call_item(
    id: String,
    call_id: String,
    name: &str,
    namespace: Option<&str>,
    input: &str,
    status: &str,
) -> Value {
    let mut item = json!({
        "type": "custom_tool_call",
        "id": id,
        "call_id": call_id,
        "name": name,
        "input": input,
        "status": status,
    });
    if let Some(namespace) = namespace {
        item["namespace"] = Value::String(namespace.to_owned());
    }
    item
}

/// A custom tool's whole raw input, which the chat tool call carries as
/// `{"input": "..."}`.
fn custom_input(arguments: &Value) -> String {
    if let Value::Object(arguments) = arguments {
        if let Some(Value::String(input)) = arguments.get("input") {
            return input.clone();
        }
    }
    arguments.to_string()
}

#[allow(clippy::too_many_arguments)]
fn response_object(
    echo: &ResponseEcho,
    id: &str,
    created_at: u64,
    model: &str,
    status: &str,
    output: Vec<Value>,
    usage: Value,
    incomplete_details: Value,
    error: Value,
) -> Value {
    let mut obj = echo.fields.clone();
    obj.insert("id".into(), Value::String(id.to_owned()));
    obj.insert("object".into(), json!("response"));
    obj.insert("created_at".into(), created_at.into());
    obj.insert("status".into(), json!(status));
    obj.insert("model".into(), Value::String(model.to_owned()));
    obj.insert("output".into(), Value::Array(output));
    obj.insert("usage".into(), usage);
    obj.insert("incomplete_details".into(), incomplete_details);
    obj.insert("error".into(), error);
    Value::Object(obj)
}

/// Non-streaming `response` object: a reasoning item, a message item and
/// the released function calls, in that order, each only when present.
pub(crate) fn response_json(completion: &Completion, echo: &ResponseEcho) -> Value {
    let rid = &completion.id;
    let mut output = Vec::new();
    if !completion.reasoning_content.is_empty() {
        output.push(reasoning_item(
            item_id("rs", rid, output.len()),
            &completion.reasoning_content,
        ));
    }
    if !completion.content.is_empty() {
        output.push(message_item(
            item_id("msg", rid, output.len()),
            &completion.content,
            "completed",
        ));
    }
    for (index, call) in released_calls(completion).iter().enumerate() {
        let (namespace, name, kind) = resolved_tool(echo, &call.name);
        let output_index = output.len();
        let call_id = call_id(rid, index, call);
        output.push(match kind {
            ToolKind::Function => {
                let arguments = call.arguments.to_string();
                function_call_item(
                    item_id("fc", rid, output_index),
                    call_id,
                    &name,
                    namespace.as_deref(),
                    &arguments,
                    "completed",
                )
            }
            ToolKind::Custom => custom_tool_call_item(
                item_id("ctc", rid, output_index),
                call_id,
                &name,
                namespace.as_deref(),
                &custom_input(&call.arguments),
                "completed",
            ),
        });
    }
    let (status, incomplete) = status(completion);
    let mut response = response_object(
        echo,
        &response_id(rid),
        completion.created,
        &completion.model,
        status,
        output,
        usage(completion),
        incomplete,
        Value::Null,
    );
    response["timings"] = completion_timings(completion);
    response["hipfire"] = completion_hipfire(completion);
    response
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ItemKind {
    Reasoning,
    Message,
}

#[derive(Debug)]
struct OpenItem {
    kind: ItemKind,
    id: String,
    output_index: usize,
    text: String,
}

/// Streaming encoder: `response.created` / `response.in_progress`, one
/// output item per reasoning or text span as the pipeline emits it, the
/// released function calls, then `response.completed` (or
/// `response.incomplete`). Every event carries a `sequence_number`.
#[derive(Debug)]
pub(crate) struct ResponseStream {
    echo: ResponseEcho,
    request_id: String,
    created_at: u64,
    model: String,
    sequence: u64,
    output: Vec<Value>,
    open: Option<OpenItem>,
}

impl ResponseStream {
    pub(crate) fn new(echo: ResponseEcho, request_id: &str, created_at: u64, model: &str) -> Self {
        Self {
            echo,
            request_id: request_id.to_owned(),
            created_at,
            model: model.to_owned(),
            sequence: 0,
            output: Vec::new(),
            open: None,
        }
    }

    fn emit(&mut self, bytes: &mut Vec<u8>, name: &str, mut data: Value) {
        data["type"] = json!(name);
        data["sequence_number"] = self.sequence.into();
        self.sequence += 1;
        bytes.extend(sse_event(name, &data));
    }

    fn snapshot(
        &self,
        status: &str,
        output: Vec<Value>,
        usage: Value,
        incomplete: Value,
        error: Value,
    ) -> Value {
        response_object(
            &self.echo,
            &response_id(&self.request_id),
            self.created_at,
            &self.model,
            status,
            output,
            usage,
            incomplete,
            error,
        )
    }

    /// `response.created` and `response.in_progress`.
    pub(crate) fn preamble(&mut self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for name in ["response.created", "response.in_progress"] {
            let response = self.snapshot(
                "in_progress",
                Vec::new(),
                Value::Null,
                Value::Null,
                Value::Null,
            );
            self.emit(&mut bytes, name, json!({ "response": response }));
        }
        bytes
    }

    /// One pipeline event (`token` / `reasoning`) as item events.
    pub(crate) fn event(&mut self, event: &Value) -> Option<Vec<u8>> {
        let kind = match event.get("type").and_then(Value::as_str) {
            Some("token") => ItemKind::Message,
            Some("reasoning") => ItemKind::Reasoning,
            _ => return None,
        };
        let text = event
            .get("text")
            .and_then(Value::as_str)
            .filter(|t| !t.is_empty())?;
        let mut bytes = Vec::new();
        if self.open.as_ref().map(|item| item.kind) != Some(kind) {
            self.close_item(&mut bytes);
            self.open_item(kind, &mut bytes);
        }
        let Some(item) = self.open.as_mut() else {
            return None;
        };
        item.text.push_str(text);
        let (id, output_index) = (item.id.clone(), item.output_index);
        let name = match kind {
            ItemKind::Message => "response.output_text.delta",
            ItemKind::Reasoning => "response.reasoning_text.delta",
        };
        let mut data = json!({
            "item_id": id,
            "output_index": output_index,
            "content_index": 0,
            "delta": text,
        });
        if kind == ItemKind::Message {
            data["logprobs"] = json!([]);
        }
        self.emit(&mut bytes, name, data);
        Some(bytes)
    }

    fn open_item(&mut self, kind: ItemKind, bytes: &mut Vec<u8>) {
        let output_index = self.output.len();
        let (id, item) = match kind {
            ItemKind::Reasoning => {
                let id = item_id("rs", &self.request_id, output_index);
                let item = json!({ "type": "reasoning", "id": id, "summary": [], "content": [] });
                (id, item)
            }
            ItemKind::Message => {
                let id = item_id("msg", &self.request_id, output_index);
                (id.clone(), message_item(id, "", "in_progress"))
            }
        };
        self.emit(
            bytes,
            "response.output_item.added",
            json!({ "output_index": output_index, "item": item }),
        );
        if kind == ItemKind::Message {
            self.emit(
                bytes,
                "response.content_part.added",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": output_text_part(""),
                }),
            );
        }
        self.open = Some(OpenItem {
            kind,
            id,
            output_index,
            text: String::new(),
        });
    }

    fn close_item(&mut self, bytes: &mut Vec<u8>) {
        let Some(item) = self.open.take() else {
            return;
        };
        let base =
            json!({ "item_id": item.id, "output_index": item.output_index, "content_index": 0 });
        let done = match item.kind {
            ItemKind::Message => {
                let mut text_done = base.clone();
                text_done["text"] = Value::String(item.text.clone());
                text_done["logprobs"] = json!([]);
                self.emit(bytes, "response.output_text.done", text_done);
                let mut part_done = base;
                part_done["part"] = output_text_part(&item.text);
                self.emit(bytes, "response.content_part.done", part_done);
                message_item(item.id, &item.text, "completed")
            }
            ItemKind::Reasoning => {
                let mut text_done = base;
                text_done["text"] = Value::String(item.text.clone());
                self.emit(bytes, "response.reasoning_text.done", text_done);
                reasoning_item(item.id, &item.text)
            }
        };
        self.emit(
            bytes,
            "response.output_item.done",
            json!({ "output_index": item.output_index, "item": done }),
        );
        self.output.push(done);
    }

    /// Close the open item, stream the released function calls, then
    /// `response.completed` / `response.incomplete` with the full response.
    pub(crate) fn terminal(&mut self, completion: &Completion) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.close_item(&mut bytes);
        for (index, call) in released_calls(completion).iter().enumerate() {
            let (namespace, name, kind) = resolved_tool(&self.echo, &call.name);
            let output_index = self.output.len();
            let prefix = match kind {
                ToolKind::Function => "fc",
                ToolKind::Custom => "ctc",
            };
            let id = item_id(prefix, &self.request_id, output_index);
            let call_id = call_id(&self.request_id, index, call);
            let item = match kind {
                ToolKind::Function => {
                    let arguments = call.arguments.to_string();
                    self.emit(
                        &mut bytes,
                        "response.output_item.added",
                        json!({
                            "output_index": output_index,
                            "item": function_call_item(
                                id.clone(),
                                call_id.clone(),
                                &name,
                                namespace.as_deref(),
                                "",
                                "in_progress",
                            ),
                        }),
                    );
                    self.emit(
                        &mut bytes,
                        "response.function_call_arguments.delta",
                        json!({ "item_id": id, "output_index": output_index, "delta": arguments }),
                    );
                    self.emit(
                        &mut bytes,
                        "response.function_call_arguments.done",
                        json!({ "item_id": id, "output_index": output_index, "name": name, "arguments": arguments }),
                    );
                    function_call_item(
                        id,
                        call_id,
                        &name,
                        namespace.as_deref(),
                        &arguments,
                        "completed",
                    )
                }
                ToolKind::Custom => {
                    let input = custom_input(&call.arguments);
                    self.emit(
                        &mut bytes,
                        "response.output_item.added",
                        json!({
                            "output_index": output_index,
                            "item": custom_tool_call_item(
                                id.clone(),
                                call_id.clone(),
                                &name,
                                namespace.as_deref(),
                                "",
                                "in_progress",
                            ),
                        }),
                    );
                    self.emit(
                        &mut bytes,
                        "response.custom_tool_call_input.delta",
                        json!({ "item_id": id, "output_index": output_index, "delta": input }),
                    );
                    self.emit(
                        &mut bytes,
                        "response.custom_tool_call_input.done",
                        json!({ "item_id": id, "output_index": output_index, "input": input }),
                    );
                    custom_tool_call_item(
                        id,
                        call_id,
                        &name,
                        namespace.as_deref(),
                        &input,
                        "completed",
                    )
                }
            };
            self.emit(
                &mut bytes,
                "response.output_item.done",
                json!({ "output_index": output_index, "item": item }),
            );
            self.output.push(item);
        }
        let (status, incomplete) = status(completion);
        let output = std::mem::take(&mut self.output);
        let mut response =
            self.snapshot(status, output, usage(completion), incomplete, Value::Null);
        response["timings"] = completion_timings(completion);
        response["hipfire"] = completion_hipfire(completion);
        let name = if status == "incomplete" {
            "response.incomplete"
        } else {
            "response.completed"
        };
        self.emit(&mut bytes, name, json!({ "response": response }));
        bytes
    }

    /// `error` then `response.failed`, ending a stream that fails after it
    /// started.
    pub(crate) fn failure(&mut self, message: &str, status: u16) -> Vec<u8> {
        let code = if (400..500).contains(&status) {
            "invalid_request_error"
        } else {
            "server_error"
        };
        let mut bytes = Vec::new();
        self.emit(
            &mut bytes,
            "error",
            json!({ "code": code, "message": message, "param": Value::Null }),
        );
        let output = std::mem::take(&mut self.output);
        let response = self.snapshot(
            "failed",
            output,
            Value::Null,
            Value::Null,
            json!({ "code": code, "message": message }),
        );
        self.emit(
            &mut bytes,
            "response.failed",
            json!({ "response": response }),
        );
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completion(
        content: &str,
        reasoning: &str,
        calls: Vec<ToolCall>,
        finish: &str,
    ) -> Completion {
        Completion {
            id: "chatcmpl-9-2".into(),
            created: 1_700_000_000,
            model: "qwen3.5:4b".into(),
            content: content.into(),
            reasoning_content: reasoning.into(),
            preserve_thinking: false,
            tool_calls: calls,
            done: json!({
                "finish_reason": finish,
                "prompt_tokens": 20,
                "cached_tokens": 8,
                "tokens": 6,
            }),
            logprobs: None,
            reasoning: None,
        }
    }

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: None,
            name: name.into(),
            arguments,
            rendered_body: None,
        }
    }

    fn events(bytes: &[u8]) -> Vec<(String, Value)> {
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .split("\n\n")
            .filter(|frame| !frame.is_empty())
            .map(|frame| {
                let mut lines = frame.lines();
                let event = lines.next().unwrap().strip_prefix("event: ").unwrap();
                let data = lines.next().unwrap().strip_prefix("data: ").unwrap();
                (event.to_owned(), serde_json::from_str(data).unwrap())
            })
            .collect()
    }

    #[test]
    fn lowers_instructions_string_input_and_sampling() {
        let lowered = lower_request(&json!({
            "model": "qwen3.5:4b",
            "instructions": "Be brief.",
            "input": "hi",
            "max_output_tokens": 64,
            "temperature": 0.3,
            "top_p": 0.9,
            "reasoning": { "effort": "low", "summary": "auto" },
            "store": false,
            "include": ["reasoning.encrypted_content"],
            "stream": true,
        }))
        .unwrap();
        assert!(lowered.stream);
        assert_eq!(
            lowered.chat,
            json!({
                "model": "qwen3.5:4b",
                "max_tokens": 64,
                "messages": [
                    { "role": "system", "content": "Be brief." },
                    { "role": "user", "content": "hi" },
                ],
                "temperature": 0.3,
                "top_p": 0.9,
                "reasoning_effort": "low",
            })
        );
        assert!(crate::serve::is_batch_eligible_request(
            &lowered.chat,
            None,
            Some("qwen3_5"),
            true,
            true
        ));
    }

    #[test]
    fn leading_developer_messages_fold_and_late_system_becomes_user() {
        let lowered = lower_request(&json!({
            "model": "m",
            "instructions": "base",
            "input": [
                { "type": "message", "role": "developer",
                  "content": [{ "type": "input_text", "text": "dev one" }] },
                { "type": "message", "role": "system", "content": "dev two" },
                { "type": "message", "role": "user", "content": "hi" },
                { "type": "message", "role": "system", "content": "late" },
                { "type": "message", "role": "developer", "content": "late dev" },
                { "type": "message", "role": "user", "content": "again" },
            ],
        }))
        .unwrap();
        assert_eq!(
            lowered.chat["messages"],
            json!([
                { "role": "system", "content": "base\n\ndev one\n\ndev two" },
                { "role": "user", "content": "hi" },
                { "role": "user", "content": "late" },
                { "role": "user", "content": "late dev" },
                { "role": "user", "content": "again" },
            ])
        );
    }

    #[test]
    fn function_call_round_trip_folds_into_one_assistant_turn() {
        let lowered = lower_request(&json!({
            "model": "m",
            "input": [
                { "role": "user", "content": [{ "type": "input_text", "text": "weather in Paris?" }] },
                { "type": "reasoning", "id": "rs_1", "summary": [],
                  "content": [{ "type": "reasoning_text", "text": "need the tool" }] },
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "Checking." }] },
                { "type": "function_call", "call_id": "call_a", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" },
                { "type": "function_call", "call_id": "call_b", "name": "get_time", "arguments": "{}" },
                { "type": "function_call_output", "call_id": "call_a", "output": "18C" },
                { "type": "function_call_output", "call_id": "call_b", "output": [{ "type": "input_text", "text": "noon" }] },
            ],
            "tools": [{ "type": "function", "name": "get_weather", "description": "Weather",
                        "parameters": { "type": "object", "properties": { "city": { "type": "string" } } },
                        "strict": true }],
            "tool_choice": { "type": "function", "name": "get_weather" },
        }))
        .unwrap();
        let chat = &lowered.chat;
        assert_eq!(
            chat["messages"],
            json!([
                { "role": "user", "content": "weather in Paris?" },
                { "role": "assistant", "content": "Checking.", "reasoning_content": "need the tool",
                  "tool_calls": [
                      { "id": "call_a", "type": "function", "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" } },
                      { "id": "call_b", "type": "function", "function": { "name": "get_time", "arguments": "{}" } },
                  ] },
                { "role": "tool", "tool_call_id": "call_a", "content": "18C" },
                { "role": "tool", "tool_call_id": "call_b", "content": "noon" },
            ])
        );
        assert_eq!(
            chat["tools"],
            json!([{ "type": "function", "function": { "name": "get_weather", "description": "Weather",
                     "parameters": { "type": "object", "properties": { "city": { "type": "string" } } } } }])
        );
        assert_eq!(
            chat["tool_choice"],
            json!({ "type": "function", "function": { "name": "get_weather" } })
        );
        let normalized =
            crate::serve::complete::normalize_openai_messages(chat.get("messages"), false);
        assert_eq!(
            normalized[1]["tool_calls"][0]["arguments"],
            json!({ "city": "Paris" })
        );
    }

    #[test]
    fn function_call_without_a_message_opens_its_own_assistant_turn() {
        let lowered = lower_request(&json!({ "model": "m", "input": [
            { "role": "user", "content": "x" },
            { "type": "reasoning", "summary": [{ "type": "summary_text", "text": "plan" }] },
            { "type": "function_call", "call_id": "c", "name": "f", "arguments": "{}" },
            { "type": "function_call_output", "call_id": "c", "output": "ok" },
            { "role": "user", "content": "next" },
        ] }))
        .unwrap();
        assert_eq!(
            lowered.chat["messages"],
            json!([
                { "role": "user", "content": "x" },
                { "role": "assistant", "content": "", "reasoning_content": "plan",
                  "tool_calls": [{ "id": "c", "type": "function", "function": { "name": "f", "arguments": "{}" } }] },
                { "role": "tool", "tool_call_id": "c", "content": "ok" },
                { "role": "user", "content": "next" },
            ])
        );
    }

    #[test]
    fn namespace_tools_flatten_and_round_trip() {
        let lowered = lower_request(&json!({
            "model": "m",
            "input": [
                { "role": "user", "content": "go" },
                { "type": "function_call", "call_id": "c1", "namespace": "multi_agent_v1",
                  "name": "spawn_agent", "arguments": "{\"task\":\"x\"}" },
                { "type": "function_call_output", "call_id": "c1", "output": "ok" },
            ],
            "tools": [
                { "type": "function", "name": "exec_command", "parameters": {} },
                { "type": "namespace", "name": "multi_agent_v1",
                  "description": "Tools for spawning and managing sub-agents.",
                  "tools": [
                      { "type": "function", "name": "spawn_agent", "description": "Spawn",
                        "parameters": { "type": "object", "properties": {} } },
                      { "type": "function", "name": "wait_agent" },
                  ] },
                { "type": "web_search", "external_web_access": true },
            ],
            "tool_choice": { "type": "function", "namespace": "multi_agent_v1", "name": "spawn_agent" },
        }))
        .unwrap();
        assert_eq!(lowered.chat["tools"].as_array().unwrap().len(), 3);
        assert_eq!(
            lowered.chat["tools"][1]["function"]["name"],
            json!("multi_agent_v1.spawn_agent")
        );
        assert_eq!(
            lowered.chat["tools"][1]["function"]["description"],
            json!("Tools for spawning and managing sub-agents.\n\nSpawn")
        );
        assert_eq!(
            lowered.chat["tools"][2]["function"]["name"],
            json!("multi_agent_v1.wait_agent")
        );
        assert_eq!(
            lowered.chat["tools"][2]["function"]["parameters"],
            json!({ "type": "object", "properties": {} })
        );
        assert_eq!(
            lowered.chat["tool_choice"],
            json!({ "type": "function", "function": { "name": "multi_agent_v1.spawn_agent" } })
        );
        assert_eq!(
            lowered.chat["messages"][1]["tool_calls"][0]["function"]["name"],
            json!("multi_agent_v1.spawn_agent")
        );
        assert_eq!(
            lowered.chat["messages"][2],
            json!({ "role": "tool", "tool_call_id": "c1", "content": "ok" })
        );

        // Both spellings resolve; output goes back namespaced.
        let done = completion(
            "",
            "",
            vec![
                call("multi_agent_v1.spawn_agent", json!({ "task": "x" })),
                call("multi_agent_v1__wait_agent", json!({})),
            ],
            "tool_calls",
        );
        let response = response_json(&done, &lowered.echo);
        assert_eq!(response["output"][0]["type"], json!("function_call"));
        assert_eq!(response["output"][0]["id"], json!("fc_9-2_0"));
        assert_eq!(response["output"][0]["name"], json!("spawn_agent"));
        assert_eq!(response["output"][0]["namespace"], json!("multi_agent_v1"));
        assert_eq!(response["output"][1]["id"], json!("fc_9-2_1"));
        assert_eq!(response["output"][1]["name"], json!("wait_agent"));
        assert_eq!(response["output"][1]["namespace"], json!("multi_agent_v1"));
        assert_eq!(response["output"][1]["arguments"], json!("{}"));
    }

    #[test]
    fn custom_tools_lower_and_round_trip() {
        let lowered = lower_request(&json!({
            "model": "m",
            "input": [
                { "role": "user", "content": "patch" },
                { "type": "custom_tool_call", "call_id": "ctc_1", "name": "apply_patch",
                  "input": "*** Begin Patch\n*** End Patch" },
                { "type": "custom_tool_call_output", "call_id": "ctc_1", "output": "done" },
            ],
            "tools": [
                { "type": "custom", "name": "apply_patch", "description": "Apply a patch",
                  "format": { "type": "grammar", "syntax": "lark", "definition": "start: begin_patch" } },
            ],
        }))
        .unwrap();
        assert_eq!(
            lowered.chat["tools"][0]["function"]["name"],
            json!("apply_patch")
        );
        let description = lowered.chat["tools"][0]["function"]["description"]
            .as_str()
            .unwrap();
        assert!(
            description.contains("The input must follow this lark grammar:\nstart: begin_patch"),
            "{description}"
        );
        assert_eq!(
            lowered.chat["tools"][0]["function"]["parameters"],
            json!({
                "type": "object",
                "properties": { "input": { "type": "string", "description": "the raw input" } },
                "required": ["input"],
            })
        );
        assert_eq!(
            lowered.chat["messages"][1]["tool_calls"][0]["function"]["arguments"],
            json!("{\"input\":\"*** Begin Patch\\n*** End Patch\"}")
        );
        assert_eq!(
            lowered.chat["messages"][2],
            json!({ "role": "tool", "tool_call_id": "ctc_1", "content": "done" })
        );

        let done = completion(
            "",
            "",
            vec![call(
                "apply_patch",
                json!({ "input": "*** Begin Patch\n*** End Patch" }),
            )],
            "tool_calls",
        );
        let response = response_json(&done, &lowered.echo);
        assert_eq!(response["output"][0]["type"], json!("custom_tool_call"));
        assert_eq!(response["output"][0]["id"], json!("ctc_9-2_0"));
        assert_eq!(response["output"][0]["name"], json!("apply_patch"));
        assert_eq!(
            response["output"][0]["input"],
            json!("*** Begin Patch\n*** End Patch")
        );
    }

    #[test]
    fn custom_call_streams_its_input_and_additional_tools_merge() {
        let lowered = lower_request(&json!({
            "model": "m",
            "input": [
                { "role": "user", "content": "x" },
                { "type": "additional_tools", "tools": [
                    { "type": "custom", "name": "apply_patch" },
                    { "type": "namespace", "name": "ns", "tools": [
                        { "type": "function", "name": "f" },
                    ] },
                ] },
            ],
            "tools": [{ "type": "function", "name": "early", "parameters": {} }],
        }))
        .unwrap();
        assert_eq!(lowered.chat["tools"].as_array().unwrap().len(), 3);
        assert_eq!(lowered.chat["tools"][0]["function"]["name"], json!("early"));
        assert_eq!(
            lowered.chat["tools"][1]["function"]["name"],
            json!("apply_patch")
        );
        assert_eq!(lowered.chat["tools"][2]["function"]["name"], json!("ns.f"));
        assert_eq!(lowered.chat["messages"].as_array().unwrap().len(), 1);

        let none = lower_request(&json!({
            "model": "m", "input": "x", "tool_choice": "none",
            "tools": [{ "type": "function", "name": "f" }],
        }))
        .unwrap();
        assert!(none.chat.get("tools").is_none());

        let mut stream = ResponseStream::new(lowered.echo, "chatcmpl-9-2", 1_700_000_000, "m");
        let mut bytes = stream.preamble();
        bytes.extend(stream.terminal(&completion(
            "",
            "",
            vec![call("apply_patch", json!({ "input": "patch" }))],
            "tool_calls",
        )));
        let events = events(&bytes);
        let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.custom_tool_call_input.delta",
                "response.custom_tool_call_input.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_eq!(events[2].1["item"]["type"], json!("custom_tool_call"));
        assert_eq!(events[3].1["delta"], json!("patch"));
        assert_eq!(events[4].1["input"], json!("patch"));
        assert_eq!(events[5].1["item"]["input"], json!("patch"));
    }

    #[test]
    fn text_format_and_choice_modes_lower() {
        let base = json!({ "model": "m", "input": "x" });
        let with = |key: &str, value: Value| {
            let mut body = base.clone();
            body[key] = value;
            lower_request(&body)
        };
        let schema = json!({ "type": "object", "properties": { "a": { "type": "string" } } });
        assert_eq!(
            with("text", json!({ "format": { "type": "json_schema", "name": "r", "schema": schema, "strict": true } }))
                .unwrap()
                .chat["response_format"],
            json!({ "type": "json_schema", "json_schema": { "name": "r", "schema": schema, "strict": true } })
        );
        assert!(with("text", json!({ "format": { "type": "text" } }))
            .unwrap()
            .chat
            .get("response_format")
            .is_none());
        for mode in ["auto", "none", "required"] {
            assert_eq!(
                with("tool_choice", json!(mode)).unwrap().chat["tool_choice"],
                json!(mode)
            );
        }
        assert!(with("tool_choice", json!({ "type": "allowed_tools" })).is_err());
    }

    #[test]
    fn lowering_refuses_stored_state_and_hosted_tools() {
        let cases = [
            (
                json!({ "model": "m", "input": "x", "previous_response_id": "resp_1" }),
                "previous_response_id",
            ),
            (json!({ "model": "m" }), "input"),
            (
                json!({ "model": "m", "input": [{ "type": "item_reference", "id": "x" }] }),
                "item_reference",
            ),
            (
                json!({ "model": "m", "input": "x", "background": true }),
                "background",
            ),
            (
                json!({ "model": "m", "input": [{ "role": "user", "content": [{ "type": "input_file", "file_id": "f" }] }] }),
                "input_file",
            ),
        ];
        for (body, needle) in cases {
            let error = lower_request(&body).expect_err(needle);
            assert!(error.contains(needle), "{needle}: {error}");
        }
    }

    #[test]
    fn response_json_renders_items_usage_and_echo() {
        let lowered = lower_request(&json!({
            "model": "m", "input": "x", "instructions": "sys", "max_output_tokens": 50,
            "tools": [{ "type": "function", "name": "get_weather", "parameters": {} }],
        }))
        .unwrap();
        let done = completion(
            "Checking.",
            "need the tool",
            vec![call("get_weather", json!({ "city": "Paris" }))],
            "tool_calls",
        );
        let response = response_json(&done, &lowered.echo);
        assert_eq!(response["id"], json!("resp_9-2"));
        assert_eq!(response["object"], json!("response"));
        assert_eq!(response["status"], json!("completed"));
        assert_eq!(response["instructions"], json!("sys"));
        assert_eq!(response["max_output_tokens"], json!(50));
        assert_eq!(response["store"], json!(false));
        assert_eq!(
            response["output"],
            json!([
                { "type": "reasoning", "id": "rs_9-2_0", "summary": [],
                  "content": [{ "type": "reasoning_text", "text": "need the tool" }] },
                { "type": "message", "id": "msg_9-2_1", "status": "completed", "role": "assistant",
                  "content": [{ "type": "output_text", "text": "Checking.", "annotations": [], "logprobs": [] }] },
                { "type": "function_call", "id": "fc_9-2_2", "call_id": "call_9-2_0", "name": "get_weather",
                  "arguments": "{\"city\":\"Paris\"}", "status": "completed" },
            ])
        );
        assert_eq!(
            response["usage"],
            json!({ "input_tokens": 20, "input_tokens_details": { "cached_tokens": 8 },
                    "output_tokens": 6, "total_tokens": 26 })
        );

        // The rendered output goes back in as the same assistant turn.
        let mut input = vec![json!({ "role": "user", "content": "x" })];
        input.extend(response["output"].as_array().unwrap().iter().cloned());
        input.push(
            json!({ "type": "function_call_output", "call_id": "call_9-2_0", "output": "18C" }),
        );
        let next = lower_request(&json!({ "model": "m", "input": input })).unwrap();
        assert_eq!(
            next.chat["messages"][1],
            json!({ "role": "assistant", "content": "Checking.", "reasoning_content": "need the tool",
                    "tool_calls": [{ "id": "call_9-2_0", "type": "function",
                                     "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" } }] })
        );
        assert_eq!(
            next.chat["messages"][2]["tool_call_id"],
            json!("call_9-2_0")
        );
    }

    #[test]
    fn length_is_incomplete_and_hides_unreleased_calls() {
        let echo = lower_request(&json!({ "model": "m", "input": "x" }))
            .unwrap()
            .echo;
        let response = response_json(
            &completion("part", "", vec![call("f", json!({}))], "length"),
            &echo,
        );
        assert_eq!(response["status"], json!("incomplete"));
        assert_eq!(
            response["incomplete_details"],
            json!({ "reason": "max_output_tokens" })
        );
        assert_eq!(response["output"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn stream_event_order_and_sequence_numbers() {
        let echo = lower_request(&json!({ "model": "m", "input": "x" }))
            .unwrap()
            .echo;
        let mut stream = ResponseStream::new(echo, "chatcmpl-9-2", 1_700_000_000, "qwen3.5:4b");
        let mut bytes = stream.preamble();
        for event in [
            json!({ "type": "reasoning", "text": "need " }),
            json!({ "type": "reasoning", "text": "the tool" }),
            json!({ "type": "token", "text": "Checking." }),
            json!({ "type": "token", "text": "" }),
        ] {
            if let Some(chunk) = stream.event(&event) {
                bytes.extend(chunk);
            }
        }
        bytes.extend(stream.terminal(&completion(
            "Checking.",
            "need the tool",
            vec![call("get_weather", json!({ "city": "Paris" }))],
            "tool_calls",
        )));
        let events = events(&bytes);
        let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        for (seq, (name, data)) in events.iter().enumerate() {
            assert_eq!(data["type"], json!(name));
            assert_eq!(data["sequence_number"], json!(seq));
        }
        assert_eq!(events[0].1["response"]["status"], json!("in_progress"));
        assert_eq!(events[5].1["text"], json!("need the tool"));
        assert_eq!(events[9].1["item_id"], json!("msg_9-2_1"));
        assert_eq!(events[9].1["output_index"], json!(1));
        assert_eq!(events[10].1["text"], json!("Checking."));
        assert_eq!(events[13].1["item"]["arguments"], json!(""));
        assert_eq!(events[15].1["arguments"], json!("{\"city\":\"Paris\"}"));
        let completed = &events[17].1["response"];
        assert_eq!(completed["status"], json!("completed"));
        let output = completed["output"].as_array().unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output[0]["content"][0]["text"], json!("need the tool"));
        assert_eq!(output[1]["content"][0]["text"], json!("Checking."));
        assert_eq!(output[2]["call_id"], json!("call_9-2_0"));
        assert_eq!(completed["usage"]["output_tokens"], json!(6));
    }

    #[test]
    fn stream_failure_emits_error_then_failed() {
        let echo = lower_request(&json!({ "model": "m", "input": "x" }))
            .unwrap()
            .echo;
        let mut stream = ResponseStream::new(echo, "chatcmpl-1-1", 0, "m");
        let _ = stream.preamble();
        let events = events(&stream.failure("boom", 500));
        assert_eq!(events[0].0, "error");
        assert_eq!(events[0].1["message"], json!("boom"));
        assert_eq!(events[1].0, "response.failed");
        assert_eq!(events[1].1["response"]["status"], json!("failed"));
        assert_eq!(
            events[1].1["response"]["error"]["code"],
            json!("server_error")
        );
        assert_eq!(events[1].1["sequence_number"], json!(3));
    }
}
