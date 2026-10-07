//! A dropped MCP call must notify the peer, not merely stop waiting locally.
//! Reproduces openai/codex#26956 using the real RMCP byte-stream lifecycle.

use super::ElicitationResponse;
use super::RmcpClient;
use crate::InProcessTransportFactory;
use crate::McpProtocolMode;
use futures::future::BoxFuture;
use pretty_assertions::assert_eq;
use rmcp::ErrorData;
use rmcp::RoleServer;
use rmcp::ServerHandler;
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::model::CallToolResponse;
use rmcp::model::CallToolResult;
use rmcp::model::CancelledNotificationParam;
use rmcp::model::ClientInfo;
use rmcp::model::ElicitationAction;
use rmcp::model::InputRequiredResult;
use rmcp::model::RequestId;
use rmcp::model::ServerCapabilities;
use rmcp::model::ServerInfo;
use rmcp::service::NotificationContext;
use rmcp::service::RequestContext;
use rmcp::service::RunningService;
use rmcp::service::ServerInitializeError;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::time::timeout;

const DEADLINE: Duration = Duration::from_secs(5);
const MODES: [McpProtocolMode; 2] = [McpProtocolMode::Legacy, McpProtocolMode::V20260728];

struct PendingCall {
    id: RequestId,
    reply: oneshot::Sender<Result<CallToolResponse, ErrorData>>,
}

#[derive(Clone)]
struct TestServer {
    started: mpsc::UnboundedSender<PendingCall>,
    cancelled: mpsc::UnboundedSender<Option<RequestId>>,
}

impl ServerHandler for TestServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn call_tool(
        &self,
        _request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let (reply, response) = oneshot::channel();
        self.started
            .send(PendingCall {
                id: context.id,
                reply,
            })
            .expect("test owns the request receiver");
        tokio::select! {
            result = response => result.expect("test resolves the request"),
            () = context.ct.cancelled() => Err(ErrorData::internal_error("cancelled", None)),
        }
    }

    async fn on_cancelled(
        &self,
        notification: CancelledNotificationParam,
        _context: NotificationContext<RoleServer>,
    ) {
        self.cancelled.send(notification.request_id).unwrap();
    }
}

struct TestTransport {
    server: TestServer,
    connections: mpsc::UnboundedSender<
        Result<RunningService<RoleServer, TestServer>, ServerInitializeError>,
    >,
}

impl InProcessTransportFactory for TestTransport {
    fn open(&self) -> BoxFuture<'static, std::io::Result<tokio::io::DuplexStream>> {
        let server = self.server.clone();
        let connections = self.connections.clone();
        Box::pin(async move {
            let (client, peer) = tokio::io::duplex(/*max_buf_size*/ 4096);
            tokio::spawn(async move {
                let _ = connections.send(server.serve(peer).await);
            });
            Ok(client)
        })
    }
}

struct Probe {
    client: Arc<RmcpClient>,
    server: RunningService<RoleServer, TestServer>,
    started: mpsc::UnboundedReceiver<PendingCall>,
    cancelled: mpsc::UnboundedReceiver<Option<RequestId>>,
}

impl Probe {
    async fn new(mode: McpProtocolMode) -> anyhow::Result<Self> {
        let (started, started_rx) = mpsc::unbounded_channel();
        let (cancelled, cancelled_rx) = mpsc::unbounded_channel();
        let (connections, mut connection_rx) = mpsc::unbounded_channel();
        let mut client = RmcpClient::new_in_process_client(Arc::new(TestTransport {
            server: TestServer { started, cancelled },
            connections,
        }))
        .await?;
        client.protocol_mode = mode;
        let info = client
            .initialize(
                ClientInfo::default().with_protocol_version(mode.preferred_protocol_version()),
                Some(DEADLINE),
                Box::new(|_, _| {
                    Box::pin(async {
                        Ok(ElicitationResponse {
                            action: ElicitationAction::Accept,
                            content: None,
                            meta: None,
                        })
                    })
                }),
            )
            .await?;
        assert_eq!(info.protocol_version, mode.preferred_protocol_version());
        let server = timeout(DEADLINE, connection_rx.recv()).await?.unwrap()?;
        Ok(Self {
            client: Arc::new(client),
            server,
            started: started_rx,
            cancelled: cancelled_rx,
        })
    }

    fn call(
        &self,
        deadline: Option<Duration>,
    ) -> tokio::task::JoinHandle<anyhow::Result<CallToolResult>> {
        let client = Arc::clone(&self.client);
        tokio::spawn(async move { client.call_tool("hold".into(), None, None, deadline).await })
    }

    async fn finish(self) -> anyhow::Result<()> {
        self.client.shutdown().await;
        self.server.cancel().await?;
        Ok(())
    }
}

#[tokio::test]
async fn aborted_tool_call_notifies_only_its_request_and_allows_recovery() -> anyhow::Result<()> {
    for mode in MODES {
        let mut probe = Probe::new(mode).await?;
        let interrupted = probe.call(None);
        let first = timeout(DEADLINE, probe.started.recv()).await?.unwrap();
        let surviving = probe.call(None);
        let second = timeout(DEADLINE, probe.started.recv()).await?.unwrap();
        assert_ne!(first.id, second.id);

        interrupted.abort();
        assert!(interrupted.await.unwrap_err().is_cancelled());
        assert_eq!(
            timeout(DEADLINE, probe.cancelled.recv()).await?,
            Some(Some(first.id))
        );
        assert!(
            !second.reply.is_closed(),
            "cancellation must not close another request"
        );
        let mut result = CallToolResult::success(Vec::new());
        if mode == McpProtocolMode::Legacy {
            result.result_type = None;
        }
        second.reply.send(Ok(result.clone().into())).unwrap();
        assert_eq!(timeout(DEADLINE, surviving).await???, result);

        let recovered = probe.call(None);
        let recovery = timeout(DEADLINE, probe.started.recv()).await?.unwrap();
        recovery.reply.send(Ok(result.clone().into())).unwrap();
        assert_eq!(timeout(DEADLINE, recovered).await???, result);
        assert!(
            timeout(Duration::from_millis(100), probe.cancelled.recv())
                .await
                .is_err()
        );
        probe.finish().await?;
    }
    Ok(())
}

#[tokio::test]
async fn timed_out_tool_call_notifies_the_peer() -> anyhow::Result<()> {
    for mode in MODES {
        let mut probe = Probe::new(mode).await?;
        let call = probe.call(Some(Duration::from_secs(1)));
        let request = timeout(DEADLINE, probe.started.recv()).await?.unwrap();
        let error = timeout(DEADLINE, call).await??.unwrap_err();
        assert_eq!(
            error.to_string(),
            "timed out awaiting tools/call after 1000ms"
        );
        assert_eq!(
            timeout(DEADLINE, probe.cancelled.recv()).await?,
            Some(Some(request.id))
        );
        probe.finish().await?;
    }
    Ok(())
}

#[tokio::test]
async fn completed_and_failed_tool_calls_do_not_send_cancellation() -> anyhow::Result<()> {
    for mode in MODES {
        let mut probe = Probe::new(mode).await?;
        for response in [
            Ok(CallToolResult::success(Vec::new())),
            Err(ErrorData::internal_error("test failure", None)),
        ] {
            let expected = response.clone();
            let call = probe.call(None);
            let request = timeout(DEADLINE, probe.started.recv()).await?.unwrap();
            request.reply.send(response.map(Into::into)).unwrap();
            let result = timeout(DEADLINE, call).await??;
            match expected {
                Ok(mut expected) => {
                    if mode == McpProtocolMode::Legacy {
                        expected.result_type = None;
                    }
                    assert_eq!(result?, expected);
                }
                Err(_) => assert!(result.unwrap_err().to_string().contains("test failure")),
            }
            assert!(
                timeout(Duration::from_millis(100), probe.cancelled.recv())
                    .await
                    .is_err()
            );
        }
        probe.finish().await?;
    }
    Ok(())
}

#[tokio::test]
async fn cancelled_continuation_notifies_the_current_request_not_the_previous_round()
-> anyhow::Result<()> {
    let mut probe = Probe::new(McpProtocolMode::V20260728).await?;
    let call = probe.call(None);
    let first = timeout(DEADLINE, probe.started.recv()).await?.unwrap();
    first
        .reply
        .send(Ok(InputRequiredResult::from_request_state("next").into()))
        .unwrap();
    let second = timeout(DEADLINE, probe.started.recv()).await?.unwrap();
    assert_ne!(first.id, second.id);
    call.abort();
    assert!(call.await.unwrap_err().is_cancelled());
    assert_eq!(
        timeout(DEADLINE, probe.cancelled.recv()).await?,
        Some(Some(second.id))
    );
    assert!(
        timeout(Duration::from_millis(100), probe.cancelled.recv())
            .await
            .is_err()
    );
    probe.finish().await?;
    Ok(())
}
