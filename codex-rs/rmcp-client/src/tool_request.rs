//! Keeps cancellation attached to the exact MCP request, including continuations.

use std::time::Duration;

use rmcp::model::CancelledNotificationParam;
use rmcp::model::RequestId;
use rmcp::model::ServerResult;
use rmcp::service::Peer;
use rmcp::service::RequestHandle;
use rmcp::service::RoleClient;
use rmcp::service::ServiceError;
use tokio::runtime::Handle;
use tracing::warn;

pub(crate) async fn await_response(
    request: RequestHandle<RoleClient>,
) -> Result<ServerResult, ServiceError> {
    let mut cancellation = CancelOnDrop {
        peer: request.peer.clone(),
        request_id: Some(request.id.clone()),
        runtime: Handle::current(),
    };
    let response = request.await_response().await;
    // Both a successful response and a protocol error resolve the request.
    // Disarm before any further await so a later interrupt cannot cancel it.
    cancellation.request_id = None;
    response
}

struct CancelOnDrop {
    peer: Peer<RoleClient>,
    request_id: Option<RequestId>,
    runtime: Handle,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let Some(request_id) = self.request_id.take() else {
            return;
        };
        let peer = self.peer.clone();
        // Turn interruption aborts the calling task; active-time expiry drops
        // the same future. RMCP does not cancel requests on drop. Let this
        // bounded cleanup survive its caller, while retaining the original
        // connection (request IDs can be reused after reconnecting). Runtime
        // shutdown ends the cleanup too; delivery to a closed peer is best effort.
        self.runtime.spawn(async move {
            let notification = CancelledNotificationParam::new(
                Some(request_id),
                Some("client stopped waiting for tool response".to_string()),
            );
            match tokio::time::timeout(
                Duration::from_secs(/*secs*/ 5),
                peer.notify_cancelled(notification),
            )
            .await
            {
                Ok(Ok(())) => {}
                // Do not log transport errors: they can contain server data.
                Ok(Err(_)) => warn!("Failed to send MCP tool cancellation"),
                Err(_) => warn!("Timed out sending MCP tool cancellation"),
            }
        });
    }
}
