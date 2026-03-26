"""
Server-Sent Events (SSE) Manager for Real-time Updates

This module provides Server-Sent Events functionality for:
- Real-time alert notifications
- Camera status updates
- System health monitoring
"""

import json
import time
import threading
import queue
from collections import defaultdict
from flask import Response, stream_with_context


class SSEManager:
    """Server-Sent Events Manager for real-time client notifications"""
    
    def __init__(self):
        # Store queues for each client identifier
        # key: client_id, value: { 'queue': Queue, 'event_types': set }
        self._clients = {}
        self._lock = threading.Lock()
        
    def add_client(self, client_id, event_types):
        """Add a client with its subscribed event types"""
        with self._lock:
            if client_id not in self._clients:
                self._clients[client_id] = {
                    'queue': queue.Queue(maxsize=100),
                    'event_types': set(event_types)
                }
                print(f"[SSE] Client {client_id} added with subscriptions: {event_types}")
            else:
                # Update existing client subscriptions
                self._clients[client_id]['event_types'].update(event_types)
                print(f"[SSE] Client {client_id} updated subscriptions: {self._clients[client_id]['event_types']}")
        return self._clients[client_id]['queue']
                
    def remove_client(self, client_id):
        """Remove a client from all event types"""
        with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                print(f"[SSE] Client {client_id} removed from all subscriptions")
                
    def emit_event(self, event_type, data):
        """Emit an event to all clients subscribed to that event type"""
        message = generate_sse_message(event_type, data)
        count = 0
        
        with self._lock:
            # We iterate over a copy of the items to avoid issues if a client is removed during iteration
            for client_id, info in list(self._clients.items()):
                if event_type in info['event_types'] or 'all' in info['event_types']:
                    try:
                        # Non-blocking put, if queue is full, skip this client
                        info['queue'].put_nowait(message)
                        count += 1
                    except queue.Full:
                        print(f"[SSE] Queue full for client {client_id}, dropping event")
        
        if count > 0:
            print(f"[SSE] Emitted {event_type} to {count} clients")
        
        return count
    
    def get_client_count(self):
        """Get total number of connected clients"""
        with self._lock:
            return len(self._clients)
    
    def get_all_subscriptions(self):
        """Get all event types and their client counts"""
        stats = defaultdict(int)
        with self._lock:
            for info in self._clients.values():
                for et in info['event_types']:
                    stats[et] += 1
        return dict(stats)


# Global SSE manager instance
sse_manager = SSEManager()


def generate_sse_message(event_type, data, event_id=None):
    """Generate SSE formatted message"""
    message = f"event: {event_type}\n"
    message += f"data: {json.dumps(data)}\n"
    if event_id:
        message += f"id: {event_id}\n"
    message += "\n"
    return message


def create_sse_response(event_types, client_id):
    """
    Create a streaming SSE response for specified event types
    """
    # Create or get queue for this client
    client_queue = sse_manager.add_client(client_id, event_types)
    
    def event_stream():
        try:
            # Send initial connection message
            yield generate_sse_message('connected', {
                'status': 'connected',
                'client_id': client_id,
                'subscriptions': event_types,
                'timestamp': time.time()
            })
            
            # Keep connection alive with heartbeat
            heartbeat_interval = 30  # seconds
            last_heartbeat = time.time()
            
            while True:
                # Check for new events in the queue
                try:
                    # Wait for a message from the queue with a timeout
                    # The timeout allows us to send heartbeats
                    message = client_queue.get(timeout=1.0)
                    yield message
                except queue.Empty:
                    # No message, check if it's time for a heartbeat
                    if time.time() - last_heartbeat >= heartbeat_interval:
                        yield generate_sse_message('heartbeat', {
                            'timestamp': time.time()
                        })
                        last_heartbeat = time.time()
                except Exception as e:
                    print(f"[SSE] Error in event stream for {client_id}: {e}")
                    break
                    
        finally:
            # Ensure client is removed when connection closes
            sse_manager.remove_client(client_id)
            print(f"[SSE] Connection closed for client {client_id}")
                
    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


def emit_alert(alert_data):
    """Emit an alert event to all connected clients"""
    return sse_manager.emit_event('alert', {
        'type': 'alert',
        'data': alert_data,
        'timestamp': time.time()
    })


def emit_camera_status(camera_name, status):
    """Emit a camera status update event"""
    return sse_manager.emit_event('camera_status', {
        'type': 'camera_status',
        'camera_name': camera_name,
        'status': status,
        'timestamp': time.time()
    })


def emit_detection(camera_name, detection_type, confidence):
    """Emit a detection event when anomaly is detected"""
    return sse_manager.emit_event('detection', {
        'type': 'detection',
        'camera_name': camera_name,
        'detection_type': detection_type,
        'confidence': confidence,
        'timestamp': time.time()
    })


def emit_system_health(status):
    """Emit system health status"""
    return sse_manager.emit_event('health', {
        'type': 'health',
        'status': status,
        'subscriptions': sse_manager.get_all_subscriptions(),
        'timestamp': time.time()
    })
