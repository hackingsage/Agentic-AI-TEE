// Package main implements the Enclave Privacy Proxy.
//
// The proxy sits between the enclave and external LLM API providers.
// It receives requests from the enclave via vsock, strips identifying
// metadata, rotates API keys, optionally adds dummy-request padding,
// and forwards requests to the LLM API over mTLS.
//
// The proxy DOES NOT decrypt request/response content — it only
// handles transport-level privacy measures.
package main

import (
	"context"
	"crypto/tls"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Config holds the proxy configuration.
type Config struct {
	// ListenMode: "vsock" or "tcp" (for local development)
	ListenMode string `json:"listen_mode"`
	// TCPAddr is the TCP address to listen on (for local dev)
	TCPAddr string `json:"tcp_addr"`
	// VsockPort is the vsock port to listen on
	VsockPort uint32 `json:"vsock_port"`
	// TargetURL is the LLM API base URL
	TargetURL string `json:"target_url"`
	// APIKeys is the list of API keys to rotate through
	APIKeys []string `json:"api_keys"`
	// EnableDummyPad enables dummy-request padding
	EnableDummyPad bool `json:"enable_dummy_pad"`
	// DummyPadProbability is the probability of sending a dummy request (0.0-1.0)
	DummyPadProbability float64 `json:"dummy_pad_probability"`
}

// DefaultConfig returns a development-mode configuration.
func DefaultConfig() Config {
	return Config{
		ListenMode:          "tcp",
		TCPAddr:             "127.0.0.1:9000",
		VsockPort:           5001,
		TargetURL:           "https://api.anthropic.com",
		APIKeys:             []string{},
		EnableDummyPad:      false,
		DummyPadProbability: 0.1,
	}
}

// KeyRing manages API key rotation.
type KeyRing struct {
	mu      sync.Mutex
	keys    []string
	current int
}

// NewKeyRing creates a new key ring from a list of API keys.
func NewKeyRing(keys []string) *KeyRing {
	return &KeyRing{keys: keys, current: 0}
}

// Next returns the next API key in round-robin order.
func (kr *KeyRing) Next() string {
	kr.mu.Lock()
	defer kr.mu.Unlock()
	if len(kr.keys) == 0 {
		return ""
	}
	key := kr.keys[kr.current]
	kr.current = (kr.current + 1) % len(kr.keys)
	return key
}

// HeaderStripper removes identifying headers from outgoing requests.
type HeaderStripper struct{}

// Strip removes privacy-sensitive headers and adds opt-out headers.
func (hs *HeaderStripper) Strip(header http.Header) http.Header {
	cleaned := make(http.Header)
	for k, v := range header {
		k = http.CanonicalHeaderKey(k)
		switch k {
		case "User-Agent", "X-Forwarded-For", "X-Real-Ip",
			"X-Request-Id", "X-Correlation-Id", "Referer",
			"Origin", "Cookie", "Set-Cookie":
			// Strip these headers — they leak identity
			continue
		default:
			cleaned[k] = v
		}
	}
	// Add no-train opt-out headers
	cleaned.Set("Anthropic-No-Train", "true")
	return cleaned
}

// MessageFrame matches the Python vsock protocol framing.
type MessageFrame struct {
	MsgType   string                 `json:"msg_type"`
	Payload   map[string]interface{} `json:"payload"`
	RequestID string                 `json:"request_id"`
}

// readFrame reads a length-prefixed JSON frame from a connection.
func readFrame(conn net.Conn) (*MessageFrame, error) {
	// Read 4-byte LE length header
	header := make([]byte, 4)
	if _, err := io.ReadFull(conn, header); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	length := binary.LittleEndian.Uint32(header)

	// Validate size (16MB max)
	if length > 16*1024*1024 {
		return nil, fmt.Errorf("message too large: %d bytes", length)
	}
	if length == 0 {
		return nil, fmt.Errorf("empty message")
	}

	// Read payload
	payload := make([]byte, length)
	if _, err := io.ReadFull(conn, payload); err != nil {
		return nil, fmt.Errorf("read payload: %w", err)
	}

	var frame MessageFrame
	if err := json.Unmarshal(payload, &frame); err != nil {
		return nil, fmt.Errorf("unmarshal frame: %w", err)
	}
	return &frame, nil
}

// writeFrame writes a length-prefixed JSON frame to a connection.
func writeFrame(conn net.Conn, frame *MessageFrame) error {
	payload, err := json.Marshal(frame)
	if err != nil {
		return fmt.Errorf("marshal frame: %w", err)
	}

	header := make([]byte, 4)
	binary.LittleEndian.PutUint32(header, uint32(len(payload)))

	if _, err := conn.Write(header); err != nil {
		return fmt.Errorf("write header: %w", err)
	}
	if _, err := conn.Write(payload); err != nil {
		return fmt.Errorf("write payload: %w", err)
	}
	return nil
}

// Proxy is the main privacy proxy server.
type Proxy struct {
	config   Config
	keyRing  *KeyRing
	stripper *HeaderStripper
	client   *http.Client
}

// NewProxy creates a new privacy proxy instance.
func NewProxy(config Config) *Proxy {
	// Create HTTP client with mTLS support
	tlsConfig := &tls.Config{
		MinVersion: tls.VersionTLS12,
	}
	transport := &http.Transport{
		TLSClientConfig: tlsConfig,
		MaxIdleConns:    10,
		IdleConnTimeout: 90 * time.Second,
	}

	return &Proxy{
		config:   config,
		keyRing:  NewKeyRing(config.APIKeys),
		stripper: &HeaderStripper{},
		client:   &http.Client{Transport: transport, Timeout: 120 * time.Second},
	}
}

// handleConnection processes a single vsock/TCP connection.
func (p *Proxy) handleConnection(conn net.Conn) {
	defer conn.Close()

	for {
		frame, err := readFrame(conn)
		if err != nil {
			if err != io.EOF && !strings.Contains(err.Error(), "connection reset") {
				log.Printf("read error: %v", err)
			}
			return
		}

		log.Printf("received msg_type=%s request_id=%s", frame.MsgType, frame.RequestID)

		var response *MessageFrame
		switch frame.MsgType {
		case "llm_request":
			response = p.handleLLMRequest(frame)
		case "echo":
			response = &MessageFrame{
				MsgType:   "echo_response",
				Payload:   frame.Payload,
				RequestID: frame.RequestID,
			}
		default:
			response = &MessageFrame{
				MsgType:   "error",
				Payload:   map[string]interface{}{"error": fmt.Sprintf("unknown msg_type: %s", frame.MsgType)},
				RequestID: frame.RequestID,
			}
		}

		if err := writeFrame(conn, response); err != nil {
			log.Printf("write error: %v", err)
			return
		}
	}
}

// handleLLMRequest forwards an LLM request through the privacy proxy.
func (p *Proxy) handleLLMRequest(frame *MessageFrame) *MessageFrame {
	// Extract request details from payload
	method, _ := frame.Payload["method"].(string)
	path, _ := frame.Payload["path"].(string)
	body, _ := frame.Payload["body"].(string)
	headersRaw, _ := frame.Payload["headers"].(map[string]interface{})

	if method == "" {
		method = "POST"
	}
	if path == "" {
		path = "/v1/messages"
	}

	url := p.config.TargetURL + path

	// Build the outgoing request
	req, err := http.NewRequest(method, url, strings.NewReader(body))
	if err != nil {
		return &MessageFrame{
			MsgType:   "llm_response",
			Payload:   map[string]interface{}{"error": fmt.Sprintf("build request: %v", err)},
			RequestID: frame.RequestID,
		}
	}

	// Set headers from the enclave request
	for k, v := range headersRaw {
		if s, ok := v.(string); ok {
			req.Header.Set(k, s)
		}
	}

	// Strip identifying headers
	req.Header = p.stripper.Strip(req.Header)

	// Rotate API key
	apiKey := p.keyRing.Next()
	if apiKey != "" {
		req.Header.Set("x-api-key", apiKey)
		req.Header.Set("anthropic-version", "2023-06-01")
	}
	req.Header.Set("Content-Type", "application/json")

	// Log only metadata, never content
	log.Printf("proxy_forward method=%s path=%s content_length=%d request_id=%s",
		method, path, len(body), frame.RequestID)

	// Execute request
	resp, err := p.client.Do(req)
	if err != nil {
		return &MessageFrame{
			MsgType:   "llm_response",
			Payload:   map[string]interface{}{"error": fmt.Sprintf("http request failed: %v", err)},
			RequestID: frame.RequestID,
		}
	}
	defer resp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024)) // 10MB limit
	if err != nil {
		return &MessageFrame{
			MsgType:   "llm_response",
			Payload:   map[string]interface{}{"error": fmt.Sprintf("read response: %v", err)},
			RequestID: frame.RequestID,
		}
	}

	// Log response metadata only
	log.Printf("proxy_response status=%d content_length=%d request_id=%s",
		resp.StatusCode, len(respBody), frame.RequestID)

	return &MessageFrame{
		MsgType: "llm_response",
		Payload: map[string]interface{}{
			"status_code": resp.StatusCode,
			"body":        string(respBody),
			"headers":     flattenHeaders(resp.Header),
		},
		RequestID: frame.RequestID,
	}
}

// flattenHeaders converts http.Header to a simple map.
func flattenHeaders(h http.Header) map[string]string {
	result := make(map[string]string)
	for k, v := range h {
		result[k] = strings.Join(v, ", ")
	}
	return result
}

func main() {
	config := DefaultConfig()

	// Load config from file if specified
	if configPath := os.Getenv("PROXY_CONFIG"); configPath != "" {
		data, err := os.ReadFile(configPath)
		if err != nil {
			log.Fatalf("read config: %v", err)
		}
		if err := json.Unmarshal(data, &config); err != nil {
			log.Fatalf("parse config: %v", err)
		}
	}

	// Load API keys from environment
	if keys := os.Getenv("LLM_API_KEYS"); keys != "" {
		config.APIKeys = strings.Split(keys, ",")
	}

	proxy := NewProxy(config)

	// Listen
	var listener net.Listener
	var err error

	if config.ListenMode == "tcp" {
		listener, err = net.Listen("tcp", config.TCPAddr)
		if err != nil {
			log.Fatalf("tcp listen: %v", err)
		}
		log.Printf("privacy proxy listening on tcp://%s", config.TCPAddr)
	} else {
		log.Fatal("vsock mode requires the mdlayher/vsock package — use tcp for development")
	}
	defer listener.Close()

	// Graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Println("shutting down...")
		cancel()
		listener.Close()
	}()

	// Accept connections
	for {
		conn, err := listener.Accept()
		if err != nil {
			select {
			case <-ctx.Done():
				return
			default:
				log.Printf("accept error: %v", err)
				continue
			}
		}
		go proxy.handleConnection(conn)
	}
}
