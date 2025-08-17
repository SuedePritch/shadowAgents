
package shadowagents

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// mockTransport is a thread-safe http.RoundTripper that allows us to
// define specific responses for specific requests during a test.
type mockTransport struct {
	mu       sync.Mutex
	handlers map[string]http.HandlerFunc
}

// newMockTransport creates a new mock transport.
func newMockTransport() *mockTransport {
	return &mockTransport{
		handlers: make(map[string]http.HandlerFunc),
	}
}

// RegisterHandler sets a specific handler function for a request body substring.
// When a request comes in, we check if its body contains the key. If so, we use the handler.
func (m *mockTransport) RegisterHandler(requestBodySubstring string, handler http.HandlerFunc) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.handlers[requestBodySubstring] = handler
}

// RoundTrip is the core of the mock. It intercepts the outgoing HTTP request.
func (m *mockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	bodyBytes, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	req.Body.Close() // We've read it, so close it.
	bodyString := string(bodyBytes)

	// To avoid race conditions, we prioritize the more specific handler.
	// The request containing the function response is very specific.
	if strings.Contains(bodyString, "functionResponse") {
		if handler, ok := m.handlers["functionResponse"]; ok {
			recorder := httptest.NewRecorder()
			handler(recorder, req)
			return recorder.Result(), nil
		}
	}

	for key, handler := range m.handlers {
		if key == "functionResponse" {
			continue // Already handled
		}
		if strings.Contains(bodyString, key) {
			// We found a matching handler.
			recorder := httptest.NewRecorder()
			handler(recorder, req)
			return recorder.Result(), nil
		}
	}

	// No handler found for this request.
	return nil, fmt.Errorf("mockTransport: no handler registered for request body containing: %s", bodyString)
}

// setupTestWithMocks initializes the global client to use our mock transport.
// It returns a teardown function to restore the original state.
func setupTestWithMocks(t *testing.T, transport *mockTransport) func() {
	// Create a real HTTP client that uses our mock transport
	httpClient := &http.Client{
		Transport: transport,
	}

	// Store the original global client to restore it later
	originalClient := globalClient

	// Create a new genai.Client that is configured to use our HTTP client
	var err error
	globalClient, err = genai.NewClient(context.Background(), option.WithAPIKey("mock-key"), option.WithHTTPClient(httpClient))
	if err != nil {
		t.Fatalf("Failed to create mock genai client: %v", err)
	}

	// Return a teardown function
	return func() {
		globalClient.Close()
		globalClient = originalClient
	}
}

// TestAgentWithMockedAPI tests the full agent Run loop without making real API calls.
func TestAgentWithMockedAPI(t *testing.T) {
	// 1. Define the mock transport and register handlers for the conversation flow.
	transport := newMockTransport()

	// Handler for the first message: "What is the weather in Tokyo?"
	// It should respond with a function call to 'get_weather'.
	transport.RegisterHandler("weather in Tokyo", func(w http.ResponseWriter, r *http.Request) {
		response := `[{
			"candidates": [{
				"content": {
					"parts": [{
						"functionCall": {
							"name": "get_weather",
							"args": {"location": "Tokyo"}
						}
					}]
				}
			}]
		}]`
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(response))
	})

	// Handler for the second message, which will be the function response.
	// It should respond with a final text answer.
	// FIX: Use a more specific key that only exists in the second request.
	transport.RegisterHandler("functionResponse", func(w http.ResponseWriter, r *http.Request) {
		response := `[{
			"candidates": [{
				"content": {
					"parts": [{"text": "The weather in Tokyo is 15°C and rainy."}]
				}
			}]
		}]`
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(response))
	})

	// 2. Setup the test environment to use the mocks.
	teardown := setupTestWithMocks(t, transport)
	defer teardown() // Ensure we clean up after the test.

	// 3. Define the tool, same as in the integration test.
	getWeatherTool := Tool{
		Schema: &genai.FunctionDeclaration{
			Name: "get_weather", Description: "Get weather",
			Parameters: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{"location": {Type: genai.TypeString}},
				Required:   []string{"location"},
			},
		},
		Execute: func(ctx context.Context, args map[string]any) (map[string]any, error) {
			// The mock Execute function.
			return map[string]any{"temperature": "15°C", "condition": "rainy"}, nil
		},
	}

	// 4. Create and run the agent.
	weatherAgent, err := NewAgent(AgentConfig{Name: "Weather Agent", Model: "gemini-1.5-flash-latest"})
	if err != nil {
		t.Fatalf("NewAgent() failed: %v", err)
	}
	weatherAgent.RegisterTools(getWeatherTool)

	result, err := weatherAgent.Run(context.Background(), "What is the weather in Tokyo?")
	if err != nil {
		t.Fatalf("Agent.Run() failed with error: %v", err)
	}

	// 5. Assert the final result.
	expected := "The weather in Tokyo is 15°C and rainy."
	if result != expected {
		t.Errorf("Expected result '%s', but got '%s'", expected, result)
	}
}

// ToJSON is a helper to marshal structs to JSON strings for responses.
func ToJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(b)
}
